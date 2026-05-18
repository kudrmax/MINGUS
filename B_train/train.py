#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This script loads the database from the DATA.json file
and then trains MINGUS with the data
"""
import sys
import argparse
import os
import time
import math
import csv
import json
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

import B_train.loadDB as dataset
import B_train.MINGUS_model as mod

from pathlib import Path
from B_train._resume import (
    CheckpointState,
    is_phase_done,
    load_checkpoint,
    mark_phase_done,
    save_best,
    save_checkpoint,
)

# Device configuration
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(1)


def _append_epoch_row(work_dir: Path, row: dict) -> None:
    """Append one row to <work_dir>/epochs.csv. Writes header if file doesn't exist."""
    csv_path = work_dir / "epochs.csv"
    file_exists = csv_path.exists()
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["phase", "epoch", "val_loss", "val_ppl", "val_acc", "time_sec"]
        )
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


if __name__ == '__main__':

    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--COND_TYPE_PITCH', type=str, default='D-C-B-BE-O',
                    help='conditioning features for pitch model (paper-optimal: D-C-B-BE-O per §3.2)')
    parser.add_argument('--COND_TYPE_DURATION', type=str, default='B-BE-O',
                    help='conditioning features for duration model (paper-optimal: B-BE-O per §3.2)')
    parser.add_argument('--TRAIN_BATCH_SIZE', type=int, default=20,
                        help='training batch size')
    parser.add_argument('--EVAL_BATCH_SIZE', type=int, default=10,
                        help='evaluation batch size')
    parser.add_argument('--BPTT', type=int, default=35,
                    help='length of a note sequence for training')
    parser.add_argument('--EPOCHS', type=int, default=10,
                    help='epochs for training')
    parser.add_argument('--SEGMENTATION', action='store_false', default=True,
                        help='train with NO melody segmentation')
    parser.add_argument('--AUGMENTATION', action='store_true', default=False,
                        help='augment dataset')
    parser.add_argument('--augmentation_const', type=int, default=3,
                        help='how many times to augment the data')
    parser.add_argument('--work-dir', type=str, default='B_train/models',
                        help='directory for checkpoints and final artefacts (resumable)')
    args = parser.parse_args(sys.argv[1:])
    
    # Constants for MINGUS training
    print('Training summary:')
    print('-' * 80)
    print('TRAIN_BATCH_SIZE:', args.TRAIN_BATCH_SIZE)
    print('EVAL_BATCH_SIZE:', args.EVAL_BATCH_SIZE)
    print('EPOCHS:', args.EPOCHS)
    print('sequence length:', args.BPTT)
    print('SEGMENTATION:', args.SEGMENTATION)
    print('AUGMENTATION:', args.AUGMENTATION) 
    print('augmentation_const:', args.augmentation_const)
    print('pitch model conditionings:', args.COND_TYPE_PITCH)
    print('duration model conditionings:', args.COND_TYPE_DURATION)
    print('-' * 80)
    
    # LOAD DATA
    
    MusicDB = dataset.MusicDB(device, args.TRAIN_BATCH_SIZE, args.EVAL_BATCH_SIZE,
                 args.BPTT, args.AUGMENTATION, args.SEGMENTATION, args.augmentation_const)
    
    vocabPitch, vocabDuration, vocabBeat, vocabOffset = MusicDB.getVocabs()
    
    pitch_to_ix, duration_to_ix, beat_to_ix, offset_to_ix = MusicDB.getInverseVocabs()
    
    train_pitch_batched, train_duration_batched, train_chord_batched, train_next_chord_batched, train_bass_batched, train_beat_batched, train_offset_batched  = MusicDB.getTrainingData()
    val_pitch_batched, val_duration_batched, val_chord_batched, val_next_chord_batched, val_bass_batched, val_beat_batched, val_offset_batched  = MusicDB.getValidationData()
    test_pitch_batched, test_duration_batched, test_chord_batched, test_next_chord_batched, test_bass_batched, test_beat_batched, test_offset_batched  = MusicDB.getTestData()


    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    # ============ PITCH MODEL ============
    isPitch = True
    pitch_vocab_size = len(vocabPitch)
    pitch_embed_dim = 512
    duration_vocab_size = len(vocabDuration)
    duration_embed_dim = 512
    chord_encod_dim = 64
    next_chord_encod_dim = 32
    beat_vocab_size = len(vocabBeat)
    beat_embed_dim = 64
    bass_embed_dim = 64
    offset_vocab_size = len(vocabOffset)
    offset_embed_dim = 32
    emsize = 200
    nhid = 200
    nlayers = 4
    nhead = 4
    dropout = 0.2
    pitch_pad_idx = pitch_to_ix['<pad>']
    duration_pad_idx = duration_to_ix['<pad>']
    beat_pad_idx = beat_to_ix['<pad>']
    offset_pad_idx = offset_to_ix['<pad>']

    modelPitch = mod.TransformerModel(
        pitch_vocab_size, pitch_embed_dim,
        duration_vocab_size, duration_embed_dim,
        bass_embed_dim, chord_encod_dim, next_chord_encod_dim,
        beat_vocab_size, beat_embed_dim,
        offset_vocab_size, offset_embed_dim,
        emsize, nhead, nhid, nlayers,
        pitch_pad_idx, duration_pad_idx, beat_pad_idx, offset_pad_idx,
        device, dropout, isPitch, args.COND_TYPE_PITCH,
    ).to(device)

    criterion = nn.CrossEntropyLoss(ignore_index=pitch_pad_idx)
    lr = 0.5
    optimizer = torch.optim.SGD(modelPitch.parameters(), lr=lr, momentum=0.9, nesterov=True)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)

    pitch_train_time = None
    if is_phase_done(work_dir, "pitch"):
        print('Pitch phase already complete - loading best checkpoint.')
        best_model_pitch = modelPitch
        best_model_pitch.load_state_dict(torch.load(work_dir / "pitch_best.pt", map_location=device))
    else:
        # Resume from latest checkpoint if any
        ck = load_checkpoint(work_dir, "pitch", modelPitch, optimizer)
        start_epoch = (ck.epoch + 1) if ck else 1
        best_val_loss = ck.best_val_loss if ck else float("inf")
        best_model_pitch = modelPitch
        if ck:
            print(f'Resuming pitch phase from epoch {start_epoch} (best_val_loss={best_val_loss:.4f})')

        path = (f'B_train/runs/pitchModel/COND {args.COND_TYPE_PITCH} '
                f'Epochs {args.EPOCHS} Augmentation {args.AUGMENTATION}')
        # NOTE: do NOT rmtree on resume - append to existing tensorboard run
        os.makedirs(path, exist_ok=True)
        writer = SummaryWriter(path)
        step = 0

        pitch_start_time = time.time()
        print('Starting pitch model training...')
        for epoch in range(start_epoch, args.EPOCHS + 1):
            epoch_start_time = time.time()
            step = mod.train(modelPitch, vocabPitch,
                             train_pitch_batched, train_duration_batched,
                             train_chord_batched, train_next_chord_batched,
                             train_bass_batched, train_beat_batched, train_offset_batched,
                             criterion, optimizer, scheduler, epoch, args.BPTT, device,
                             writer, step, isPitch)
            val_loss, val_acc = mod.evaluate(modelPitch, pitch_to_ix,
                                             val_pitch_batched, val_duration_batched,
                                             val_chord_batched, val_next_chord_batched,
                                             val_bass_batched, val_beat_batched, val_offset_batched,
                                             criterion, args.BPTT, device, isPitch)
            writer.add_scalar('Validation accuracy', val_acc, global_step=epoch)
            epoch_time = time.time() - epoch_start_time
            print('-' * 89)
            print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
                  'valid ppl {:5.2f} | valid acc {:5.2f}'.format(
                      epoch, epoch_time,
                      val_loss, math.exp(val_loss), val_acc))
            print('-' * 89)

            # Append epoch summary to train.log for ETA monitor in Colab notebook
            with open(work_dir / 'train.log', 'a') as f:
                f.write(f'pitch end of epoch {epoch} | time: {epoch_time:.2f}s | valid loss {val_loss:.2f}\n')

            _append_epoch_row(work_dir, {
                "phase": "pitch",
                "epoch": epoch,
                "val_loss": val_loss,
                "val_ppl": math.exp(val_loss),
                "val_acc": val_acc,
                "time_sec": epoch_time,
            })

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_pitch = modelPitch
                save_best(work_dir, "pitch", modelPitch)

            scheduler.step()
            save_checkpoint(work_dir, "pitch",
                            CheckpointState(epoch=epoch, best_val_loss=best_val_loss),
                            modelPitch, optimizer)

        pitch_train_time = time.time() - pitch_start_time
        pitch_best_val_loss = best_val_loss
        writer.close()
        mark_phase_done(work_dir, "pitch")

    # Always load best for test eval (works for both fresh-completed and is_phase_done branches)
    best_model_pitch.load_state_dict(torch.load(work_dir / "pitch_best.pt", map_location=device))

    # Test the pitch model
    test_loss, test_acc = mod.evaluate(best_model_pitch, pitch_to_ix,
                                       test_pitch_batched, test_duration_batched,
                                       test_chord_batched, test_next_chord_batched,
                                       test_bass_batched, test_beat_batched, test_offset_batched,
                                       criterion, args.BPTT, device, isPitch)
    print('=' * 89)
    print('| End of pitch training | test loss {:5.2f} | test ppl {:5.2f} | test acc {:5.2f}'.format(
        test_loss, math.exp(test_loss), test_acc))
    print('=' * 89)
    pitch_test_loss = test_loss
    pitch_test_acc = test_acc

    savePATHpitch = str(work_dir / "pitchModel" / f"MINGUS COND {args.COND_TYPE_PITCH} Epochs {args.EPOCHS}.pt")
    os.makedirs(os.path.dirname(savePATHpitch), exist_ok=True)
    torch.save(best_model_pitch.state_dict(), savePATHpitch)


    # ============ DURATION MODEL ============
    isPitch = False
    pitch_embed_dim = 64
    duration_embed_dim = 64
    chord_encod_dim = 64
    next_chord_encod_dim = 32
    beat_embed_dim = 32
    bass_embed_dim = 32
    offset_embed_dim = 32
    emsize = 200
    nhid = 200
    nlayers = 4
    nhead = 4
    dropout = 0.2

    modelDuration = mod.TransformerModel(
        pitch_vocab_size, pitch_embed_dim,
        duration_vocab_size, duration_embed_dim,
        bass_embed_dim, chord_encod_dim, next_chord_encod_dim,
        beat_vocab_size, beat_embed_dim,
        offset_vocab_size, offset_embed_dim,
        emsize, nhead, nhid, nlayers,
        pitch_pad_idx, duration_pad_idx, beat_pad_idx, offset_pad_idx,
        device, dropout, isPitch, args.COND_TYPE_DURATION,
    ).to(device)

    criterion = nn.CrossEntropyLoss(ignore_index=duration_pad_idx)
    lr = 0.05
    optimizer = torch.optim.SGD(modelDuration.parameters(), lr=lr, momentum=0.9, nesterov=True)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)

    duration_train_time = None
    if is_phase_done(work_dir, "duration"):
        print('Duration phase already complete - loading best checkpoint.')
        best_model_duration = modelDuration
        best_model_duration.load_state_dict(torch.load(work_dir / "duration_best.pt", map_location=device))
    else:
        ck = load_checkpoint(work_dir, "duration", modelDuration, optimizer)
        start_epoch = (ck.epoch + 1) if ck else 1
        best_val_loss = ck.best_val_loss if ck else float("inf")
        best_model_duration = modelDuration
        if ck:
            print(f'Resuming duration phase from epoch {start_epoch} (best_val_loss={best_val_loss:.4f})')

        path = (f'B_train/runs/durationModel/COND {args.COND_TYPE_DURATION} '
                f'EPOCHS {args.EPOCHS} Augmentation {args.AUGMENTATION}')
        os.makedirs(path, exist_ok=True)
        writer = SummaryWriter(path)
        step = 0

        duration_start_time = time.time()
        print('Starting duration model training...')
        for epoch in range(start_epoch, args.EPOCHS + 1):
            epoch_start_time = time.time()
            step = mod.train(modelDuration, vocabDuration,
                             train_pitch_batched, train_duration_batched,
                             train_chord_batched, train_next_chord_batched,
                             train_bass_batched, train_beat_batched, train_offset_batched,
                             criterion, optimizer, scheduler, epoch, args.BPTT, device,
                             writer, step, isPitch)
            val_loss, val_acc = mod.evaluate(modelDuration, duration_to_ix,
                                             val_pitch_batched, val_duration_batched,
                                             val_chord_batched, val_next_chord_batched,
                                             val_bass_batched, val_beat_batched, val_offset_batched,
                                             criterion, args.BPTT, device, isPitch)
            writer.add_scalar('Validation accuracy', val_acc, global_step=epoch)
            epoch_time = time.time() - epoch_start_time
            print('-' * 89)
            print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
                  'valid ppl {:5.2f} | valid acc {:5.2f}'.format(
                      epoch, epoch_time,
                      val_loss, math.exp(val_loss), val_acc))
            print('-' * 89)

            with open(work_dir / 'train.log', 'a') as f:
                f.write(f'duration end of epoch {epoch} | time: {epoch_time:.2f}s | valid loss {val_loss:.2f}\n')

            _append_epoch_row(work_dir, {
                "phase": "duration",
                "epoch": epoch,
                "val_loss": val_loss,
                "val_ppl": math.exp(val_loss),
                "val_acc": val_acc,
                "time_sec": epoch_time,
            })

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_duration = modelDuration
                save_best(work_dir, "duration", modelDuration)

            scheduler.step()
            save_checkpoint(work_dir, "duration",
                            CheckpointState(epoch=epoch, best_val_loss=best_val_loss),
                            modelDuration, optimizer)

        duration_train_time = time.time() - duration_start_time
        duration_best_val_loss = best_val_loss
        writer.close()
        mark_phase_done(work_dir, "duration")

    # Always load best for test eval (works for both fresh-completed and is_phase_done branches)
    best_model_duration.load_state_dict(torch.load(work_dir / "duration_best.pt", map_location=device))

    # Final test
    test_loss, test_acc = mod.evaluate(best_model_duration, duration_to_ix,
                                       test_pitch_batched, test_duration_batched,
                                       test_chord_batched, test_next_chord_batched,
                                       test_bass_batched, test_beat_batched, test_offset_batched,
                                       criterion, args.BPTT, device, isPitch)
    print('=' * 89)
    print('| End of duration training | test loss {:5.2f} | test ppl {:5.2f} | test acc {:5.2f}'.format(
        test_loss, math.exp(test_loss), test_acc))
    print('=' * 89)
    duration_test_loss = test_loss
    duration_test_acc = test_acc

    savePATHduration = str(work_dir / "durationModel" / f"MINGUS COND {args.COND_TYPE_DURATION} Epochs {args.EPOCHS}.pt")
    os.makedirs(os.path.dirname(savePATHduration), exist_ok=True)
    torch.save(best_model_duration.state_dict(), savePATHduration)
    print(f'Both phases complete. Final artefacts in {work_dir}/')

    # ============ SUMMARY ============
    data_json_path = "A_preprocessData/data/DATA.json"
    with open(data_json_path) as f:
        data = json.load(f)

    pitch_n_params = sum(p.numel() for p in best_model_pitch.parameters())
    duration_n_params = sum(p.numel() for p in best_model_duration.parameters())

    def _best_val_from_checkpoint(work_dir: Path, phase: str):
        path = work_dir / f"{phase}_state.pt"
        if not path.exists():
            return None
        blob = torch.load(path, map_location="cpu")
        return blob.get("best_val_loss")

    pitch_best_val = (
        pitch_best_val_loss if pitch_train_time is not None
        else _best_val_from_checkpoint(work_dir, "pitch")
    )
    duration_best_val = (
        duration_best_val_loss if duration_train_time is not None
        else _best_val_from_checkpoint(work_dir, "duration")
    )

    summary = {
        "epochs": args.EPOCHS,
        "bptt": args.BPTT,
        "device": str(device),
        "split": {
            "train": len(data.get("train", [])),
            "val": len(data.get("validation", [])),
            "test": len(data.get("test", [])),
        },
        "pitch": {
            "test_loss": pitch_test_loss,
            "test_ppl": math.exp(pitch_test_loss),
            "test_acc": pitch_test_acc,
            "best_val_loss": pitch_best_val,
            "train_time_sec": pitch_train_time,
            "n_params": pitch_n_params,
        },
        "duration": {
            "test_loss": duration_test_loss,
            "test_ppl": math.exp(duration_test_loss),
            "test_acc": duration_test_acc,
            "best_val_loss": duration_best_val,
            "train_time_sec": duration_train_time,
            "n_params": duration_n_params,
        },
    }
    summary_path = work_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Wrote summary to {summary_path}")


