# Lightweight Training Guide

This guide explains how to use the lightweight training pipeline with pre-extracted features for efficient training on low-end PCs.

## Overview

The lightweight pipeline uses pre-extracted features (pose keypoints or frame embeddings) instead of processing videos on-the-fly. This significantly reduces RAM usage and training time.

## Model Configuration

The model uses the following lightweight settings:

- **embed_dim**: 128
- **num_heads**: 4
- **depth (num_layers)**: 2
- **mlp_dim (ff_dim)**: 256
- **dropout**: 0.1

## Step 1: Extract Pose Keypoints (Recommended)

Pose keypoints are much smaller than video frames and provide good performance for sign language recognition.

### Extract poses for training set:

```bash
python extract_poses.py \
    --json_path dataset/WLASL_v0.3.json \
    --video_dir dataset/videos \
    --output_dir dataset/poses \
    --class_list dataset/wlasl_class_list.txt \
    --split train \
    --num_frames 16
```

### Extract poses for test set:

```bash
python extract_poses.py \
    --json_path dataset/WLASL_v0.3.json \
    --video_dir dataset/videos \
    --output_dir dataset/poses \
    --class_list dataset/wlasl_class_list.txt \
    --split test \
    --num_frames 16
```

### Options:

- `--split`: Filter by split ('train', 'test', or None for all)
- `--max_videos`: Limit number of videos (for testing)
- `--num_frames`: Number of frames to extract per video (default: 16)

## Step 2: Extract Frame Embeddings (Optional)

Frame embeddings are pre-extracted CNN features. They're larger than poses but smaller than raw video frames.

### Extract features for training set:

```bash
python extract_features.py \
    --json_path dataset/WLASL_v0.3.json \
    --video_dir dataset/videos \
    --output_dir dataset/features \
    --class_list dataset/wlasl_class_list.txt \
    --split train \
    --num_frames 16 \
    --embed_dim 128
```

### Extract features for test set:

```bash
python extract_features.py \
    --json_path dataset/WLASL_v0.3.json \
    --video_dir dataset/videos \
    --output_dir dataset/features \
    --class_list dataset/wlasl_class_list.txt \
    --split test \
    --num_frames 16 \
    --embed_dim 128
```

## Step 3: Configure Training

Edit `configs/small_config.yaml`:

```yaml
model:
  embed_dim: 128
  num_heads: 4
  num_layers: 2  # depth = 2
  ff_dim: 256  # mlp_dim = 256
  dropout: 0.1
  input_type: "pose"  # or "features" or "video"

efficiency:
  use_pre_extracted: true  # Enable pre-extracted features
  use_pose: true  # Use pose keypoints (or false for features)
  pose_dir: "dataset/poses"
  feature_dir: "dataset/features"
```

## Step 4: Train with Pre-extracted Features

```bash
python train.py --config configs/small_config.yaml --epochs 50
```

## Memory Efficiency

### With Pre-extracted Features:

1. **Pose Keypoints**: ~2 KB per video (16 frames × 33 keypoints × 4 values × 4 bytes)
2. **Frame Embeddings**: ~8 KB per video (16 frames × 128 dims × 4 bytes)
3. **Raw Video Frames**: ~3 MB per video (16 frames × 224×224×3 × 1 byte)

### RAM Usage:

- **Batch size 4 with poses**: ~32 KB in RAM
- **Batch size 4 with features**: ~128 KB in RAM
- **Batch size 4 with video**: ~12 MB in RAM

## Comparison

| Method | RAM per Batch (size 4) | Disk Space | Training Speed |
|--------|------------------------|------------|----------------|
| Video | ~12 MB | Original videos | Slow |
| Features | ~128 KB | ~50 GB | Medium |
| Pose | ~32 KB | ~10 GB | Fast |

## Tips

1. **Start with poses**: Pose keypoints provide good performance with minimal storage
2. **Batch size**: You can use larger batch sizes (8-16) with pre-extracted features
3. **Disk space**: Pre-extraction uses additional disk space but saves RAM
4. **Incremental extraction**: Extract features in batches if disk space is limited

## Troubleshooting

### Out of Memory:

1. Reduce batch size: `--batch_size 2`
2. Use pose keypoints instead of features
3. Reduce number of frames: `--num_frames 8`

### Slow Feature Extraction:

1. Extract in parallel (process multiple videos)
2. Use GPU for feature extraction (if available)
3. Extract only for training set first

### Missing Features:

The training script will skip videos without extracted features. Make sure to extract features for all videos in your dataset.

## Next Steps

1. Extract pose keypoints for your dataset
2. Update config to use pre-extracted features
3. Train with larger batch sizes for faster training
4. Monitor RAM usage to verify efficiency

