# Transformer-Based Sign Language Recognition (WLASL)

A lightweight, efficient Transformer-based Sign Language Recognition system using the WLASL dataset.

## Features

- **Lightweight Transformer Architecture**: Efficient model design optimized for low-memory systems
- **WLASL Dataset Support**: Full support for WLASL dataset with video loading and preprocessing
- **Memory-Efficient Data Pipeline**: tf.data pipeline with streaming from disk
- **Checkpointing**: Automatic model checkpointing during training
- **Inference Scripts**: Easy-to-use scripts for testing trained models
- **Dummy Test**: Quick pipeline verification on small subset
- **Configurable**: YAML configuration files for easy hyperparameter tuning

## Project Structure

```
project_root/
├── src/
│   ├── __init__.py
│   ├── data_pipeline.py      # Data loading and preprocessing
│   ├── model_builder.py      # Transformer model architecture
│   ├── train_main.py         # Main training script
│   ├── run_inference.py      # Inference on single video
│   ├── run_saved_model.py    # Test saved model on multiple videos
│   ├── dummy_test.py         # Lightweight pipeline test
│   └── utils.py              # Utility functions
├── checkpoints/              # Model checkpoints (created during training)
├── models/                   # Saved models (created during training)
├── configs/
│   ├── small_config.yaml     # Small model configuration
│   └── fusion_config.yaml    # Fusion model configuration (future)
├── dataset/                  # WLASL dataset (should contain videos and JSON files)
├── requirements.txt          # Python dependencies
└── README.md                 # This file
```

## Installation

1. **Clone the repository** (if applicable) or navigate to the project directory.

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Verify dataset structure**:
   - Ensure `dataset/videos/` contains video files (`.mp4` format)
   - Ensure `dataset/WLASL_v0.3.json` exists
   - Ensure `dataset/wlasl_class_list.txt` exists

## Quick Start

### 1. Run Dummy Test (Recommended First Step)

Verify the pipeline works correctly with a small subset:

**Using standalone script:**
```bash
python dummy_test.py --num_videos 20 --epochs 2 --batch_size 2
```

**Using module:**
```bash
python -m src.dummy_test --num_videos 20 --epochs 2 --batch_size 2
```

This should complete in < 5 minutes and verify:
- Data pipeline loads videos correctly
- Model compiles and trains without errors
- Checkpoints are created
- Predictions work

### 2. Train Model

**Full training with default config**:
```bash
python train.py --config configs/small_config.yaml --epochs 50 --batch_size 4
```

**Or using module:**
```bash
python -m src.train_main --config configs/small_config.yaml --epochs 50 --batch_size 4
```

**Training with custom parameters**:
```bash
python -m src.train_main \
    --json_path dataset/WLASL_v0.3.json \
    --video_dir dataset/videos \
    --class_list dataset/wlasl_class_list.txt \
    --batch_size 4 \
    --epochs 50 \
    --learning_rate 0.0001 \
    --checkpoint_dir checkpoints \
    --model_dir models
```

**Training with small model (for debugging)**:
```bash
python -m src.train_main --small_model --subset_size 100 --epochs 10
```

**Training with mixed precision (if GPU supports it)**:
```bash
python -m src.train_main --mixed_precision --batch_size 8
```

### 3. Run Inference

**Single video inference**:
```bash
python inference.py \
    --video_path path/to/video.mp4 \
    --model_path models/final_slr_model \
    --gloss_mapping models/gloss_to_idx.json
```

Note: The model is saved in SavedModel format (directory) by default. If you have a weights-only file, specify the path and the script will attempt to load it.

**Or using module:**
```bash
python -m src.run_inference \
    --video_path path/to/video.mp4 \
    --model_path models/final_slr_model.h5 \
    --gloss_mapping models/gloss_to_idx.json
```

**Test saved model on multiple videos**:
```bash
python -m src.run_saved_model \
    --model_path models/final_slr_model.h5 \
    --video_paths video1.mp4 video2.mp4 video3.mp4
```

## Configuration

Edit `configs/small_config.yaml` to customize:

- **Model architecture**: `embed_dim`, `num_layers`, `num_heads`, etc.
- **Training parameters**: `batch_size`, `learning_rate`, `epochs`
- **Data parameters**: `num_frames`, `frame_size`
- **Efficiency settings**: `subset_size`, `max_videos_per_class`

Example configuration:
```yaml
model:
  num_frames: 16
  frame_size: [224, 224]
  embed_dim: 128
  num_heads: 4
  num_layers: 2
  ff_dim: 256
  dropout: 0.1

training:
  batch_size: 4
  learning_rate: 0.0001
  epochs: 50
```

## Model Architecture

The model uses a lightweight Transformer architecture:

1. **CNN Backbone**: Extracts spatial features from each video frame
2. **Positional Encoding**: Adds temporal position information
3. **Transformer Blocks**: 2-4 layers of self-attention for temporal modeling
4. **Global Pooling**: Aggregates temporal features
5. **Classification Head**: Outputs class probabilities

### Model Size Options

- **Small Model** (for testing): `embed_dim=64`, `num_layers=1`, `num_heads=2`
- **Default Model**: `embed_dim=128`, `num_layers=2`, `num_heads=4`
- **Large Model**: `embed_dim=256`, `num_layers=4`, `num_heads=8` (may cause OOM on 8GB RAM)

## Memory Efficiency

The system is designed for 8 GB RAM:

- **Streaming Data Loading**: Videos are loaded on-demand, not all at once
- **Small Batch Sizes**: Default batch size is 4 (reduce to 2 if needed)
- **Lightweight Model**: Small embeddings and few layers
- **Disk Caching**: Optional disk caching to reduce RAM usage
- **Mixed Precision**: Optional FP16 training to reduce memory (if GPU supports)

### Troubleshooting Memory Issues

If you encounter Out-of-Memory (OOM) errors:

1. **Reduce batch size**: `--batch_size 2` or `--batch_size 1`
2. **Use smaller model**: `--small_model`
3. **Reduce frame size**: Edit config to `frame_size: [112, 112]`
4. **Reduce number of frames**: Edit config to `num_frames: 8`
5. **Use subset**: `--subset_size 100` for debugging

## Training Output

Training creates:

- **Checkpoints**: `checkpoints/ckpt_XX.weights.h5` (best model weights based on validation accuracy)
- **Final Model**: `models/final_slr_model/` (SavedModel format - directory)
- **Final Weights**: `models/final_slr_model.weights.h5` (weights backup)
- **Gloss Mapping**: `models/gloss_to_idx.json`
- **Training Config**: `models/training_config.yaml`
- **Logs**: `logs/training_log.csv` and TensorBoard logs in `logs/`

Note: The model uses SavedModel format (directory) instead of HDF5 because it's a subclassed model. Checkpoints save only weights for efficiency.

## Evaluation

Monitor training with TensorBoard:

```bash
tensorboard --logdir logs
```

View training metrics:
- Training/validation accuracy
- Training/validation loss
- Learning rate schedule

## Inference Output

Example inference output:

```
Predicted sign: THANK YOU
Confidence: 0.9234

Top-3 predictions:
  1. thank you: 0.9234
  2. hello: 0.0456
  3. please: 0.0310
```

## Reproducibility

The code uses fixed random seeds for reproducibility:
- NumPy: seed=42
- TensorFlow: seed=42
- Python: seed=42

Set custom seed with `--seed` argument.

## Future Extensions

- **Pose Keypoints**: MediaPipe integration for pose-based recognition
- **Fusion Models**: Combine video and pose features
- **Web UI**: Gradio/Streamlit interface for easy testing
- **Model Quantization**: Further reduce model size for deployment

## Troubleshooting

### Common Issues

1. **Video loading errors**: Ensure videos are in correct format (MP4) and paths are correct
2. **OOM errors**: Reduce batch size, use smaller model, or reduce frame size
3. **Slow training**: Reduce number of frames or use GPU if available
4. **Import errors**: Ensure all dependencies are installed: `pip install -r requirements.txt`

### Getting Help

- Check logs in `logs/training_log.csv`
- Verify dataset structure matches expected format
- Run dummy test first to verify setup: `python -m src.dummy_test`

## License

This project is for educational/research purposes.

## Citation

If you use this code, please cite the WLASL dataset:

```
@inproceedings{li2020word,
  title={Word-level deep sign language recognition from video: A new large-scale dataset and methods comparison},
  author={Li, Dongxu and Rodriguez, Cristian and Yu, Xin and Li, Hongdong},
  booktitle={Proceedings of the IEEE/CVF winter conference on applications of computer vision},
  year={2020}
}
```

