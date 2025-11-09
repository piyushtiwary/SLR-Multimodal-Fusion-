# Project Summary

## Overview

This project implements an efficient Transformer-based Sign Language Recognition (SLR) system using the WLASL dataset. The system is designed to run on low-end PCs with 8 GB RAM.

## Project Structure

```
project_root/
├── src/                          # Source code
│   ├── __init__.py              # Package initialization
│   ├── utils.py                 # Utility functions (data loading, config, etc.)
│   ├── data_pipeline.py         # Video loading and tf.data pipeline
│   ├── model_builder.py         # Transformer model architecture
│   ├── train_main.py            # Main training script
│   ├── run_inference.py         # Inference on single video
│   ├── run_saved_model.py       # Test saved model on multiple videos
│   └── dummy_test.py            # Lightweight pipeline test
├── configs/                      # Configuration files
│   ├── small_config.yaml        # Small model configuration
│   └── fusion_config.yaml       # Fusion model configuration (future)
├── checkpoints/                  # Model checkpoints (created during training)
├── models/                       # Saved models (created during training)
├── dataset/                      # WLASL dataset
│   ├── videos/                  # Video files
│   ├── WLASL_v0.3.json         # Annotation file
│   └── wlasl_class_list.txt    # Class list
├── train.py                     # Standalone training script
├── inference.py                 # Standalone inference script
├── dummy_test.py                # Standalone dummy test script
├── validate_setup.py            # Setup validation script
├── requirements.txt             # Python dependencies
├── README.md                    # Comprehensive documentation
└── .gitignore                   # Git ignore rules
```

## Key Features

### 1. Data Pipeline (`src/data_pipeline.py`)
- Loads videos from WLASL dataset
- Extracts frames with temporal sampling
- Applies lightweight augmentations (brightness, contrast, temporal crop)
- Creates efficient tf.data pipeline with prefetching
- Memory-efficient streaming from disk

### 2. Model Architecture (`src/model_builder.py`)
- **CNN Backbone**: Lightweight 3-layer CNN for spatial feature extraction
- **Positional Encoding**: Temporal position encoding for sequences
- **Transformer Blocks**: 2-4 layers of self-attention
- **Classification Head**: Dense layers for class prediction
- **Configurable**: Easy to scale down for low-memory systems

### 3. Training (`src/train_main.py`)
- Supports YAML configuration files
- Model checkpointing (saves best model)
- Early stopping and learning rate scheduling
- Mixed precision training (optional)
- TensorBoard logging
- CSV logging

### 4. Inference (`src/run_inference.py`)
- Single video prediction
- Top-K predictions with confidence scores
- Loads saved models and gloss mappings

### 5. Dummy Test (`src/dummy_test.py`)
- Quick pipeline verification
- Uses small model (1 layer, 64 embed_dim)
- Tests on 20 videos for 2 epochs
- Completes in < 5 minutes

## Usage Examples

### 1. Validate Setup
```bash
python validate_setup.py
```

### 2. Run Dummy Test
```bash
python dummy_test.py --num_videos 20 --epochs 2
```

### 3. Train Model
```bash
python train.py --config configs/small_config.yaml --epochs 50 --batch_size 4
```

### 4. Run Inference
```bash
python inference.py --video_path dataset/videos/12345.mp4
```

## Configuration

Edit `configs/small_config.yaml` to customize:

- **Model**: `embed_dim`, `num_layers`, `num_heads`, `ff_dim`
- **Training**: `batch_size`, `learning_rate`, `epochs`
- **Data**: `num_frames`, `frame_size`
- **Efficiency**: `subset_size`, `max_videos_per_class`

## Memory Efficiency

The system is optimized for 8 GB RAM:

1. **Streaming Data Loading**: Videos loaded on-demand
2. **Small Batch Sizes**: Default batch size is 4
3. **Lightweight Model**: Small embeddings (128) and few layers (2)
4. **No Caching**: Dataset not cached to memory by default
5. **Mixed Precision**: Optional FP16 training

## Model Architecture Details

### Input
- Shape: `(batch, num_frames, height, width, channels)`
- Default: `(batch, 16, 224, 224, 3)`

### CNN Backbone
- 3 Conv2D layers with BatchNormalization
- GlobalAveragePooling2D
- Dense layer to `embed_dim`

### Transformer
- Positional encoding
- 2-4 Transformer blocks
- Multi-head self-attention
- Feed-forward network

### Output
- Shape: `(batch, num_classes)`
- Softmax activation

## Training Output

After training, you'll find:

- `checkpoints/ckpt_XX.h5`: Best model checkpoints
- `models/final_slr_model.h5`: Final trained model
- `models/gloss_to_idx.json`: Gloss to index mapping
- `models/training_config.yaml`: Training configuration
- `logs/training_log.csv`: Training metrics
- `logs/`: TensorBoard logs

## Troubleshooting

### Out of Memory (OOM)
1. Reduce batch size: `--batch_size 2`
2. Use smaller model: `--small_model`
3. Reduce frame size: Edit config to `[112, 112]`
4. Reduce frames: Edit config to `num_frames: 8`

### Slow Training
1. Use GPU if available
2. Reduce number of frames
3. Use smaller frame size
4. Enable mixed precision (if GPU supports)

### Video Loading Errors
1. Check video file paths
2. Verify video format (MP4)
3. Check file permissions

## Next Steps

1. **Run dummy test** to verify setup
2. **Train on small subset** for debugging
3. **Train full model** on complete dataset
4. **Evaluate** on test set
5. **Run inference** on new videos

## Future Enhancements

- [ ] MediaPipe pose extraction
- [ ] Fusion model (video + pose)
- [ ] Web UI (Gradio/Streamlit)
- [ ] Model quantization
- [ ] Real-time inference
- [ ] Multi-modal fusion

## Dependencies

See `requirements.txt` for full list. Key dependencies:

- TensorFlow >= 2.10.0
- NumPy >= 1.21.0
- OpenCV >= 4.5.0
- PyYAML >= 5.4.0

## License

Educational/Research purposes.

## Citation

If using WLASL dataset:
```
@inproceedings{li2020word,
  title={Word-level deep sign language recognition from video: A new large-scale dataset and methods comparison},
  author={Li, Dongxu and Rodriguez, Cristian and Yu, Xin and Li, Hongdong},
  booktitle={Proceedings of the IEEE/CVF winter conference on applications of computer vision},
  year={2020}
}
```

