# Installing MediaPipe for Pose Extraction

MediaPipe is required for extracting pose keypoints from videos. Follow these instructions to install it.

## Quick Install

```bash
pip install mediapipe
```

Or install all requirements:

```bash
pip install -r requirements.txt
```

## Platform-Specific Notes

### Windows

MediaPipe should install directly with pip:

```bash
pip install mediapipe
```

If you encounter issues, try:

```bash
pip install --upgrade pip
pip install mediapipe
```

### Linux

```bash
pip install mediapipe
```

### macOS

On Apple Silicon (M1/M2), you may need:

```bash
pip install mediapipe-silicon
```

Or the standard installation:

```bash
pip install mediapipe
```

## Verify Installation

Test that MediaPipe is installed correctly:

```python
import mediapipe as mp
print("MediaPipe version:", mp.__version__)
```

## Troubleshooting

### Common Issues

1. **Import Error**: Make sure you're using the correct Python environment
   ```bash
   python -m pip install mediapipe
   ```

2. **Version Conflicts**: Update pip first
   ```bash
   pip install --upgrade pip
   pip install mediapipe
   ```

3. **Apple Silicon (M1/M2)**: Use the silicon-specific package
   ```bash
   pip install mediapipe-silicon
   ```

### Alternative: Skip Pose Extraction

If you can't install MediaPipe, you can:

1. Use frame embeddings instead (no MediaPipe required)
2. Use raw video frames (slower, uses more RAM)

To use frame embeddings:

```bash
python extract_features.py --split train
```

Then update config:

```yaml
efficiency:
  use_pre_extracted: true
  use_pose: false  # Use features instead

model:
  input_type: "features"
```

## After Installation

Once MediaPipe is installed, you can extract poses:

```bash
python extract_poses.py --split train
python extract_poses.py --split test
```

