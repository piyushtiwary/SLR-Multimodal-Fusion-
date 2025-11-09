# Quick Start: Pose Extraction

## Why is it slow?

Pose extraction processes each video frame-by-frame using MediaPipe, which is CPU-intensive. For a large dataset:

- **~8,300 training videos** × **~2-5 seconds per video** = **4-12 hours**
- **~1,400 test videos** × **~2-5 seconds per video** = **1-2 hours**

## Recommended Approach

### Step 1: Test with Small Subset

**Always test with a small number first:**

```bash
python extract_poses.py --split train --max_videos 10
```

This should complete in 1-2 minutes and verify everything works.

### Step 2: Extract Poses in Batches

For the full dataset, consider extracting in smaller batches:

```bash
# First 100 videos
python extract_poses.py --split train --max_videos 100

# Next 100 videos (you'll need to modify the script or use a different approach)
# Or just let it run for the full dataset
```

### Step 3: Monitor Progress

The script shows progress with `tqdm`. You should see:

```
Extracting poses: 45%|████▌     | 450/1000 [15:23<18:45, 2.34s/video]
```

### Step 4: Run in Background (Optional)

On Linux/Mac, you can run in background:

```bash
nohup python extract_poses.py --split train > pose_extraction.log 2>&1 &
```

On Windows PowerShell, use `Start-Process`:

```powershell
Start-Process python -ArgumentList "extract_poses.py --split train" -NoNewWindow
```

## Time Estimates

| Videos | Estimated Time |
|--------|----------------|
| 10     | 1-2 minutes    |
| 100    | 10-15 minutes  |
| 1,000  | 1-2 hours      |
| 8,300  | 4-12 hours     |

## If It's Really Stuck

1. **Check if process is running:**
   ```powershell
   Get-Process python
   ```

2. **Check if files are being created:**
   ```powershell
   (Get-ChildItem dataset\poses\*.npy).Count
   ```

3. **Check CPU usage:**
   - If CPU is at 100%, it's working
   - If CPU is at 0%, it might be stuck

4. **Stop and restart:**
   - Press `Ctrl+C` to stop
   - Check for error messages
   - Restart with `--max_videos` to test

## Faster Alternative: Use Frame Embeddings

If pose extraction is too slow, use frame embeddings instead (faster but uses more disk space):

```bash
python extract_features.py --split train --max_videos 10
```

Frame embeddings are faster because they use a lightweight CNN instead of MediaPipe.

## Tips

1. **Start small**: Always test with `--max_videos 10` first
2. **Be patient**: Full extraction takes hours
3. **Monitor progress**: Check the progress bar
4. **Use batches**: Extract in smaller chunks if needed
5. **Check disk space**: Poses use ~2 KB per video (~20 MB for 10K videos)

