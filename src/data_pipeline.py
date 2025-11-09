"""
Data pipeline for loading and preprocessing WLASL videos.
"""

import os
import numpy as np
import tensorflow as tf
from typing import Tuple, Optional, Dict
import cv2


def load_video_frames(
    video_path: str,
    num_frames: int = 16,
    frame_size: Tuple[int, int] = (224, 224),
    temporal_stride: int = 1
) -> np.ndarray:
    """
    Load and preprocess video frames.
    
    Args:
        video_path: Path to video file
        num_frames: Number of frames to extract
        frame_size: Target frame size (height, width)
        temporal_stride: Stride for temporal sampling
        
    Returns:
        Array of shape (num_frames, height, width, 3)
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    
    frames = []
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if total_frames == 0:
        cap.release()
        # Return black frames if video is empty
        return np.zeros((num_frames, frame_size[0], frame_size[1], 3), dtype=np.uint8)
    
    # Calculate frame indices to sample
    if total_frames <= num_frames * temporal_stride:
        # If video is shorter than needed, sample with stride 1
        indices = list(range(total_frames))
    else:
        # Sample evenly spaced frames
        step = max(1, total_frames // (num_frames * temporal_stride))
        indices = list(range(0, total_frames, step))[:num_frames * temporal_stride]
        indices = indices[::temporal_stride][:num_frames]
    
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            # Resize frame
            frame = cv2.resize(frame, (frame_size[1], frame_size[0]))
            frames.append(frame)
        else:
            # Pad with last frame if needed
            if frames:
                frames.append(frames[-1])
            else:
                frames.append(np.zeros((frame_size[0], frame_size[1], 3), dtype=np.uint8))
    
    cap.release()
    
    # Pad if we don't have enough frames
    while len(frames) < num_frames:
        if frames:
            frames.append(frames[-1])
        else:
            frames.append(np.zeros((frame_size[0], frame_size[1], 3), dtype=np.uint8))
    
    frames = frames[:num_frames]
    return np.array(frames, dtype=np.uint8)


@tf.function
def preprocess_video(frames: tf.Tensor, augment: bool = False) -> tf.Tensor:
    """
    Preprocess video frames for model input.
    
    Args:
        frames: Tensor of shape (num_frames, height, width, 3)
        augment: Whether to apply augmentations
        
    Returns:
        Preprocessed frames tensor
    """
    # Convert to float32 and normalize to [0, 1]
    frames = tf.cast(frames, tf.float32) / 255.0
    
    if augment:
        # Random brightness
        frames = tf.image.random_brightness(frames, max_delta=0.1)
        
        # Random contrast
        frames = tf.image.random_contrast(frames, lower=0.9, upper=1.1)
        
        # Random horizontal flip (with low probability to preserve sign meaning)
        if tf.random.uniform([]) > 0.9:
            frames = tf.image.flip_left_right(frames)
    
    # Normalize to ImageNet mean/std if needed (optional)
    # frames = tf.image.per_image_standardization(frames)
    
    return frames


@tf.function
def temporal_crop(frames: tf.Tensor, crop_length: int) -> tf.Tensor:
    """
    Randomly crop a temporal segment from video.
    
    Args:
        frames: Tensor of shape (num_frames, ...)
        crop_length: Length of crop
        
    Returns:
        Cropped frames tensor
    """
    num_frames = tf.shape(frames)[0]
    max_start = tf.maximum(0, num_frames - crop_length)
    start_idx = tf.random.uniform([], 0, max_start + 1, dtype=tf.int32)
    return frames[start_idx:start_idx + crop_length]


def create_dataset(
    video_label_pairs: list,
    batch_size: int = 4,
    num_frames: int = 16,
    frame_size: Tuple[int, int] = (224, 224),
    augment: bool = False,
    shuffle: bool = True,
    cache: bool = False,
    prefetch: int = 2
) -> tf.data.Dataset:
    """
    Create TensorFlow dataset from video-label pairs.
    
    Args:
        video_label_pairs: List of (video_path, label) tuples
        batch_size: Batch size
        num_frames: Number of frames per video
        frame_size: Target frame size (height, width)
        augment: Whether to apply augmentations
        shuffle: Whether to shuffle dataset
        cache: Whether to cache dataset to disk
        prefetch: Number of batches to prefetch
        
    Returns:
        TensorFlow Dataset
    """
    def generator():
        for video_path, label in video_label_pairs:
            try:
                frames = load_video_frames(
                    video_path.numpy().decode('utf-8') if isinstance(video_path, tf.Tensor) else video_path,
                    num_frames=num_frames,
                    frame_size=frame_size
                )
                yield frames, label
            except Exception as e:
                print(f"Error loading video {video_path}: {e}")
                # Yield zero frames and label -1 as error indicator
                yield np.zeros((num_frames, frame_size[0], frame_size[1], 3), dtype=np.uint8), -1
    
    # Create dataset from generator
    video_paths = [pair[0] for pair in video_label_pairs]
    labels = [pair[1] for pair in video_label_pairs]
    
    dataset = tf.data.Dataset.from_tensor_slices((video_paths, labels))
    
    def load_and_preprocess(video_path, label):
        # Load video using py_function (since cv2 is not TF-native)
        frames = tf.py_function(
            func=lambda x: load_video_frames(
                x.numpy().decode('utf-8'),
                num_frames=num_frames,
                frame_size=frame_size
            ),
            inp=[video_path],
            Tout=tf.uint8
        )
        frames.set_shape((num_frames, frame_size[0], frame_size[1], 3))
        
        # Preprocess
        frames = preprocess_video(frames, augment=augment)
        
        # Filter out error cases (label == -1)
        return frames, label
    
    dataset = dataset.map(
        load_and_preprocess,
        num_parallel_calls=tf.data.AUTOTUNE,
        deterministic=not augment
    )
    
    # Filter out error cases
    dataset = dataset.filter(lambda x, y: y >= 0)
    
    if shuffle:
        dataset = dataset.shuffle(buffer_size=min(1000, len(video_label_pairs)))
    
    if cache:
        dataset = dataset.cache()
    
    dataset = dataset.batch(batch_size, drop_remainder=False)
    dataset = dataset.prefetch(prefetch)
    
    return dataset


def create_train_val_datasets(
    json_path: str,
    video_dir: str,
    class_list_path: Optional[str] = None,
    train_split: str = 'train',
    val_split: str = 'test',
    batch_size: int = 4,
    num_frames: int = 16,
    frame_size: Tuple[int, int] = (224, 224),
    val_batch_size: Optional[int] = None,
    subset_size: Optional[int] = None,
    max_videos_per_class: Optional[int] = None
) -> Tuple[tf.data.Dataset, tf.data.Dataset, Dict]:
    """
    Create training and validation datasets from WLASL data.
    
    Args:
        json_path: Path to WLASL JSON file
        video_dir: Directory containing videos
        class_list_path: Path to class list file
        train_split: Split name for training ('train')
        val_split: Split name for validation ('test')
        batch_size: Training batch size
        num_frames: Number of frames per video
        frame_size: Target frame size
        val_batch_size: Validation batch size (defaults to batch_size)
        subset_size: Limit total videos (for debugging)
        max_videos_per_class: Limit videos per class (for debugging)
        
    Returns:
        Tuple of (train_dataset, val_dataset, gloss_to_idx)
    """
    # Import here to avoid circular imports
    from .utils import parse_wlasl_data
    
    # Parse training data
    train_pairs, gloss_to_idx = parse_wlasl_data(
        json_path=json_path,
        video_dir=video_dir,
        class_list_path=class_list_path,
        split=train_split,
        max_videos_per_class=max_videos_per_class,
        subset_size=subset_size
    )
    
    print(f"Found {len(train_pairs)} training videos")
    
    # Parse validation data
    val_pairs, _ = parse_wlasl_data(
        json_path=json_path,
        video_dir=video_dir,
        class_list_path=class_list_path,
        split=val_split,
        max_videos_per_class=max_videos_per_class,
        subset_size=subset_size // 4 if subset_size else None  # Use smaller val set
    )
    
    print(f"Found {len(val_pairs)} validation videos")
    
    # Create datasets
    train_dataset = create_dataset(
        video_label_pairs=train_pairs,
        batch_size=batch_size,
        num_frames=num_frames,
        frame_size=frame_size,
        augment=True,
        shuffle=True,
        cache=False,  # Don't cache to disk for memory efficiency
        prefetch=2
    )
    
    val_dataset = create_dataset(
        video_label_pairs=val_pairs,
        batch_size=val_batch_size or batch_size,
        num_frames=num_frames,
        frame_size=frame_size,
        augment=False,
        shuffle=False,
        cache=False,
        prefetch=1
    )
    
    return train_dataset, val_dataset, gloss_to_idx

