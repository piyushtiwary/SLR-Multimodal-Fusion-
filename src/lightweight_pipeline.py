"""
Lightweight data pipeline for pre-extracted features.
Streams features from disk for efficient training with minimal RAM usage.
"""

import os
import numpy as np
import tensorflow as tf
from typing import Tuple, Optional, Dict, List
import json


def load_pre_extracted_features(feature_path: str) -> np.ndarray:
    """
    Load pre-extracted features from disk.
    
    Args:
        feature_path: Path to .npy file containing features
        
    Returns:
        Feature array
    """
    if not os.path.exists(feature_path):
        raise FileNotFoundError(f"Feature file not found: {feature_path}")
    
    return np.load(feature_path)


def create_lightweight_dataset(
    feature_label_pairs: List[Tuple[str, int]],
    feature_dir: str,
    batch_size: int = 4,
    shuffle: bool = True,
    prefetch: int = 2,
    use_pose: bool = False
) -> tf.data.Dataset:
    """
    Create TensorFlow dataset from pre-extracted features.
    Streams from disk with minimal RAM usage.
    
    Args:
        feature_label_pairs: List of (video_id, label) tuples
        feature_dir: Directory containing pre-extracted features
        batch_size: Batch size
        shuffle: Whether to shuffle dataset
        prefetch: Number of batches to prefetch
        use_pose: Whether features are pose keypoints (True) or frame embeddings (False)
        
    Returns:
        TensorFlow Dataset
    """
    def load_features(video_id, label):
        # Load pre-extracted features
        feature_path = os.path.join(feature_dir, f"{video_id}.npy")
        try:
            features = tf.py_function(
                func=lambda x: load_pre_extracted_features(x.numpy().decode('utf-8')),
                inp=[feature_path],
                Tout=tf.float32
            )
            
            # Set shape based on feature type
            if use_pose:
                # Pose keypoints: (num_frames, 33, 4)
                features.set_shape((None, 33, 4))
                # Flatten to (num_frames, 132) for transformer
                features = tf.reshape(features, (-1, 132))
            else:
                # Frame embeddings: (num_frames, embed_dim)
                features.set_shape((None, None))
            
            return features, label
        except Exception as e:
            # Return zeros and label -1 as error indicator
            if use_pose:
                error_features = tf.zeros((16, 132), dtype=tf.float32)
            else:
                error_features = tf.zeros((16, 128), dtype=tf.float32)
            return error_features, tf.constant(-1, dtype=tf.int32)
    
    # Create dataset
    video_ids = [pair[0] for pair in feature_label_pairs]
    labels = [pair[1] for pair in feature_label_pairs]
    
    dataset = tf.data.Dataset.from_tensor_slices((video_ids, labels))
    
    # Load features
    dataset = dataset.map(
        load_features,
        num_parallel_calls=tf.data.AUTOTUNE,
        deterministic=not shuffle
    )
    
    # Filter out error cases (label == -1)
    dataset = dataset.filter(lambda x, y: y >= 0)
    
    if shuffle:
        dataset = dataset.shuffle(buffer_size=min(1000, len(feature_label_pairs)))
    
    # Batch
    dataset = dataset.batch(batch_size, drop_remainder=False)
    
    # Prefetch
    dataset = dataset.prefetch(prefetch)
    
    return dataset


def create_lightweight_train_val_datasets(
    json_path: str,
    feature_dir: str,
    video_dir: str,
    class_list_path: Optional[str] = None,
    train_split: str = 'train',
    val_split: str = 'test',
    batch_size: int = 4,
    val_batch_size: Optional[int] = None,
    use_pose: bool = False,
    subset_size: Optional[int] = None,
    max_videos_per_class: Optional[int] = None
) -> Tuple[tf.data.Dataset, tf.data.Dataset, Dict]:
    """
    Create training and validation datasets from pre-extracted features.
    
    Args:
        json_path: Path to WLASL JSON file
        feature_dir: Directory containing pre-extracted features
        video_dir: Directory containing videos (used to get video IDs)
        class_list_path: Path to class list file
        train_split: Split name for training ('train')
        val_split: Split name for validation ('test')
        batch_size: Training batch size
        val_batch_size: Validation batch size (defaults to batch_size)
        use_pose: Whether features are pose keypoints
        subset_size: Limit total videos (for debugging)
        max_videos_per_class: Limit videos per class (for debugging)
        
    Returns:
        Tuple of (train_dataset, val_dataset, gloss_to_idx)
    """
    from .utils import parse_wlasl_data
    
    # Parse training data - use video_dir to get video paths and IDs
    train_pairs, gloss_to_idx = parse_wlasl_data(
        json_path=json_path,
        video_dir=video_dir,
        class_list_path=class_list_path,
        split=train_split,
        max_videos_per_class=max_videos_per_class,
        subset_size=subset_size
    )
    
    # Convert video paths to video IDs and filter to only include videos with extracted features
    train_feature_pairs = []
    for video_path, label in train_pairs:
        video_id = os.path.splitext(os.path.basename(video_path))[0]
        feature_path = os.path.join(feature_dir, f"{video_id}.npy")
        if os.path.exists(feature_path):
            train_feature_pairs.append((video_id, label))
    
    print(f"Found {len(train_feature_pairs)} training videos with extracted features")
    
    # Parse validation data
    val_pairs, _ = parse_wlasl_data(
        json_path=json_path,
        video_dir=video_dir,
        class_list_path=class_list_path,
        split=val_split,
        max_videos_per_class=max_videos_per_class,
        subset_size=subset_size // 4 if subset_size else None
    )
    
    # Convert video paths to video IDs
    val_feature_pairs = []
    for video_path, label in val_pairs:
        video_id = os.path.splitext(os.path.basename(video_path))[0]
        feature_path = os.path.join(feature_dir, f"{video_id}.npy")
        if os.path.exists(feature_path):
            val_feature_pairs.append((video_id, label))
    
    print(f"Found {len(val_feature_pairs)} validation videos with extracted features")
    
    # Create datasets
    train_dataset = create_lightweight_dataset(
        feature_label_pairs=train_feature_pairs,
        feature_dir=feature_dir,
        batch_size=batch_size,
        shuffle=True,
        prefetch=2,
        use_pose=use_pose
    )
    
    val_dataset = create_lightweight_dataset(
        feature_label_pairs=val_feature_pairs,
        feature_dir=feature_dir,
        batch_size=val_batch_size or batch_size,
        shuffle=False,
        prefetch=1,
        use_pose=use_pose
    )
    
    return train_dataset, val_dataset, gloss_to_idx

