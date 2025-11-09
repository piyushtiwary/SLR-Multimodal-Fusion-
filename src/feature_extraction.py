"""
Feature extraction for Sign Language Recognition.
Pre-extracts frame embeddings from videos for efficient training.
"""

import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tqdm import tqdm
import json
from typing import Tuple, Optional


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
        return np.zeros((num_frames, frame_size[0], frame_size[1], 3), dtype=np.uint8)
    
    # Calculate frame indices to sample
    if total_frames <= num_frames * temporal_stride:
        indices = list(range(total_frames))
    else:
        step = max(1, total_frames // (num_frames * temporal_stride))
        indices = list(range(0, total_frames, step))[:num_frames * temporal_stride]
        indices = indices[::temporal_stride][:num_frames]
    
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frame = cv2.resize(frame, (frame_size[1], frame_size[0]))
            frames.append(frame)
        else:
            if frames:
                frames.append(frames[-1])
            else:
                frames.append(np.zeros((frame_size[0], frame_size[1], 3), dtype=np.uint8))
    
    cap.release()
    
    # Pad if needed
    while len(frames) < num_frames:
        if frames:
            frames.append(frames[-1])
        else:
            frames.append(np.zeros((frame_size[0], frame_size[1], 3), dtype=np.uint8))
    
    frames = frames[:num_frames]
    return np.array(frames, dtype=np.uint8)


def create_feature_extractor(embed_dim: int = 128):
    """
    Create a lightweight CNN feature extractor.
    
    Args:
        embed_dim: Embedding dimension
        
    Returns:
        Keras model for feature extraction
    """
    inputs = keras.Input(shape=(224, 224, 3))
    
    # Normalize
    x = keras.layers.Rescaling(1.0 / 255.0)(inputs)
    
    # Lightweight CNN
    x = keras.layers.Conv2D(32, 3, strides=2, padding='same', activation='relu')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Conv2D(64, 3, strides=2, padding='same', activation='relu')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Conv2D(128, 3, strides=2, padding='same', activation='relu')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dense(embed_dim, activation='relu')(x)
    
    model = keras.Model(inputs=inputs, outputs=x, name='feature_extractor')
    return model


def extract_features_for_dataset(
    json_path: str,
    video_dir: str,
    output_dir: str,
    class_list_path: Optional[str] = None,
    num_frames: int = 16,
    frame_size: Tuple[int, int] = (224, 224),
    embed_dim: int = 128,
    max_videos: Optional[int] = None,
    split: Optional[str] = None,
    batch_size: int = 32
):
    """
    Extract frame embeddings for all videos in the dataset.
    
    Args:
        json_path: Path to WLASL JSON file
        video_dir: Directory containing videos
        output_dir: Directory to save extracted features
        class_list_path: Path to class list file
        num_frames: Number of frames to extract per video
        frame_size: Target frame size
        embed_dim: Embedding dimension
        max_videos: Maximum number of videos to process
        split: Filter by split ('train', 'test', or None)
        batch_size: Batch size for feature extraction
    """
    from .utils import parse_wlasl_data
    
    # Parse dataset
    video_label_pairs, gloss_to_idx = parse_wlasl_data(
        json_path=json_path,
        video_dir=video_dir,
        class_list_path=class_list_path,
        split=split,
        subset_size=max_videos
    )
    
    print(f"Found {len(video_label_pairs)} videos to process")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Create feature extractor
    feature_extractor = create_feature_extractor(embed_dim=embed_dim)
    
    # Extract features for each video
    success_count = 0
    fail_count = 0
    
    for video_path, label in tqdm(video_label_pairs, desc="Extracting features"):
        try:
            # Get video ID from path
            video_id = os.path.splitext(os.path.basename(video_path))[0]
            
            # Load frames
            frames = load_video_frames(
                video_path=video_path,
                num_frames=num_frames,
                frame_size=frame_size
            )
            
            # Preprocess frames
            frames_tensor = tf.convert_to_tensor(frames, dtype=tf.float32)
            frames_tensor = frames_tensor / 255.0
            
            # Extract features for each frame
            frame_features = []
            for frame in frames_tensor:
                frame_batch = tf.expand_dims(frame, 0)
                features = feature_extractor(frame_batch, training=False)
                frame_features.append(features[0].numpy())
            
            frame_features = np.array(frame_features, dtype=np.float32)
            
            # Save features
            output_path = os.path.join(output_dir, f"{video_id}.npy")
            np.save(output_path, frame_features)
            
            success_count += 1
            
        except Exception as e:
            print(f"\nError processing {video_path}: {e}")
            fail_count += 1
            continue
    
    # Save metadata
    metadata = {
        'num_videos': len(video_label_pairs),
        'success_count': success_count,
        'fail_count': fail_count,
        'num_frames': num_frames,
        'frame_size': frame_size,
        'embed_dim': embed_dim,
        'gloss_to_idx': gloss_to_idx
    }
    
    metadata_path = os.path.join(output_dir, 'metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\nFeature extraction completed!")
    print(f"Success: {success_count}, Failed: {fail_count}")
    print(f"Metadata saved to {metadata_path}")


def main():
    """Main entry point for feature extraction."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Extract frame embeddings from WLASL videos')
    parser.add_argument('--json_path', type=str, default='dataset/WLASL_v0.3.json',
                        help='Path to WLASL JSON file')
    parser.add_argument('--video_dir', type=str, default='dataset/videos',
                        help='Directory containing videos')
    parser.add_argument('--output_dir', type=str, default='dataset/features',
                        help='Directory to save extracted features')
    parser.add_argument('--class_list', type=str, default='dataset/wlasl_class_list.txt',
                        help='Path to class list file')
    parser.add_argument('--num_frames', type=int, default=16,
                        help='Number of frames to extract per video')
    parser.add_argument('--frame_size', type=int, nargs=2, default=[224, 224],
                        help='Frame size (height width)')
    parser.add_argument('--embed_dim', type=int, default=128,
                        help='Embedding dimension')
    parser.add_argument('--max_videos', type=int, default=None,
                        help='Maximum number of videos to process')
    parser.add_argument('--split', type=str, default=None,
                        help='Filter by split (train, test, or None)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for feature extraction')
    
    args = parser.parse_args()
    
    extract_features_for_dataset(
        json_path=args.json_path,
        video_dir=args.video_dir,
        output_dir=args.output_dir,
        class_list_path=args.class_list,
        num_frames=args.num_frames,
        frame_size=tuple(args.frame_size),
        embed_dim=args.embed_dim,
        max_videos=args.max_videos,
        split=args.split,
        batch_size=args.batch_size
    )


if __name__ == '__main__':
    main()

