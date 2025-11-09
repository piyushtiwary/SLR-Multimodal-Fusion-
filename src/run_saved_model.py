"""
Script to load and test a saved model on new videos.
"""

import os
import sys
import argparse
import numpy as np
import tensorflow as tf
from tensorflow import keras
import json

from .run_inference import (
    load_model_and_mappings,
    predict_video,
    print_predictions
)
import os


def test_model_on_videos(
    model_path: str,
    video_paths: list,
    gloss_mapping_path: str = None,
    num_frames: int = 16,
    frame_size: tuple = (224, 224)
):
    """
    Test saved model on multiple videos.
    
    Args:
        model_path: Path to saved model
        video_paths: List of video paths to test
        gloss_mapping_path: Path to gloss_to_idx.json
        num_frames: Number of frames to extract
        frame_size: Target frame size
    """
    # Load model
    model, idx_to_gloss = load_model_and_mappings(
        model_path=model_path,
        gloss_mapping_path=gloss_mapping_path
    )
    
    print(f"\nTesting model on {len(video_paths)} videos...")
    print("=" * 60)
    
    # Process each video
    for i, video_path in enumerate(video_paths, 1):
        if not os.path.exists(video_path):
            print(f"\n[{i}/{len(video_paths)}] Video not found: {video_path}")
            continue
        
        print(f"\n[{i}/{len(video_paths)}] Processing: {os.path.basename(video_path)}")
        
        try:
            results = predict_video(
                model=model,
                video_path=video_path,
                num_frames=num_frames,
                frame_size=frame_size,
                idx_to_gloss=idx_to_gloss,
                top_k=3
            )
            print_predictions(results)
        except Exception as e:
            print(f"Error processing video: {e}")
    
    print("\n" + "=" * 60)
    print("Testing completed!")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Test saved model on videos')
    parser.add_argument('--model_path', type=str, default='models/final_slr_model',
                        help='Path to saved model (SavedModel directory or weights file)')
    parser.add_argument('--gloss_mapping', type=str, default='models/gloss_to_idx.json',
                        help='Path to gloss_to_idx.json')
    parser.add_argument('--video_paths', type=str, nargs='+', required=True,
                        help='Paths to video files to test')
    parser.add_argument('--num_frames', type=int, default=16,
                        help='Number of frames to extract')
    parser.add_argument('--frame_size', type=int, nargs=2, default=[224, 224],
                        help='Frame size (height width)')
    
    args = parser.parse_args()
    
    # Check if model exists
    if not os.path.exists(args.model_path):
        print(f"Error: Model file not found: {args.model_path}")
        print("Please train a model first using train_main.py")
        return
    
    # Test model
    test_model_on_videos(
        model_path=args.model_path,
        video_paths=args.video_paths,
        gloss_mapping_path=args.gloss_mapping,
        num_frames=args.num_frames,
        frame_size=tuple(args.frame_size)
    )


if __name__ == '__main__':
    main()

