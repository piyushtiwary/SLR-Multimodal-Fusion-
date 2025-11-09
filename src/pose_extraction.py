"""
Pose keypoint extraction using MediaPipe for Sign Language Recognition.
Pre-extracts and saves pose keypoints to disk for efficient training.
"""

import os
import cv2
import numpy as np
from tqdm import tqdm
import json
from typing import List, Tuple, Optional
import pickle

# Try to import MediaPipe, provide helpful error if not available
try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False
    print("Warning: MediaPipe is not installed.")
    print("Please install it using: pip install mediapipe")
    print("Or install all requirements: pip install -r requirements.txt")


class PoseExtractor:
    """Extract pose keypoints from videos using MediaPipe."""
    
    def __init__(self):
        """Initialize MediaPipe pose solution."""
        if not MEDIAPIPE_AVAILABLE:
            raise ImportError(
                "MediaPipe is required for pose extraction. "
                "Please install it using: pip install mediapipe"
            )
        
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,  # 0, 1, or 2 (1 is balanced)
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils
    
    def extract_pose_from_frame(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """
        Extract pose keypoints from a single frame.
        
        Args:
            frame: Input frame (BGR format)
            
        Returns:
            Pose keypoints array of shape (33, 4) or None if no pose detected
            Each keypoint has [x, y, z, visibility]
        """
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process frame
        results = self.pose.process(rgb_frame)
        
        if results.pose_landmarks:
            # Extract keypoints
            keypoints = []
            for landmark in results.pose_landmarks.landmark:
                keypoints.append([
                    landmark.x,
                    landmark.y,
                    landmark.z,
                    landmark.visibility
                ])
            return np.array(keypoints, dtype=np.float32)
        else:
            return None
    
    def extract_pose_from_video(
        self,
        video_path: str,
        num_frames: int = 16,
        frame_size: Tuple[int, int] = (224, 224),
        temporal_stride: int = 1
    ) -> np.ndarray:
        """
        Extract pose keypoints from a video.
        
        Args:
            video_path: Path to video file
            num_frames: Number of frames to extract
            frame_size: Target frame size (height, width)
            temporal_stride: Stride for temporal sampling
            
        Returns:
            Pose keypoints array of shape (num_frames, 33, 4)
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames == 0:
            cap.release()
            # Return zero keypoints if video is empty
            return np.zeros((num_frames, 33, 4), dtype=np.float32)
        
        # Calculate frame indices to sample
        if total_frames <= num_frames * temporal_stride:
            indices = list(range(total_frames))
        else:
            step = max(1, total_frames // (num_frames * temporal_stride))
            indices = list(range(0, total_frames, step))[:num_frames * temporal_stride]
            indices = indices[::temporal_stride][:num_frames]
        
        pose_keypoints = []
        last_valid_keypoints = None
        
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                # Resize frame
                frame = cv2.resize(frame, (frame_size[1], frame_size[0]))
                
                # Extract pose
                keypoints = self.extract_pose_from_frame(frame)
                if keypoints is not None:
                    pose_keypoints.append(keypoints)
                    last_valid_keypoints = keypoints
                else:
                    # Use last valid keypoints or zeros
                    if last_valid_keypoints is not None:
                        pose_keypoints.append(last_valid_keypoints)
                    else:
                        pose_keypoints.append(np.zeros((33, 4), dtype=np.float32))
            else:
                # Pad with last valid keypoints or zeros
                if last_valid_keypoints is not None:
                    pose_keypoints.append(last_valid_keypoints)
                else:
                    pose_keypoints.append(np.zeros((33, 4), dtype=np.float32))
        
        cap.release()
        
        # Pad if we don't have enough frames
        while len(pose_keypoints) < num_frames:
            if pose_keypoints:
                pose_keypoints.append(pose_keypoints[-1])
            else:
                pose_keypoints.append(np.zeros((33, 4), dtype=np.float32))
        
        pose_keypoints = pose_keypoints[:num_frames]
        return np.array(pose_keypoints, dtype=np.float32)
    
    def __del__(self):
        """Clean up MediaPipe pose solution."""
        if hasattr(self, 'pose'):
            self.pose.close()


def extract_poses_for_dataset(
    json_path: str,
    video_dir: str,
    output_dir: str,
    class_list_path: Optional[str] = None,
    num_frames: int = 16,
    frame_size: Tuple[int, int] = (224, 224),
    max_videos: Optional[int] = None,
    split: Optional[str] = None
):
    """
    Extract pose keypoints for all videos in the dataset.
    
    Args:
        json_path: Path to WLASL JSON file
        video_dir: Directory containing videos
        output_dir: Directory to save extracted poses
        class_list_path: Path to class list file
        num_frames: Number of frames to extract per video
        frame_size: Target frame size
        max_videos: Maximum number of videos to process (for testing)
        split: Filter by split ('train', 'test', or None)
    """
    if not MEDIAPIPE_AVAILABLE:
        raise ImportError(
            "MediaPipe is required for pose extraction. "
            "Please install it using: pip install mediapipe"
        )
    
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
    
    # Initialize pose extractor
    pose_extractor = PoseExtractor()
    
    # Extract poses for each video
    success_count = 0
    fail_count = 0
    
    print(f"\nStarting pose extraction for {len(video_label_pairs)} videos...")
    print("This may take a while. Progress will be shown below.\n")
    
    for idx, (video_path, label) in enumerate(tqdm(video_label_pairs, desc="Extracting poses", unit="video"), 1):
        try:
            # Get video ID from path
            video_id = os.path.splitext(os.path.basename(video_path))[0]
            
            # Extract pose keypoints (this is the slow part)
            pose_keypoints = pose_extractor.extract_pose_from_video(
                video_path=video_path,
                num_frames=num_frames,
                frame_size=frame_size
            )
            
            # Save pose keypoints
            output_path = os.path.join(output_dir, f"{video_id}.npy")
            np.save(output_path, pose_keypoints)
            
            success_count += 1
            
            # Print progress every 10 videos
            if idx % 10 == 0:
                print(f"Processed {idx}/{len(video_label_pairs)} videos ({success_count} success, {fail_count} failed)")
            
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
        'gloss_to_idx': gloss_to_idx
    }
    
    metadata_path = os.path.join(output_dir, 'metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\nPose extraction completed!")
    print(f"Success: {success_count}, Failed: {fail_count}")
    print(f"Metadata saved to {metadata_path}")


def main():
    """Main entry point for pose extraction."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Extract pose keypoints from WLASL videos')
    parser.add_argument('--json_path', type=str, default='dataset/WLASL_v0.3.json',
                        help='Path to WLASL JSON file')
    parser.add_argument('--video_dir', type=str, default='dataset/videos',
                        help='Directory containing videos')
    parser.add_argument('--output_dir', type=str, default='dataset/poses',
                        help='Directory to save extracted poses')
    parser.add_argument('--class_list', type=str, default='dataset/wlasl_class_list.txt',
                        help='Path to class list file')
    parser.add_argument('--num_frames', type=int, default=16,
                        help='Number of frames to extract per video')
    parser.add_argument('--frame_size', type=int, nargs=2, default=[224, 224],
                        help='Frame size (height width)')
    parser.add_argument('--max_videos', type=int, default=None,
                        help='Maximum number of videos to process (for testing)')
    parser.add_argument('--split', type=str, default=None,
                        help='Filter by split (train, test, or None)')
    
    args = parser.parse_args()
    
    extract_poses_for_dataset(
        json_path=args.json_path,
        video_dir=args.video_dir,
        output_dir=args.output_dir,
        class_list_path=args.class_list,
        num_frames=args.num_frames,
        frame_size=tuple(args.frame_size),
        max_videos=args.max_videos,
        split=args.split
    )


if __name__ == '__main__':
    main()

