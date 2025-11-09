#!/usr/bin/env python
"""
Standalone script to extract pose keypoints from WLASL videos.
Run with: python extract_poses.py [options]
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Check if MediaPipe is available before importing
try:
    import mediapipe
except ImportError:
    print("=" * 60)
    print("ERROR: MediaPipe is not installed!")
    print("=" * 60)
    print("\nMediaPipe is required for pose extraction.")
    print("\nTo install MediaPipe, run:")
    print("  pip install mediapipe")
    print("\nOr install all requirements:")
    print("  pip install -r requirements.txt")
    print("\n" + "=" * 60)
    sys.exit(1)

from src.pose_extraction import main

if __name__ == '__main__':
    main()

