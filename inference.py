#!/usr/bin/env python
"""
Standalone inference script.
Run with: python inference.py --video_path path/to/video.mp4
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.run_inference import main

if __name__ == '__main__':
    main()

