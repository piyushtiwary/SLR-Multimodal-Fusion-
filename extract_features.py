#!/usr/bin/env python
"""
Standalone script to extract frame embeddings from WLASL videos.
Run with: python extract_features.py [options]
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.feature_extraction import main

if __name__ == '__main__':
    main()

