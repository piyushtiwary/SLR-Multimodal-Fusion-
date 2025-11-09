#!/usr/bin/env python
"""
Standalone training script.
Run with: python train.py [options]
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.train_main import main

if __name__ == '__main__':
    main()

