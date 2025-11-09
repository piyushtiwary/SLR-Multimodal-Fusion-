#!/usr/bin/env python
"""
Standalone dummy test script.
Run with: python dummy_test.py [options]
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.dummy_test import main

if __name__ == '__main__':
    main()

