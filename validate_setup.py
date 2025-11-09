#!/usr/bin/env python
"""
Quick validation script to check if the project setup is correct.
"""

import sys
import os

def check_imports():
    """Check if all required modules can be imported."""
    print("Checking imports...")
    try:
        import tensorflow as tf
        print(f"  ✓ TensorFlow {tf.__version__}")
    except ImportError as e:
        print(f"  ✗ TensorFlow: {e}")
        return False
    
    try:
        import numpy as np
        print(f"  ✓ NumPy {np.__version__}")
    except ImportError as e:
        print(f"  ✗ NumPy: {e}")
        return False
    
    try:
        import cv2
        print(f"  ✓ OpenCV {cv2.__version__}")
    except ImportError as e:
        print(f"  ✗ OpenCV: {e}")
        return False
    
    try:
        import yaml
        print(f"  ✓ PyYAML")
    except ImportError as e:
        print(f"  ✗ PyYAML: {e}")
        return False
    
    return True

def check_project_structure():
    """Check if project structure is correct."""
    print("\nChecking project structure...")
    required_dirs = ['src', 'configs', 'dataset']
    required_files = [
        'src/utils.py',
        'src/data_pipeline.py',
        'src/model_builder.py',
        'src/train_main.py',
        'src/run_inference.py',
        'src/dummy_test.py',
        'configs/small_config.yaml',
        'requirements.txt',
        'README.md'
    ]
    
    all_good = True
    for dir_path in required_dirs:
        if os.path.exists(dir_path):
            print(f"  ✓ {dir_path}/")
        else:
            print(f"  ✗ {dir_path}/ (missing)")
            all_good = False
    
    for file_path in required_files:
        if os.path.exists(file_path):
            print(f"  ✓ {file_path}")
        else:
            print(f"  ✗ {file_path} (missing)")
            all_good = False
    
    return all_good

def check_dataset():
    """Check if dataset files exist."""
    print("\nChecking dataset...")
    dataset_files = [
        'dataset/WLASL_v0.3.json',
        'dataset/wlasl_class_list.txt',
        'dataset/videos'
    ]
    
    all_good = True
    for path in dataset_files:
        if os.path.exists(path):
            if os.path.isdir(path):
                # Count video files
                video_count = len([f for f in os.listdir(path) if f.endswith('.mp4')])
                print(f"  ✓ {path} ({video_count} videos)")
            else:
                print(f"  ✓ {path}")
        else:
            print(f"  ✗ {path} (missing)")
            all_good = False
    
    return all_good

def check_src_imports():
    """Check if src modules can be imported."""
    print("\nChecking src module imports...")
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    
    modules = [
        'src.utils',
        'src.data_pipeline',
        'src.model_builder',
        'src.train_main',
        'src.run_inference',
        'src.dummy_test'
    ]
    
    all_good = True
    for module_name in modules:
        try:
            __import__(module_name)
            print(f"  ✓ {module_name}")
        except Exception as e:
            print(f"  ✗ {module_name}: {e}")
            all_good = False
    
    return all_good

def main():
    """Main validation function."""
    print("=" * 60)
    print("Project Setup Validation")
    print("=" * 60)
    
    checks = [
        ("Imports", check_imports),
        ("Project Structure", check_project_structure),
        ("Dataset", check_dataset),
        ("Source Modules", check_src_imports)
    ]
    
    results = []
    for name, check_func in checks:
        try:
            result = check_func()
            results.append((name, result))
        except Exception as e:
            print(f"\nError during {name} check: {e}")
            results.append((name, False))
    
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    for name, result in results:
        status = "PASS" if result else "FAIL"
        print(f"  {name}: {status}")
    
    all_passed = all(result for _, result in results)
    if all_passed:
        print("\n✓ All checks passed! Project setup is correct.")
        return 0
    else:
        print("\n✗ Some checks failed. Please fix the issues above.")
        return 1

if __name__ == '__main__':
    sys.exit(main())

