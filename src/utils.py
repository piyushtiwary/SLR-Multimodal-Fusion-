"""
Utility functions for Sign Language Recognition project.
"""

import os
import json
import random
import numpy as np
import tensorflow as tf
from typing import Dict, List, Tuple, Optional
import yaml


def set_seeds(seed: int = 42) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


def load_wlasl_annotations(json_path: str) -> Dict:
    """
    Load WLASL annotation JSON file.
    
    Args:
        json_path: Path to WLASL JSON annotation file
        
    Returns:
        Dictionary with gloss labels and video instances
    """
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data


def load_class_list(class_list_path: str) -> Dict[str, int]:
    """
    Load class list mapping gloss names to class indices.
    
    Args:
        class_list_path: Path to wlasl_class_list.txt
        
    Returns:
        Dictionary mapping gloss name to class index
    """
    gloss_to_idx = {}
    with open(class_list_path, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) == 2:
                idx, gloss = parts
                gloss_to_idx[gloss] = int(idx)
    return gloss_to_idx


def parse_wlasl_data(
    json_path: str,
    video_dir: str,
    class_list_path: Optional[str] = None,
    split: Optional[str] = None,
    max_videos_per_class: Optional[int] = None,
    subset_size: Optional[int] = None
) -> List[Tuple[str, int]]:
    """
    Parse WLASL dataset and return list of (video_path, label) tuples.
    
    Args:
        json_path: Path to WLASL JSON file
        video_dir: Directory containing video files
        class_list_path: Path to class list file (optional, for filtering)
        split: 'train', 'test', or None (for all splits)
        max_videos_per_class: Maximum videos per class (for debugging)
        subset_size: Total number of videos to use (for debugging)
        
    Returns:
        List of (video_path, label_index) tuples
    """
    data = load_wlasl_annotations(json_path)
    
    # Load class mapping if provided
    if class_list_path:
        gloss_to_idx = load_class_list(class_list_path)
    else:
        # Create gloss_to_idx from data
        unique_glosses = sorted(set(item['gloss'] for item in data))
        gloss_to_idx = {gloss: idx for idx, gloss in enumerate(unique_glosses)}
    
    video_label_pairs = []
    class_counts = {}
    
    for item in data:
        gloss = item['gloss']
        if gloss not in gloss_to_idx:
            continue
            
        label_idx = gloss_to_idx[gloss]
        
        for instance in item['instances']:
            # Filter by split if specified
            if split and instance.get('split') != split:
                continue
            
            video_id = instance['video_id']
            video_path = os.path.join(video_dir, f"{video_id}.mp4")
            
            # Check if video exists
            if not os.path.exists(video_path):
                continue
            
            # Limit videos per class if specified
            if max_videos_per_class:
                if gloss not in class_counts:
                    class_counts[gloss] = 0
                if class_counts[gloss] >= max_videos_per_class:
                    continue
                class_counts[gloss] += 1
            
            video_label_pairs.append((video_path, label_idx))
    
    # Shuffle for randomness
    random.shuffle(video_label_pairs)
    
    # Limit total videos if specified (for debugging)
    if subset_size:
        video_label_pairs = video_label_pairs[:subset_size]
    
    return video_label_pairs, gloss_to_idx


def get_num_classes(json_path: str, class_list_path: Optional[str] = None) -> int:
    """
    Get number of classes in the dataset.
    
    Args:
        json_path: Path to WLASL JSON file
        class_list_path: Path to class list file (optional)
        
    Returns:
        Number of classes
    """
    if class_list_path:
        gloss_to_idx = load_class_list(class_list_path)
        return len(gloss_to_idx)
    else:
        data = load_wlasl_annotations(json_path)
        unique_glosses = set(item['gloss'] for item in data)
        return len(unique_glosses)


def load_config(config_path: str) -> Dict:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to YAML config file
        
    Returns:
        Configuration dictionary
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def save_config(config: Dict, save_path: str) -> None:
    """
    Save configuration to YAML file.
    
    Args:
        config: Configuration dictionary
        save_path: Path to save config file
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)


def log_memory_usage():
    """Log current memory usage (GPU and CPU)."""
    try:
        # GPU memory
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            for gpu in gpus:
                mem_info = tf.config.experimental.get_memory_info(gpu)
                print(f"GPU {gpu.name}: {mem_info['current'] / 1024**3:.2f} GB / {mem_info['peak'] / 1024**3:.2f} GB")
    except Exception as e:
        print(f"Could not get GPU memory info: {e}")


def idx_to_gloss(idx: int, gloss_to_idx: Dict[str, int]) -> str:
    """
    Convert class index to gloss name.
    
    Args:
        idx: Class index
        gloss_to_idx: Dictionary mapping gloss to index
        
    Returns:
        Gloss name
    """
    idx_to_gloss_dict = {v: k for k, v in gloss_to_idx.items()}
    return idx_to_gloss_dict.get(idx, "UNKNOWN")

