"""
Inference script for Sign Language Recognition.
"""

import os
import argparse
import numpy as np
import tensorflow as tf
from tensorflow import keras
import json
import cv2

from .utils import load_config, idx_to_gloss
from .data_pipeline import load_video_frames, preprocess_video


def load_model_and_mappings(
    model_path: str,
    gloss_mapping_path: str = None,
    model_config: dict = None,
    num_classes: int = None
):
    """
    Load trained model and gloss-to-index mapping.
    Supports both SavedModel format and weights-only files.
    
    Args:
        model_path: Path to saved model (SavedModel directory or weights file)
        gloss_mapping_path: Path to gloss_to_idx.json
        model_config: Model configuration dict (required if loading weights only)
        num_classes: Number of classes (required if loading weights only)
        
    Returns:
        Tuple of (model, idx_to_gloss_dict)
    """
    print(f"Loading model from {model_path}...")
    
    # Try to load as SavedModel first
    if os.path.isdir(model_path) or (os.path.exists(model_path) and not model_path.endswith('.h5')):
        try:
            model = keras.models.load_model(model_path, compile=False)
            print("Loaded model from SavedModel format")
        except Exception as e:
            print(f"Failed to load as SavedModel: {e}")
            raise
    # Try to load as weights file
    elif model_path.endswith('.weights.h5') or model_path.endswith('_weights.h5'):
        if model_config is None or num_classes is None:
            raise ValueError("model_config and num_classes are required when loading weights-only file")
        
        # Rebuild model from config
        from .model_builder import build_model
        model = build_model(num_classes=num_classes, config={'model': model_config})
        
        # Load weights
        model.load_weights(model_path)
        print("Loaded model weights and rebuilt model architecture")
    else:
        # Try as regular model file
        try:
            model = keras.models.load_model(model_path, compile=False)
            print("Loaded model from file")
        except Exception as e:
            print(f"Failed to load model: {e}")
            raise
    
    # Load gloss mapping if provided
    idx_to_gloss_dict = {}
    if gloss_mapping_path and os.path.exists(gloss_mapping_path):
        with open(gloss_mapping_path, 'r') as f:
            gloss_to_idx = json.load(f)
        idx_to_gloss_dict = {v: k for k, v in gloss_to_idx.items()}
    else:
        print("Warning: No gloss mapping found. Using indices as labels.")
    
    return model, idx_to_gloss_dict


def predict_video(
    model: keras.Model,
    video_path: str,
    num_frames: int = 16,
    frame_size: tuple = (224, 224),
    idx_to_gloss: dict = None,
    top_k: int = 3
):
    """
    Predict sign language class from video.
    
    Args:
        model: Trained Keras model
        video_path: Path to input video
        num_frames: Number of frames to extract
        frame_size: Target frame size
        idx_to_gloss: Mapping from class index to gloss name
        top_k: Number of top predictions to return
        
    Returns:
        Dictionary with predictions
    """
    # Load video frames
    frames = load_video_frames(
        video_path=video_path,
        num_frames=num_frames,
        frame_size=frame_size
    )
    
    # Preprocess
    frames_tensor = tf.convert_to_tensor(frames, dtype=tf.uint8)
    frames_preprocessed = preprocess_video(frames_tensor, augment=False)
    frames_batch = tf.expand_dims(frames_preprocessed, 0)  # Add batch dimension
    
    # Predict
    predictions = model.predict(frames_batch, verbose=0)
    probs = predictions[0]
    
    # Get top-k predictions
    top_k_indices = np.argsort(probs)[-top_k:][::-1]
    top_k_probs = probs[top_k_indices]
    
    # Format results
    results = {
        'predicted_class': int(top_k_indices[0]),
        'confidence': float(top_k_probs[0]),
        'top_k': []
    }
    
    for idx, prob in zip(top_k_indices, top_k_probs):
        gloss_name = idx_to_gloss.get(int(idx), f"CLASS_{idx}") if idx_to_gloss else f"CLASS_{idx}"
        results['top_k'].append({
            'gloss': gloss_name,
            'class_idx': int(idx),
            'confidence': float(prob)
        })
    
    return results


def print_predictions(results: dict):
    """
    Print prediction results in a formatted way.
    
    Args:
        results: Prediction results dictionary
    """
    predicted_gloss = results['top_k'][0]['gloss']
    confidence = results['top_k'][0]['confidence']
    
    print(f"\nPredicted sign: {predicted_gloss.upper()}")
    print(f"Confidence: {confidence:.4f}")
    print(f"\nTop-{len(results['top_k'])} predictions:")
    for i, pred in enumerate(results['top_k'], 1):
        print(f"  {i}. {pred['gloss']}: {pred['confidence']:.4f}")


def main():
    """Main inference function."""
    parser = argparse.ArgumentParser(description='Run inference on a video')
    parser.add_argument('--video_path', type=str, required=True,
                        help='Path to input video file')
    parser.add_argument('--model_path', type=str, default='models/final_slr_model',
                        help='Path to trained model (SavedModel directory or weights file)')
    parser.add_argument('--gloss_mapping', type=str, default='models/gloss_to_idx.json',
                        help='Path to gloss_to_idx.json')
    parser.add_argument('--weights_only', action='store_true',
                        help='Load weights only (requires model config)')
    parser.add_argument('--num_frames', type=int, default=16,
                        help='Number of frames to extract')
    parser.add_argument('--frame_size', type=int, nargs=2, default=[224, 224],
                        help='Frame size (height width)')
    parser.add_argument('--top_k', type=int, default=3,
                        help='Number of top predictions to show')
    
    args = parser.parse_args()
    
    # Check if video exists
    if not os.path.exists(args.video_path):
        print(f"Error: Video file not found: {args.video_path}")
        return
    
    # Check if model exists
    if not os.path.exists(args.model_path):
        # Try alternative paths
        alternatives = [
            args.model_path + '.h5',
            args.model_path.replace('.weights.h5', ''),
            'models/final_slr_model',
            'models/final_slr_model.weights.h5'
        ]
        found = False
        for alt_path in alternatives:
            if os.path.exists(alt_path):
                print(f"Model not found at {args.model_path}, trying {alt_path}...")
                args.model_path = alt_path
                found = True
                break
        
        if not found:
            print(f"Error: Model file not found: {args.model_path}")
            print("Please train a model first using train_main.py")
            return
    
    # Load model and mappings
    # If loading weights only, we'd need model config - for now, try SavedModel format first
    model, idx_to_gloss = load_model_and_mappings(
        model_path=args.model_path,
        gloss_mapping_path=args.gloss_mapping
    )
    
    # Run prediction
    print(f"Processing video: {args.video_path}")
    results = predict_video(
        model=model,
        video_path=args.video_path,
        num_frames=args.num_frames,
        frame_size=tuple(args.frame_size),
        idx_to_gloss=idx_to_gloss,
        top_k=args.top_k
    )
    
    # Print results
    print_predictions(results)


if __name__ == '__main__':
    main()

