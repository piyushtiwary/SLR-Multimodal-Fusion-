"""
Lightweight dummy test to verify the pipeline works correctly.
This test uses a very small model and dataset subset.
"""

import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import ModelCheckpoint

from .utils import set_seeds, parse_wlasl_data, get_num_classes
from .data_pipeline import create_dataset
from .model_builder import build_small_model


def dummy_test(
    json_path: str = 'dataset/WLASL_v0.3.json',
    video_dir: str = 'dataset/videos',
    class_list_path: str = 'dataset/wlasl_class_list.txt',
    num_videos: int = 20,
    num_classes: int = 10,
    epochs: int = 2,
    batch_size: int = 2
):
    """
    Run a lightweight test of the training pipeline.
    
    Args:
        json_path: Path to WLASL JSON file
        video_dir: Directory containing videos
        class_list_path: Path to class list file
        num_videos: Number of videos to use for testing
        num_classes: Number of classes to use (subset)
        epochs: Number of training epochs
        batch_size: Batch size
    """
    print("=" * 60)
    print("DUMMY TEST - Lightweight Pipeline Verification")
    print("=" * 60)
    
    # Set seeds
    set_seeds(42)
    
    # Parse data - use a very small subset
    print(f"\n1. Loading dataset (subset: {num_videos} videos, {num_classes} classes)...")
    video_label_pairs, gloss_to_idx = parse_wlasl_data(
        json_path=json_path,
        video_dir=video_dir,
        class_list_path=class_list_path,
        split=None,  # Use all splits for small test
        subset_size=num_videos,
        max_videos_per_class=2  # Max 2 videos per class
    )
    
    if len(video_label_pairs) == 0:
        print("Error: No videos found. Please check paths.")
        return False
    
    print(f"   Found {len(video_label_pairs)} videos")
    
    # Filter to use only first N classes and remap labels to [0, num_classes-1]
    unique_labels = sorted(set(label for _, label in video_label_pairs))
    
    # Limit to num_classes if we have more
    if len(unique_labels) > num_classes:
        # Filter to first num_classes
        valid_labels = set(unique_labels[:num_classes])
        video_label_pairs = [(v, l) for v, l in video_label_pairs if l in valid_labels]
        unique_labels = unique_labels[:num_classes]
        print(f"   Filtered to {num_classes} classes")
    
    # Get actual number of classes (may be less than num_classes if not enough data)
    actual_num_classes = len(unique_labels)
    print(f"   Actual number of classes: {actual_num_classes}")
    
    # Create mapping from original labels to remapped labels [0, actual_num_classes-1]
    # This ensures labels are always in range [0, actual_num_classes-1]
    label_remap = {old_label: new_label for new_label, old_label in enumerate(unique_labels)}
    
    # Remap labels to consecutive indices starting from 0
    video_label_pairs = [(video_path, label_remap[label]) for video_path, label in video_label_pairs]
    
    # Verify remapping
    remapped_labels = sorted(set(label for _, label in video_label_pairs))
    if remapped_labels != list(range(actual_num_classes)):
        print(f"   Warning: Label remapping issue. Expected [0, {actual_num_classes-1}], got {remapped_labels}")
    else:
        print(f"   ✓ Labels remapped to [0, {actual_num_classes-1}]")
    
    # Split into train/val (80/20)
    split_idx = int(len(video_label_pairs) * 0.8)
    train_pairs = video_label_pairs[:split_idx]
    val_pairs = video_label_pairs[split_idx:]
    
    print(f"   Train: {len(train_pairs)} videos, Val: {len(val_pairs)} videos")
    
    # Create datasets
    print("\n2. Creating datasets...")
    train_dataset = create_dataset(
        video_label_pairs=train_pairs,
        batch_size=batch_size,
        num_frames=8,  # Fewer frames for speed
        frame_size=(112, 112),  # Smaller frames for memory
        augment=True,
        shuffle=True,
        cache=False,
        prefetch=1
    )
    
    val_dataset = create_dataset(
        video_label_pairs=val_pairs,
        batch_size=batch_size,
        num_frames=8,
        frame_size=(112, 112),
        augment=False,
        shuffle=False,
        cache=False,
        prefetch=1
    )
    
    # Build small model
    print("\n3. Building small model...")
    model = build_small_model(num_classes=actual_num_classes)
    
    # Compile model
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss=keras.losses.SparseCategoricalCrossentropy(),
        metrics=['accuracy']
    )
    
    # Print model summary
    try:
        model.build(input_shape=(None, 8, 112, 112, 3))
        print("\nModel Summary:")
        model.summary()
    except Exception as e:
        print(f"Warning: Could not print model summary: {e}")
    
    # Create checkpoint directory
    os.makedirs('checkpoints', exist_ok=True)
    
    # Create callbacks
    # Use save_weights_only=True because subclassed models can't be saved in HDF5 format
    callbacks = [
        ModelCheckpoint(
            filepath='checkpoints/dummy_test_ckpt.weights.h5',
            save_best_only=True,
            save_weights_only=True,  # Save only weights for subclassed models
            monitor='val_accuracy',
            verbose=1
        )
    ]
    
    # Train model
    print(f"\n4. Training model ({epochs} epochs)...")
    print("   This should complete in < 5 minutes on CPU/low-end GPU...")
    
    try:
        history = model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=epochs,
            callbacks=callbacks,
            verbose=1
        )
        
        print("\n5. Training completed successfully!")
        print(f"   Final train accuracy: {history.history['accuracy'][-1]:.4f}")
        print(f"   Final val accuracy: {history.history['val_accuracy'][-1]:.4f}")
        
        # Test prediction
        print("\n6. Testing predictions...")
        for batch_x, batch_y in val_dataset.take(1):
            predictions = model.predict(batch_x, verbose=0)
            pred_classes = np.argmax(predictions, axis=1)
            
            print(f"   Sample batch predictions:")
            for i in range(min(3, len(batch_y))):
                true_label = batch_y[i].numpy()
                pred_label = pred_classes[i]
                confidence = predictions[i][pred_label]
                print(f"     True: {true_label}, Predicted: {pred_label}, Confidence: {confidence:.4f}")
            break
        
        print("\n" + "=" * 60)
        print("DUMMY TEST PASSED - Pipeline is working correctly!")
        print("=" * 60)
        return True
        
    except Exception as e:
        print(f"\nError during training: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Run dummy test to verify pipeline')
    parser.add_argument('--json_path', type=str, default='dataset/WLASL_v0.3.json',
                        help='Path to WLASL JSON file')
    parser.add_argument('--video_dir', type=str, default='dataset/videos',
                        help='Directory containing videos')
    parser.add_argument('--class_list', type=str, default='dataset/wlasl_class_list.txt',
                        help='Path to class list file')
    parser.add_argument('--num_videos', type=int, default=20,
                        help='Number of videos to use')
    parser.add_argument('--num_classes', type=int, default=10,
                        help='Number of classes to use')
    parser.add_argument('--epochs', type=int, default=2,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=2,
                        help='Batch size')
    
    args = parser.parse_args()
    
    success = dummy_test(
        json_path=args.json_path,
        video_dir=args.video_dir,
        class_list_path=args.class_list,
        num_videos=args.num_videos,
        num_classes=args.num_classes,
        epochs=args.epochs,
        batch_size=args.batch_size
    )
    
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()

