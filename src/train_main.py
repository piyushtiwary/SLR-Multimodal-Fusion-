"""
Main training script for Sign Language Recognition.
"""

import os
import argparse
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    ReduceLROnPlateau,
    CSVLogger,
    TensorBoard
)
from tensorflow.keras.optimizers import Adam, AdamW
import json

from .utils import (
    set_seeds,
    load_config,
    save_config,
    log_memory_usage,
    get_num_classes
)
from .data_pipeline import create_train_val_datasets
from .model_builder import build_model


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description='Train Transformer-based SLR model')
    parser.add_argument('--config', type=str, default='configs/small_config.yaml',
                        help='Path to config file')
    parser.add_argument('--json_path', type=str, default='dataset/WLASL_v0.3.json',
                        help='Path to WLASL JSON file')
    parser.add_argument('--video_dir', type=str, default='dataset/videos',
                        help='Directory containing video files')
    parser.add_argument('--class_list', type=str, default='dataset/wlasl_class_list.txt',
                        help='Path to class list file')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints',
                        help='Directory to save checkpoints')
    parser.add_argument('--model_dir', type=str, default='models',
                        help='Directory to save final model')
    parser.add_argument('--logs_dir', type=str, default='logs',
                        help='Directory to save logs')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                        help='Initial learning rate')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--subset_size', type=int, default=None,
                        help='Limit total videos for debugging')
    parser.add_argument('--max_videos_per_class', type=int, default=None,
                        help='Limit videos per class for debugging')
    parser.add_argument('--small_model', action='store_true',
                        help='Use smaller model architecture')
    parser.add_argument('--mixed_precision', action='store_true',
                        help='Use mixed precision training')
    
    return parser.parse_args()


def setup_mixed_precision():
    """Enable mixed precision training."""
    policy = tf.keras.mixed_precision.Policy('mixed_float16')
    tf.keras.mixed_precision.set_global_policy(policy)
    print("Mixed precision training enabled")


def create_callbacks(
    checkpoint_dir: str,
    logs_dir: str,
    monitor: str = 'val_accuracy',
    patience: int = 10
):
    """
    Create training callbacks.
    
    Args:
        checkpoint_dir: Directory to save checkpoints
        logs_dir: Directory to save logs
        monitor: Metric to monitor
        patience: Early stopping patience
        
    Returns:
        List of callbacks
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)
    
    callbacks = [
        # Model checkpoint
        # Use save_weights_only=True because subclassed models can't be saved in HDF5 format
        ModelCheckpoint(
            filepath=os.path.join(checkpoint_dir, 'ckpt_{epoch:02d}.weights.h5'),
            monitor=monitor,
            save_best_only=True,
            save_weights_only=True,  # Save only weights for subclassed models
            mode='max',
            verbose=1
        ),
        
        # Early stopping
        EarlyStopping(
            monitor=monitor,
            patience=patience,
            restore_best_weights=True,
            verbose=1
        ),
        
        # Learning rate reduction
        ReduceLROnPlateau(
            monitor=monitor,
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        ),
        
        # CSV logger
        CSVLogger(
            filename=os.path.join(logs_dir, 'training_log.csv'),
            append=False
        ),
        
        # TensorBoard
        TensorBoard(
            log_dir=logs_dir,
            histogram_freq=1,
            write_graph=True,
            update_freq='epoch'
        )
    ]
    
    return callbacks


def train(
    config_path: str = None,
    json_path: str = 'dataset/WLASL_v0.3.json',
    video_dir: str = 'dataset/videos',
    class_list_path: str = 'dataset/wlasl_class_list.txt',
    checkpoint_dir: str = 'checkpoints',
    model_dir: str = 'models',
    logs_dir: str = 'logs',
    epochs: int = 50,
    batch_size: int = 4,
    learning_rate: float = 1e-4,
    seed: int = 42,
    subset_size: int = None,
    max_videos_per_class: int = None,
    small_model: bool = False,
    mixed_precision: bool = False
):
    """
    Main training function.
    
    Args:
        config_path: Path to config file
        json_path: Path to WLASL JSON file
        video_dir: Directory containing videos
        class_list_path: Path to class list file
        checkpoint_dir: Directory to save checkpoints
        model_dir: Directory to save final model
        logs_dir: Directory to save logs
        epochs: Number of training epochs
        batch_size: Batch size
        learning_rate: Initial learning rate
        seed: Random seed
        subset_size: Limit total videos (for debugging)
        max_videos_per_class: Limit videos per class (for debugging)
        small_model: Use smaller model architecture
        mixed_precision: Use mixed precision training
    """
    # Set seeds for reproducibility
    set_seeds(seed)
    
    # Enable mixed precision if requested
    if mixed_precision:
        setup_mixed_precision()
    
    # Load config if provided
    config = {}
    if config_path and os.path.exists(config_path):
        config = load_config(config_path)
        print(f"Loaded config from {config_path}")
        
        # Override with config values
        json_path = config.get('data', {}).get('json_path', json_path)
        video_dir = config.get('data', {}).get('video_dir', video_dir)
        class_list_path = config.get('data', {}).get('class_list', class_list_path)
        batch_size = config.get('training', {}).get('batch_size', batch_size)
        learning_rate = config.get('training', {}).get('learning_rate', learning_rate)
        epochs = config.get('training', {}).get('epochs', epochs)
        small_model = config.get('model', {}).get('small_model', small_model)
    
    # Get number of classes
    num_classes = get_num_classes(json_path, class_list_path)
    print(f"Number of classes: {num_classes}")
    
    # Create datasets
    print("Creating datasets...")
    
    # Check if using pre-extracted features
    use_pre_extracted = config.get('efficiency', {}).get('use_pre_extracted', False)
    use_pose = config.get('efficiency', {}).get('use_pose', False)
    input_type = config.get('model', {}).get('input_type', 'video')
    
    if use_pre_extracted:
        # Use lightweight pipeline with pre-extracted features
        from .lightweight_pipeline import create_lightweight_train_val_datasets
        
        feature_dir = config.get('efficiency', {}).get('pose_dir' if use_pose else 'feature_dir', 
                                                       'dataset/poses' if use_pose else 'dataset/features')
        
        train_dataset, val_dataset, gloss_to_idx = create_lightweight_train_val_datasets(
            json_path=json_path,
            feature_dir=feature_dir,
            video_dir=video_dir,  # Need video_dir to get video IDs from JSON
            class_list_path=class_list_path,
            batch_size=batch_size,
            use_pose=use_pose,
            subset_size=subset_size,
            max_videos_per_class=max_videos_per_class
        )
        print(f"Using pre-extracted {'pose keypoints' if use_pose else 'features'} from {feature_dir}")
    else:
        # Use standard pipeline with video loading
        from .data_pipeline import create_train_val_datasets
        
        train_dataset, val_dataset, gloss_to_idx = create_train_val_datasets(
            json_path=json_path,
            video_dir=video_dir,
            class_list_path=class_list_path,
            batch_size=batch_size,
            num_frames=config.get('model', {}).get('num_frames', 16),
            frame_size=tuple(config.get('model', {}).get('frame_size', [224, 224])),
            subset_size=subset_size,
            max_videos_per_class=max_videos_per_class
        )
        print("Using video loading pipeline")
    
    # Build model
    print("Building model...")
    if small_model:
        model_config = {
            'num_frames': 8,
            'frame_size': (112, 112),
            'embed_dim': 64,
            'num_heads': 2,
            'num_layers': 1,
            'ff_dim': 128,
            'dropout': 0.1,
            'use_positional_encoding': True
        }
    else:
        # Get model config and filter out 'small_model' key if present
        model_config = config.get('model', {}).copy()
        model_config.pop('small_model', None)  # Remove if present
        
        # Set input_type based on config
        if use_pre_extracted:
            model_config['input_type'] = 'pose' if use_pose else 'features'
        else:
            model_config['input_type'] = model_config.get('input_type', 'video')
    
    model = build_model(
        num_classes=num_classes,
        config={'model': model_config}
    )
    
    # Print model summary
    input_type = model_config.get('input_type', 'video')
    try:
        if input_type == 'video':
            frame_size = tuple(model_config.get('frame_size', [224, 224]))
            num_frames = model_config.get('num_frames', 16)
            model.build(input_shape=(None, num_frames, frame_size[0], frame_size[1], 3))
        elif input_type == 'pose':
            num_frames = model_config.get('num_frames', 16)
            model.build(input_shape=(None, num_frames, 132))  # 33 keypoints * 4
        elif input_type == 'features':
            num_frames = model_config.get('num_frames', 16)
            embed_dim = model_config.get('embed_dim', 128)
            model.build(input_shape=(None, num_frames, embed_dim))
        model.summary()
    except Exception as e:
        print(f"Warning: Could not build model for summary: {e}")
        print("Model will be built during first forward pass")
    
    # Compile model
    optimizer = AdamW(learning_rate=learning_rate)
    
    # For mixed precision, use loss scaling
    if mixed_precision:
        loss = keras.losses.SparseCategoricalCrossentropy()
        optimizer = tf.keras.mixed_precision.LossScaleOptimizer(optimizer)
    else:
        loss = keras.losses.SparseCategoricalCrossentropy()
    
    # Custom top-3 accuracy metric
    def top_3_accuracy(y_true, y_pred):
        return tf.keras.metrics.sparse_top_k_categorical_accuracy(y_true, y_pred, k=3)
    
    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=['accuracy', top_3_accuracy]
    )
    
    # Log memory usage
    log_memory_usage()
    
    # Create callbacks
    callbacks = create_callbacks(
        checkpoint_dir=checkpoint_dir,
        logs_dir=logs_dir,
        monitor='val_accuracy',
        patience=10
    )
    
    # Save config
    os.makedirs(model_dir, exist_ok=True)
    save_config(config, os.path.join(model_dir, 'training_config.yaml'))
    
    # Save gloss_to_idx mapping
    with open(os.path.join(model_dir, 'gloss_to_idx.json'), 'w') as f:
        json.dump(gloss_to_idx, f, indent=2)
    
    # Train model
    print("Starting training...")
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=epochs,
        callbacks=callbacks,
        verbose=1
    )
    
    # Save final model
    # Use SavedModel format (tf) instead of HDF5 for subclassed models
    final_model_path = os.path.join(model_dir, 'final_slr_model')
    print(f"Saving final model to {final_model_path}")
    model.save(final_model_path, save_format='tf')  # Use SavedModel format for subclassed models
    
    # Also save weights only as backup
    weights_path = os.path.join(model_dir, 'final_slr_model.weights.h5')
    model.save_weights(weights_path)
    print(f"Also saved weights to {weights_path}")
    
    print("Training completed!")
    print(f"Best validation accuracy: {max(history.history['val_accuracy']):.4f}")
    
    return model, history


def main():
    """Main entry point."""
    args = parse_args()
    
    train(
        config_path=args.config,
        json_path=args.json_path,
        video_dir=args.video_dir,
        class_list_path=args.class_list,
        checkpoint_dir=args.checkpoint_dir,
        model_dir=args.model_dir,
        logs_dir=args.logs_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        seed=args.seed,
        subset_size=args.subset_size,
        max_videos_per_class=args.max_videos_per_class,
        small_model=args.small_model,
        mixed_precision=args.mixed_precision
    )


if __name__ == '__main__':
    main()

