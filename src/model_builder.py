"""
Lightweight Transformer-based model for Sign Language Recognition.
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from typing import Optional, Dict, Tuple
import numpy as np


class PositionalEncoding(layers.Layer):
    """Positional encoding for temporal sequences."""
    
    def __init__(self, max_len: int = 100, embed_dim: int = 128, **kwargs):
        super().__init__(**kwargs)
        self.max_len = max_len
        self.embed_dim = embed_dim
        
        # Create positional encoding matrix
        position = np.arange(max_len)[:, np.newaxis]
        div_term = np.exp(np.arange(0, embed_dim, 2) * -(np.log(10000.0) / embed_dim))
        pe = np.zeros((max_len, embed_dim))
        pe[:, 0::2] = np.sin(position * div_term)
        pe[:, 1::2] = np.cos(position * div_term)
        pe = pe[np.newaxis, ...]
        self.pe = tf.constant(pe, dtype=tf.float32)
    
    def call(self, x):
        seq_len = tf.shape(x)[1]
        return x + self.pe[:, :seq_len, :]
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'max_len': self.max_len,
            'embed_dim': self.embed_dim
        })
        return config


class TransformerBlock(layers.Layer):
    """Lightweight Transformer encoder block."""
    
    def __init__(
        self,
        embed_dim: int = 128,
        num_heads: int = 4,
        ff_dim: int = 256,
        dropout: float = 0.1,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.dropout = dropout
        
        self.attn = layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=embed_dim // num_heads,
            dropout=dropout
        )
        self.ffn = keras.Sequential([
            layers.Dense(ff_dim, activation='relu'),
            layers.Dense(embed_dim),
            layers.Dropout(dropout)
        ])
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(dropout)
        self.dropout2 = layers.Dropout(dropout)
    
    def call(self, inputs, training=False):
        # Self-attention
        attn_output = self.attn(inputs, inputs, training=training)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output, training=training)
        
        # Feed-forward
        ffn_output = self.ffn(out1, training=training)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output, training=training)
        
        return out2
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'embed_dim': self.embed_dim,
            'num_heads': self.num_heads,
            'ff_dim': self.ff_dim,
            'dropout': self.dropout
        })
        return config


class VideoTransformer(keras.Model):
    """Lightweight Transformer-based model for video classification."""
    
    def __init__(
        self,
        num_classes: int,
        num_frames: int = 16,
        frame_size: Tuple[int, int] = (224, 224),
        embed_dim: int = 128,
        num_heads: int = 4,
        num_layers: int = 2,
        ff_dim: int = 256,
        dropout: float = 0.1,
        use_positional_encoding: bool = True,
        input_type: str = 'video',  # 'video', 'pose', or 'features'
        **kwargs
    ):
        super().__init__(**kwargs)
        self.num_classes = num_classes
        self.num_frames = num_frames
        self.frame_size = frame_size
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.ff_dim = ff_dim
        self.dropout = dropout
        self.input_type = input_type
        
        # Input projection based on input type
        if input_type == 'video':
            # CNN backbone for spatial feature extraction from video frames
            cnn_channels = max(embed_dim, 64)
            self.cnn_backbone = keras.Sequential([
                layers.Conv2D(32, 3, strides=2, padding='same', activation='relu'),
                layers.BatchNormalization(),
                layers.Conv2D(64, 3, strides=2, padding='same', activation='relu'),
                layers.BatchNormalization(),
                layers.Conv2D(cnn_channels, 3, strides=2, padding='same', activation='relu'),
                layers.BatchNormalization(),
                layers.GlobalAveragePooling2D(),
                layers.Dense(embed_dim, activation='relu'),
                layers.Dropout(dropout)
            ], name='cnn_backbone')
            self.input_proj = None
        elif input_type == 'pose':
            # Project pose keypoints (132 dims: 33 keypoints * 4) to embed_dim
            self.input_proj = layers.Dense(embed_dim, activation='relu', name='pose_projection')
            self.cnn_backbone = None
        elif input_type == 'features':
            # Features are already extracted, just project to embed_dim if needed
            self.input_proj = layers.Dense(embed_dim, activation='relu', name='feature_projection')
            self.cnn_backbone = None
        else:
            raise ValueError(f"Unknown input_type: {input_type}")
        
        # Positional encoding
        if use_positional_encoding:
            self.pos_encoding = PositionalEncoding(
                max_len=num_frames,
                embed_dim=embed_dim
            )
        else:
            self.pos_encoding = None
        
        # Transformer blocks
        self.transformer_blocks = [
            TransformerBlock(
                embed_dim=embed_dim,
                num_heads=num_heads,
                ff_dim=ff_dim,
                dropout=dropout
            ) for _ in range(num_layers)
        ]
        
        # Global pooling and classification head
        self.global_pool = layers.GlobalAveragePooling1D()
        self.classifier = keras.Sequential([
            layers.Dense(ff_dim, activation='relu'),
            layers.Dropout(dropout),
            layers.Dense(num_classes, activation='softmax')
        ], name='classifier')
    
    def call(self, inputs, training=False):
        batch_size = tf.shape(inputs)[0]
        
        if self.input_type == 'video':
            # inputs shape: (batch, num_frames, height, width, channels)
            num_frames = tf.shape(inputs)[1]
            
            # Extract features from each frame using CNN
            frame_features = tf.reshape(
                inputs,
                (batch_size * num_frames, self.frame_size[0], self.frame_size[1], 3)
            )
            
            # Apply CNN backbone
            frame_embeddings = self.cnn_backbone(frame_features, training=training)
            
            # Reshape back to (batch, num_frames, embed_dim)
            frame_embeddings = tf.reshape(
                frame_embeddings,
                (batch_size, num_frames, self.embed_dim)
            )
        elif self.input_type == 'pose':
            # inputs shape: (batch, num_frames, 132) - flattened pose keypoints
            # Project to embed_dim
            frame_embeddings = self.input_proj(inputs, training=training)
        elif self.input_type == 'features':
            # inputs shape: (batch, num_frames, feature_dim)
            # Project to embed_dim if needed
            if inputs.shape[-1] != self.embed_dim:
                frame_embeddings = self.input_proj(inputs, training=training)
            else:
                frame_embeddings = inputs
        
        # Add positional encoding
        if self.pos_encoding:
            frame_embeddings = self.pos_encoding(frame_embeddings)
        
        # Apply Transformer blocks
        x = frame_embeddings
        for transformer_block in self.transformer_blocks:
            x = transformer_block(x, training=training)
        
        # Global pooling
        x = self.global_pool(x)
        
        # Classification
        outputs = self.classifier(x, training=training)
        
        return outputs
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'num_classes': self.num_classes,
            'num_frames': self.num_frames,
            'frame_size': self.frame_size,
            'embed_dim': self.embed_dim,
            'num_heads': self.num_heads,
            'num_layers': self.num_layers,
            'ff_dim': self.ff_dim,
            'dropout': self.dropout,
            'use_positional_encoding': self.pos_encoding is not None,
            'input_type': self.input_type
        })
        return config


def build_model(
    num_classes: int,
    config: Optional[Dict] = None,
    **kwargs
) -> keras.Model:
    """
    Build Transformer-based SLR model.
    
    Args:
        num_classes: Number of sign language classes
        config: Configuration dictionary
        **kwargs: Additional model parameters
        
    Returns:
        Compiled Keras model
    """
    # Default config - lightweight settings
    default_config = {
        'num_frames': 16,
        'frame_size': (224, 224),
        'embed_dim': 128,
        'num_heads': 4,
        'num_layers': 2,  # depth = 2
        'ff_dim': 256,  # mlp_dim = 256
        'dropout': 0.1,
        'use_positional_encoding': True,
        'input_type': 'video'  # 'video', 'pose', or 'features'
    }
    
    if config:
        default_config.update(config.get('model', {}))
    
    # Override with kwargs
    default_config.update(kwargs)
    
    # Filter out invalid parameters that VideoTransformer doesn't accept
    valid_params = {
        'num_frames', 'frame_size', 'embed_dim', 'num_heads', 'num_layers',
        'ff_dim', 'dropout', 'use_positional_encoding', 'input_type'
    }
    model_config = {k: v for k, v in default_config.items() if k in valid_params}
    
    model = VideoTransformer(
        num_classes=num_classes,
        **model_config
    )
    
    return model


def build_small_model(num_classes: int) -> keras.Model:
    """
    Build a very small model for testing (dummy_test.py).
    
    Args:
        num_classes: Number of classes
        
    Returns:
        Small Keras model
    """
    return VideoTransformer(
        num_classes=num_classes,
        num_frames=8,  # Fewer frames
        frame_size=(112, 112),  # Smaller frames
        embed_dim=64,  # Smaller embedding
        num_heads=2,  # Fewer heads
        num_layers=1,  # Fewer layers
        ff_dim=128,  # Smaller FF dimension
        dropout=0.1
    )

