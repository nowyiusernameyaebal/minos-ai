"""
Ariadne Model - Neural network architecture for market prediction

This module defines the Ariadne model architecture, a deep learning model
designed for financial time series analysis and trading signal generation
on cryptocurrency markets. The model serves as the core prediction engine
for the Minos-AI DeFi platform.

The model combines LSTM/Transformer layers for temporal pattern recognition
with attention mechanisms to focus on relevant market signals, and incorporates
multi-head attention to capture inter-asset relationships.

Reference architecture papers:
- "Temporal Fusion Transformers for Interpretable Multi-horizon Time Series Forecasting"
- "Attention Is All You Need"
- "DeepTrade: A Deep Reinforcement Learning Approach for Financial Portfolio Management"

Author: Minos-AI Team
Date: December 15, 2024
License: Proprietary
"""

import os
import json
import logging
import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Union, Optional, Any, Callable

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model, load_model, clone_model
from tensorflow.keras.layers import (
    Input, Dense, LSTM, GRU, Bidirectional, Dropout, BatchNormalization,
    LayerNormalization, MultiHeadAttention, Concatenate, Add, GlobalAveragePooling1D,
    Conv1D, MaxPooling1D, Attention, TimeDistributed, Reshape, Permute, Lambda,
    Activation, LeakyReLU
)
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.callbacks import (
    EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, 
    TensorBoard, LearningRateScheduler, CSVLogger
)
from tensorflow.keras.regularizers import l1_l2
from tensorflow.keras.initializers import GlorotNormal, HeNormal
from tensorflow.keras.metrics import RootMeanSquaredError, MeanAbsoluteError, AUC
from tensorflow.keras.losses import MeanSquaredError, BinaryCrossentropy, CategoricalCrossentropy
import tensorflow_addons as tfa

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("ariadne_model.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Set environment variables
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow logging (1=INFO, 2=WARNING, 3=ERROR)

class AriadneModel:
    """
    Ariadne Neural Network for trading signal generation.
    
    The model predicts price movements and trading signals based on
    historical price patterns, technical indicators, sentiment analysis,
    and on-chain metrics. It employs a hybrid architecture combining
    CNNs for feature extraction, LSTMs for sequential learning, and
    attention mechanisms for focus on relevant signals.
    
    Features:
    - Multi-timeframe analysis (1m, 5m, 15m, 1h, 4h, 1d)
    - Cross-asset correlation modeling
    - Adaptive learning rate scheduling
    - Anomaly detection for market regime changes
    - Uncertainty quantification for risk assessment
    - Explainable predictions with attention visualization
    """
    
    # Class constants
    MODEL_VERSION = "1.0.0"
    SUPPORTED_MODES = ["classification", "regression", "multi_task"]
    SUPPORTED_ARCHITECTURES = ["cnn_lstm", "transformer", "hybrid", "wavenet", "tcn"]
    OPTIONAL_COMPONENTS = ["uncertainty", "attention", "residual", "ensemble"]
    
    def __init__(
        self,
        config_path: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        model_dir: Optional[str] = None,
        verbose: bool = True
    ):
        """
        Initialize the Ariadne model.
        
        Args:
            config_path: Path to the JSON configuration file
            config: Configuration dictionary (overrides config_path if provided)
            model_dir: Directory to save model checkpoints and logs
            verbose: Whether to print verbose output
        
        Raises:
            FileNotFoundError: If config_path is provided but file doesn't exist
            ValueError: If neither config_path nor config is provided or config is invalid
            RuntimeError: If TensorFlow initialization fails
        """
        self.verbose = verbose
        
        # Load configuration
        if config is not None:
            self.config = config
        elif config_path is not None:
            try:
                with open(config_path, 'r') as f:
                    self.config = json.load(f)
            except FileNotFoundError:
                logger.error(f"Configuration file not found: {config_path}")
                raise FileNotFoundError(f"Configuration file not found: {config_path}")
            except json.JSONDecodeError as e:
                logger.error(f"Invalid JSON in configuration file: {e}")
                raise ValueError(f"Invalid JSON in configuration file: {e}")
        else:
            logger.error("Either config_path or config must be provided")
            raise ValueError("Either config_path or config must be provided")
        
        # Set model directory
        if model_dir is None:
            self.model_dir = Path(self.config.get("model_dir", "./models"))
        else:
            self.model_dir = Path(model_dir)
        
        # Create model directory if it doesn't exist
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup TensorFlow
        try:
            self._setup_tensorflow()
        except Exception as e:
            logger.error(f"Failed to initialize TensorFlow: {e}")
            raise RuntimeError(f"Failed to initialize TensorFlow: {e}")
        
        # Validate configuration
        self._validate_config()
        
        # Initialize model attributes
        self.model = None
        self.ensemble_models = []
        self.feature_scaler = None
        self.label_scaler = None
        self.feature_names = self.config.get("feature_names", [])
        self.last_training_time = None
        self.training_history = None
        self.model_evaluation = {}
        self.uncertainty_model = None
        
        # Extract key configuration parameters
        self.mode = self.config.get("mode", "classification")
        self.architecture = self.config.get("architecture", "hybrid")
        self.sequence_length = self.config.get("sequence_length", 60)
        self.prediction_horizon = self.config.get("prediction_horizon", 5)
        self.n_features = self.config.get("n_features", 0)
        self.n_targets = self.config.get("n_targets", 1)
        
        # Setup additional components based on configuration
        self._setup_components()
        
        # Set random seeds for reproducibility
        self._set_random_seeds()
        
        if self.verbose:
            logger.info(f"Ariadne model initialized (Version: {self.MODEL_VERSION})")
            logger.info(f"Mode: {self.mode}, Architecture: {self.architecture}")
            logger.info(f"Model directory: {self.model_dir}")
    
    def _setup_tensorflow(self) -> None:
        """
        Configure TensorFlow environment.
        
        Sets up GPU memory growth, mixed precision, and other TensorFlow settings.
        """
        # Configure GPU memory growth to avoid allocating all memory at once
        physical_devices = tf.config.list_physical_devices('GPU')
        
        if physical_devices:
            for device in physical_devices:
                try:
                    tf.config.experimental.set_memory_growth(device, True)
                    logger.info(f"Memory growth enabled for GPU: {device}")
                except Exception as e:
                    logger.warning(f"Failed to set memory growth for GPU {device}: {e}")
            
            # Set visible devices based on configuration
            visible_devices = self.config.get("visible_gpus", None)
            if visible_devices is not None:
                try:
                    tf.config.set_visible_devices(
                        [physical_devices[i] for i in visible_devices], 'GPU'
                    )
                    logger.info(f"Visible GPUs set to: {visible_devices}")
                except Exception as e:
                    logger.warning(f"Failed to set visible GPUs: {e}")
        else:
            logger.info("No GPU detected. Running on CPU.")
        
        # Enable mixed precision for better performance on GPUs with tensor cores
        if self.config.get("mixed_precision", False):
            try:
                policy = tf.keras.mixed_precision.Policy('mixed_float16')
                tf.keras.mixed_precision.set_global_policy(policy)
                logger.info("Mixed precision policy set to mixed_float16")
            except Exception as e:
                logger.warning(f"Failed to set mixed precision policy: {e}")

        # Configure thread parallelism
        tf.config.threading.set_inter_op_parallelism_threads(
            self.config.get("inter_op_parallelism_threads", 0)
        )
        tf.config.threading.set_intra_op_parallelism_threads(
            self.config.get("intra_op_parallelism_threads", 0)
        )
    
    def _validate_config(self) -> None:
        """
        Validate the model configuration.
        
        Checks for required parameters, parameter types, and valid values.
        
        Raises:
            ValueError: If required configuration parameters are missing or invalid
        """
        # Required fields
        required_fields = ["mode", "architecture", "learning_rate"]
        for field in required_fields:
            if field not in self.config:
                logger.error(f"Missing required configuration parameter: {field}")
                raise ValueError(f"Missing required configuration parameter: {field}")
        
        # Validate mode
        if self.config["mode"] not in self.SUPPORTED_MODES:
            logger.error(f"Unsupported mode: {self.config['mode']}. " 
                        f"Supported modes are: {self.SUPPORTED_MODES}")
            raise ValueError(f"Unsupported mode: {self.config['mode']}. "
                           f"Supported modes are: {self.SUPPORTED_MODES}")
        
        # Validate architecture
        if self.config["architecture"] not in self.SUPPORTED_ARCHITECTURES:
            logger.error(f"Unsupported architecture: {self.config['architecture']}. "
                        f"Supported architectures are: {self.SUPPORTED_ARCHITECTURES}")
            raise ValueError(f"Unsupported architecture: {self.config['architecture']}. "
                           f"Supported architectures are: {self.SUPPORTED_ARCHITECTURES}")
        
        # Validate numeric parameters
        numeric_params = {
            "sequence_length": (int, (1, None)),
            "prediction_horizon": (int, (1, None)),
            "learning_rate": (float, (0, 1)),
            "dropout_rate": (float, (0, 1)),
            "l1_regularization": (float, (0, 1)),
            "l2_regularization": (float, (0, 1))
        }
        
        for param, (param_type, value_range) in numeric_params.items():
            if param in self.config:
                value = self.config[param]
                
                # Check type
                if not isinstance(value, param_type):
                    logger.error(f"Parameter {param} should be of type {param_type.__name__}, "
                                f"but got {type(value).__name__}")
                    raise ValueError(f"Parameter {param} should be of type {param_type.__name__}, "
                                   f"but got {type(value).__name__}")
                
                # Check range
                if value_range[0] is not None and value < value_range[0]:
                    logger.error(f"Parameter {param} should be >= {value_range[0]}, but got {value}")
                    raise ValueError(f"Parameter {param} should be >= {value_range[0]}, but got {value}")
                
                if value_range[1] is not None and value > value_range[1]:
                    logger.error(f"Parameter {param} should be <= {value_range[1]}, but got {value}")
                    raise ValueError(f"Parameter {param} should be <= {value_range[1]}, but got {value}")
        
        # Validate optional components
        for component in self.config.get("optional_components", []):
            if component not in self.OPTIONAL_COMPONENTS:
                logger.warning(f"Unknown optional component: {component}. "
                              f"Supported components are: {self.OPTIONAL_COMPONENTS}")
        
        # Validate layer configurations
        if "layers" in self.config:
            for layer_config in self.config["layers"]:
                if "type" not in layer_config:
                    logger.error("Each layer configuration must have a 'type' field")
                    raise ValueError("Each layer configuration must have a 'type' field")
        
        # Architecture-specific validations
        if self.config["architecture"] == "transformer":
            if "n_heads" not in self.config:
                logger.error("Transformer architecture requires 'n_heads' parameter")
                raise ValueError("Transformer architecture requires 'n_heads' parameter")
        
        logger.info("Configuration validation completed successfully")
    
    def _setup_components(self) -> None:
        """
        Setup additional model components based on configuration.
        
        Initializes components like uncertainty estimation, ensemble methods,
        and feature attribution.
        """
        optional_components = self.config.get("optional_components", [])
        
        # Uncertainty quantification
        self.uncertainty_enabled = "uncertainty" in optional_components
        if self.uncertainty_enabled:
            uncertainty_method = self.config.get("uncertainty_method", "mc_dropout")
            logger.info(f"Uncertainty quantification enabled with method: {uncertainty_method}")
            
            # Set dropout layers to be active during inference for MC Dropout
            if uncertainty_method == "mc_dropout":
                self.mc_samples = self.config.get("mc_samples", 30)
        
        # Attention mechanisms
        self.attention_enabled = "attention" in optional_components
        if self.attention_enabled:
            attention_type = self.config.get("attention_type", "self_attention")
            logger.info(f"Attention mechanism enabled: {attention_type}")
        
        # Residual connections
        self.residual_enabled = "residual" in optional_components
        if self.residual_enabled:
            logger.info("Residual connections enabled")
        
        # Ensemble methods
        self.ensemble_enabled = "ensemble" in optional_components
        if self.ensemble_enabled:
            self.ensemble_method = self.config.get("ensemble_method", "bagging")
            self.ensemble_size = self.config.get("ensemble_size", 5)
            logger.info(f"Ensemble learning enabled: {self.ensemble_method} with {self.ensemble_size} models")
    
    def _set_random_seeds(self) -> None:
        """
        Set random seeds for reproducibility across libraries.
        """
        seed = self.config.get("random_seed", 42)
        np.random.seed(seed)
        tf.random.set_seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        logger.info(f"Random seeds set to {seed}")
    
    def build_model(self, input_shape: Optional[Tuple[int, int]] = None) -> tf.keras.Model:
        """
        Build the Ariadne neural network model.
        
        Args:
            input_shape: Optional input shape tuple (sequence_length, n_features)
                        If not provided, uses values from config
        
        Returns:
            The compiled TensorFlow Keras model
            
        Raises:
            ValueError: If input_shape is invalid or architecture is not supported
            RuntimeError: If model compilation fails
        """
        # Determine input shape
        if input_shape is not None:
            if len(input_shape) != 2:
                logger.error(f"Input shape must be a tuple of (sequence_length, n_features), got {input_shape}")
                raise ValueError(f"Input shape must be a tuple of (sequence_length, n_features), got {input_shape}")
            sequence_length, n_features = input_shape
        else:
            sequence_length = self.sequence_length
            n_features = self.n_features
            
            if n_features == 0:
                logger.error("Number of features (n_features) not specified in config or input_shape")
                raise ValueError("Number of features (n_features) not specified in config or input_shape")
        
        logger.info(f"Building model with input shape: ({sequence_length}, {n_features})")
        
        # Create the model based on the specified architecture
        try:
            architecture_builders = {
                "cnn_lstm": self._build_cnn_lstm_model,
                "transformer": self._build_transformer_model,
                "hybrid": self._build_hybrid_model,
                "wavenet": self._build_wavenet_model,
                "tcn": self._build_tcn_model
            }
            
            if self.architecture not in architecture_builders:
                logger.error(f"Unsupported architecture: {self.architecture}")
                raise ValueError(f"Unsupported architecture: {self.architecture}")
                
            model = architecture_builders[self.architecture]((sequence_length, n_features))
            
            # Apply model compilation
            self._compile_model(model)
            
            # Store the model
            self.model = model
            
            # Print model summary if verbose
            if self.verbose:
                model.summary(print_fn=logger.info)
                
            # Create ensemble models if enabled
            if self.ensemble_enabled:
                self._create_ensemble_models()
                
            return model
            
        except Exception as e:
            logger.error(f"Failed to build model: {e}")
            raise RuntimeError(f"Failed to build model: {e}")
    
    def _compile_model(self, model: tf.keras.Model) -> None:
        """
        Compile the Keras model with appropriate loss, optimizer, and metrics.
        
        Args:
            model: Keras model to compile
            
        Raises:
            ValueError: If mode is not supported
        """
        # Get configuration parameters
        learning_rate = self.config.get("learning_rate", 0.001)
        optimizer_name = self.config.get("optimizer", "adam")
        
        # Create optimizer
        if optimizer_name.lower() == "adam":
            beta_1 = self.config.get("adam_beta_1", 0.9)
            beta_2 = self.config.get("adam_beta_2", 0.999)
            optimizer = Adam(learning_rate=learning_rate, beta_1=beta_1, beta_2=beta_2)
        elif optimizer_name.lower() == "rmsprop":
            rho = self.config.get("rmsprop_rho", 0.9)
            optimizer = RMSprop(learning_rate=learning_rate, rho=rho)
        elif optimizer_name.lower() == "adamw":
            weight_decay = self.config.get("weight_decay", 1e-4)
            optimizer = tfa.optimizers.AdamW(weight_decay=weight_decay, learning_rate=learning_rate)
        else:
            logger.warning(f"Unsupported optimizer: {optimizer_name}, using Adam instead")
            optimizer = Adam(learning_rate=learning_rate)
        
        # Set loss function based on the mode
        if self.mode == "regression":
            loss_fn_name = self.config.get("loss", "mse")
            if loss_fn_name == "mse":
                loss = MeanSquaredError()
            elif loss_fn_name == "mae":
                loss = tf.keras.losses.MeanAbsoluteError()
            elif loss_fn_name == "huber":
                delta = self.config.get("huber_delta", 1.0)
                loss = tf.keras.losses.Huber(delta=delta)
            elif loss_fn_name == "quantile":
                quantile = self.config.get("quantile", 0.5)
                loss = lambda y_true, y_pred: tf.reduce_mean(
                    tf.maximum(quantile * (y_true - y_pred), (quantile - 1) * (y_true - y_pred))
                )
            else:
                logger.warning(f"Unsupported loss function: {loss_fn_name}, using MSE instead")
                loss = MeanSquaredError()
                
            # Set metrics for regression
            metrics = [
                RootMeanSquaredError(name="rmse"),
                MeanAbsoluteError(name="mae"),
                tf.keras.metrics.MeanAbsolutePercentageError(name="mape")
            ]
            
        elif self.mode == "classification":
            n_classes = self.config.get("n_classes", 2)
            
            if n_classes == 2:
                loss = BinaryCrossentropy(from_logits=False)
                metrics = [
                    tf.keras.metrics.BinaryAccuracy(name="accuracy"),
                    AUC(name="auc"),
                    tf.keras.metrics.Precision(name="precision"),
                    tf.keras.metrics.Recall(name="recall"),
                    tf.keras.metrics.F1Score(name="f1")
                ]
            else:
                loss = CategoricalCrossentropy(from_logits=False)
                metrics = [
                    tf.keras.metrics.CategoricalAccuracy(name="accuracy"),
                    tf.keras.metrics.TopKCategoricalAccuracy(k=3, name="top3_accuracy"),
                    tf.keras.metrics.F1Score(name="f1", average="macro")
                ]
                
        elif self.mode == "multi_task":
            # Multi-task learning with multiple outputs
            losses = {
                "price_prediction": MeanSquaredError(),
                "signal_classification": BinaryCrossentropy(from_logits=False)
            }
            
            loss_weights = {
                "price_prediction": self.config.get("price_weight", 0.5),
                "signal_classification": self.config.get("signal_weight", 0.5)
            }
            
            metrics = {
                "price_prediction": [RootMeanSquaredError(name="rmse"), MeanAbsoluteError(name="mae")],
                "signal_classification": [
                    tf.keras.metrics.BinaryAccuracy(name="accuracy"),
                    AUC(name="auc"),
                    tf.keras.metrics.F1Score(name="f1")
                ]
            }
            
            # Compile the model with multiple losses and metrics
            model.compile(
                optimizer=optimizer,
                loss=losses,
                loss_weights=loss_weights,
                metrics=metrics
            )
            return
            
        else:
            logger.error(f"Unsupported mode: {self.mode}")
            raise ValueError(f"Unsupported mode: {self.mode}")
        
        # Compile the model
        model.compile(
            optimizer=optimizer,
            loss=loss,
            metrics=metrics
        )
    
    def _build_cnn_lstm_model(self, input_shape: Tuple[int, int]) -> tf.keras.Model:
        """
        Build a CNN-LSTM hybrid model for time series prediction.
        
        Args:
            input_shape: Tuple of (sequence_length, n_features)
            
        Returns:
            The built Keras model
        """
        # Extract configuration parameters
        sequence_length, n_features = input_shape
        dropout_rate = self.config.get("dropout_rate", 0.2)
        l1_reg = self.config.get("l1_regularization", 0.0)
        l2_reg = self.config.get("l2_regularization", 0.0)
        n_filters = self.config.get("n_filters", [64, 128, 128])
        kernel_sizes = self.config.get("kernel_sizes", [5, 3, 3])
        lstm_units = self.config.get("lstm_units", [128, 64])
        dense_units = self.config.get("dense_units", [64, 32])
        
        # Input layer
        inputs = Input(shape=input_shape, name='input')
        
        # CNN layers for feature extraction
        x = inputs
        for i, (n_filter, kernel_size) in enumerate(zip(n_filters, kernel_sizes)):
            x = Conv1D(
                filters=n_filter,
                kernel_size=kernel_size,
                padding='same',
                activation=None,
                kernel_initializer=GlorotNormal(),
                kernel_regularizer=l1_l2(l1=l1_reg, l2=l2_reg),
                name=f'conv1d_{i+1}'
            )(x)
            x = BatchNormalization(name=f'batch_norm_conv_{i+1}')(x)
            x = LeakyReLU(alpha=0.1, name=f'leaky_relu_conv_{i+1}')(x)
            x = MaxPooling1D(pool_size=2, name=f'max_pool_{i+1}')(x)
        
        # LSTM layers for temporal dependencies
        for i, units in enumerate(lstm_units):
            return_sequences = i < len(lstm_units) - 1
            x = Bidirectional(
                LSTM(
                    units=units,
                    return_sequences=return_sequences,
                    dropout=dropout_rate,
                    recurrent_dropout=dropout_rate/2,
                    kernel_regularizer=l1_l2(l1=l1_reg, l2=l2_reg),
                    name=f'lstm_{i+1}'
                ),
                name=f'bidirectional_{i+1}'
            )(x)
            
            if return_sequences:
                x = BatchNormalization(name=f'batch_norm_lstm_{i+1}')(x)
        
        # Attention mechanism if enabled
        if self.attention_enabled:
            # Simple self-attention
            attention_scores = Dense(1, activation='tanh', name='attention_dense')(x)
            attention_scores = Reshape((-1,), name='attention_reshape')(attention_scores) 
            attention_weights = Activation('softmax', name='attention_weights')(attention_scores)
            
            # Apply attention
            context_vector = tf.keras.layers.Dot(axes=1, name='attention_dot')([x, attention_weights])
            x = context_vector
        
        # Dense layers for final prediction
        for i, units in enumerate(dense_units):
            x = Dense(
                units=units,
                kernel_regularizer=l1_l2(l1=l1_reg, l2=l2_reg),
                name=f'dense_{i+1}'
            )(x)
            x = BatchNormalization(name=f'batch_norm_dense_{i+1}')(x)
            x = LeakyReLU(alpha=0.1, name=f'leaky_relu_dense_{i+1}')(x)
            x = Dropout(dropout_rate, name=f'dropout_{i+1}')(x)
        
        # Output layer
        if self.mode == "regression":
            outputs = Dense(self.n_targets, activation='linear', name='output')(x)
        elif self.mode == "classification":
            n_classes = self.config.get("n_classes", 2)
            if n_classes == 2:
                outputs = Dense(1, activation='sigmoid', name='output')(x)
            else:
                outputs = Dense(n_classes, activation='softmax', name='output')(x)
        elif self.mode == "multi_task":
            # Multiple outputs for different tasks
            price_output = Dense(self.n_targets, activation='linear', name='price_prediction')(x)
            signal_output = Dense(1, activation='sigmoid', name='signal_classification')(x)
            outputs = [price_output, signal_output]
        
        # Create the model
        model = Model(inputs=inputs, outputs=outputs, name='ariadne_transformer')
        return model
    
    def _transformer_encoder_layer(
        self,
        inputs: tf.Tensor,
        d_model: int,
        n_heads: int,
        dff: int,
        dropout_rate: float,
        l1_reg: float,
        l2_reg: float,
        layer_idx: int
    ) -> tf.Tensor:
        """
        Create a single Transformer encoder layer.
        
        Args:
            inputs: Input tensor
            d_model: Dimension of the model
            n_heads: Number of attention heads
            dff: Dimension of the feedforward network
            dropout_rate: Dropout rate
            l1_reg: L1 regularization factor
            l2_reg: L2 regularization factor
            layer_idx: Layer index for naming
            
        Returns:
            Output tensor from the encoder layer
        """
        # Multi-head attention
        attention_output = MultiHeadAttention(
            num_heads=n_heads,
            key_dim=d_model // n_heads,
            name=f'multi_head_attention_{layer_idx}'
        )(inputs, inputs)
        attention_output = Dropout(dropout_rate, name=f'dropout_attention_{layer_idx}')(attention_output)
        
        # Add & Norm (first residual connection)
        out1 = LayerNormalization(name=f'layer_norm_1_{layer_idx}')(inputs + attention_output)
        
        # Feed-forward network
        ffn_output = Dense(
            dff,
            activation='relu',
            kernel_regularizer=l1_l2(l1=l1_reg, l2=l2_reg),
            name=f'ffn_1_{layer_idx}'
        )(out1)
        ffn_output = Dropout(dropout_rate, name=f'dropout_ffn_1_{layer_idx}')(ffn_output)
        ffn_output = Dense(
            d_model,
            kernel_regularizer=l1_l2(l1=l1_reg, l2=l2_reg),
            name=f'ffn_2_{layer_idx}'
        )(ffn_output)
        ffn_output = Dropout(dropout_rate, name=f'dropout_ffn_2_{layer_idx}')(ffn_output)
        
        # Add & Norm (second residual connection)
        out2 = LayerNormalization(name=f'layer_norm_2_{layer_idx}')(out1 + ffn_output)
        
        return out2
    
    def _positional_encoding(self, max_seq_len: int, d_model: int) -> tf.Tensor:
        """
        Generate positional encoding for the Transformer model.
        
        Args:
            max_seq_len: Maximum sequence length
            d_model: Dimension of the model
            
        Returns:
            Positional encoding tensor of shape (1, max_seq_len, d_model)
        """
        def get_angles(pos, i, d_model):
            angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
            return pos * angle_rates
            
        angle_rads = get_angles(
            np.arange(max_seq_len)[:, np.newaxis],
            np.arange(d_model)[np.newaxis, :],
            d_model
        )
        
        # Apply sin to even indices in the array; 2i
        angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
        
        # Apply cos to odd indices in the array; 2i+1
        angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
        
        pos_encoding = angle_rads[np.newaxis, ...]
        
        return tf.cast(pos_encoding, dtype=tf.float32)
    
    def _build_hybrid_model(self, input_shape: Tuple[int, int]) -> tf.keras.Model:
        """
        Build a hybrid model combining CNN, LSTM, and Attention mechanisms.
        
        This architecture is designed specifically for financial time series,
        incorporating multi-scale pattern recognition and asset correlations.
        
        Args:
            input_shape: Tuple of (sequence_length, n_features)
            
        Returns:
            The built Keras model
        """
        # Extract configuration parameters
        sequence_length, n_features = input_shape
        dropout_rate = self.config.get("dropout_rate", 0.2)
        l1_reg = self.config.get("l1_regularization", 0.0)
        l2_reg = self.config.get("l2_regularization", 0.0)
        n_filters = self.config.get("n_filters", [64, 128, 196])
        kernel_sizes = self.config.get("kernel_sizes", [3, 5, 7])  # Multi-scale kernels
        lstm_units = self.config.get("lstm_units", 128)
        dense_units = self.config.get("dense_units", [128, 64])
        n_heads = self.config.get("n_heads", 4)
        
        # Input layer
        inputs = Input(shape=input_shape, name='input')
        
        # Parallel CNN branches for multi-scale feature extraction
        conv_outputs = []
        for i, (n_filter, kernel_size) in enumerate(zip(n_filters, kernel_sizes)):
            conv_branch = Conv1D(
                filters=n_filter,
                kernel_size=kernel_size,
                padding='same',
                activation=None,
                kernel_initializer=GlorotNormal(),
                kernel_regularizer=l1_l2(l1=l1_reg, l2=l2_reg),
                name=f'conv1d_branch_{i+1}'
            )(inputs)
            conv_branch = BatchNormalization(name=f'batch_norm_conv_{i+1}')(conv_branch)
            conv_branch = LeakyReLU(alpha=0.1, name=f'leaky_relu_conv_{i+1}')(conv_branch)
            
            # Skip connection within each branch if enabled
            if self.residual_enabled and kernel_size <= sequence_length // 4:
                # Projection for matching dimensions
                skip_projection = Conv1D(
                    filters=n_filter,
                    kernel_size=1,
                    padding='same',
                    name=f'skip_projection_{i+1}'
                )(inputs)
                conv_branch = Add(name=f'residual_add_{i+1}')([conv_branch, skip_projection])
                
            conv_outputs.append(conv_branch)
        
        # Concatenate multi-scale features
        if len(conv_outputs) > 1:
            x = Concatenate(axis=-1, name='concat_conv_branches')(conv_outputs)
        else:
            x = conv_outputs[0]
        
        # Bidirectional LSTM for temporal dependencies
        x = Bidirectional(
            LSTM(
                units=lstm_units,
                return_sequences=True,
                dropout=dropout_rate,
                recurrent_dropout=dropout_rate/2,
                kernel_regularizer=l1_l2(l1=l1_reg, l2=l2_reg),
                name='bidirectional_lstm'
            )
        )(x)
        
        # Self-attention mechanism
        if self.attention_enabled:
            # Multi-head self-attention
            attention_output = MultiHeadAttention(
                num_heads=n_heads,
                key_dim=lstm_units // n_heads,
                name='multi_head_attention'
            )(x, x)
            
            # Add & Norm (residual connection)
            x = LayerNormalization(name='layer_norm_attention')(x + attention_output)
        
        # Global feature pooling
        x = GlobalAveragePooling1D(name='global_avg_pooling')(x)
        
        # Dense layers with skip connections
        skip_connection = x
        for i, units in enumerate(dense_units):
            x = Dense(
                units=units,
                kernel_regularizer=l1_l2(l1=l1_reg, l2=l2_reg),
                name=f'dense_{i+1}'
            )(x)
            x = BatchNormalization(name=f'batch_norm_dense_{i+1}')(x)
            x = LeakyReLU(alpha=0.1, name=f'leaky_relu_dense_{i+1}')(x)
            x = Dropout(dropout_rate, name=f'dropout_{i+1}')(x)
            
            # Add skip connection for first dense layer if enabled
            if i == 0 and self.residual_enabled:
                # Projection for matching dimensions
                skip_projection = Dense(units, name='skip_dense_projection')(skip_connection)
                x = Add(name='residual_add_dense')([x, skip_projection])
        
        # Uncertainty estimation if enabled
        if self.uncertainty_enabled:
            uncertainty_method = self.config.get("uncertainty_method", "mc_dropout")
            
            if uncertainty_method == "evidential":
                # Evidential regression for uncertainty
                evidence = Dense(self.n_targets * 4, activation='softplus', name='evidence')(x)
                
                # Split into parameters for the Normal-Inverse-Gamma distribution
                mu = Lambda(lambda x: x[:, :self.n_targets], name='mu')(evidence)
                v = Lambda(lambda x: x[:, self.n_targets:self.n_targets*2] + 1.0, name='v')(evidence)
                alpha = Lambda(lambda x: x[:, self.n_targets*2:self.n_targets*3] + 1.0, name='alpha')(evidence)
                beta = Lambda(lambda x: x[:, self.n_targets*3:], name='beta')(evidence)
                
                # Predicted mean and uncertainty
                outputs = Concatenate(name='output')([mu, v, alpha, beta])
            else:
                # Standard output with MC dropout during inference
                if self.mode == "regression":
                    outputs = Dense(self.n_targets, activation='linear', name='output')(x)
                elif self.mode == "classification":
                    n_classes = self.config.get("n_classes", 2)
                    if n_classes == 2:
                        outputs = Dense(1, activation='sigmoid', name='output')(x)
                    else:
                        outputs = Dense(n_classes, activation='softmax', name='output')(x)
                elif self.mode == "multi_task":
                    price_output = Dense(self.n_targets, activation='linear', name='price_prediction')(x)
                    signal_output = Dense(1, activation='sigmoid', name='signal_classification')(x)
                    outputs = [price_output, signal_output]
        else:
            # Standard output
            if self.mode == "regression":
                outputs = Dense(self.n_targets, activation='linear', name='output')(x)
            elif self.mode == "classification":
                n_classes = self.config.get("n_classes", 2)
                if n_classes == 2:
                    outputs = Dense(1, activation='sigmoid', name='output')(x)
                else:
                    outputs = Dense(n_classes, activation='softmax', name='output')(x)
            elif self.mode == "multi_task":
                price_output = Dense(self.n_targets, activation='linear', name='price_prediction')(x)
                signal_output = Dense(1, activation='sigmoid', name='signal_classification')(x)
                outputs = [price_output, signal_output]
        
        # Create the model
        model = Model(inputs=inputs, outputs=outputs, name='ariadne_hybrid')
        return model
    
    def _build_wavenet_model(self, input_shape: Tuple[int, int]) -> tf.keras.Model:
        """
        Build a WaveNet-inspired model with dilated causal convolutions.
        
        This architecture is especially good for capturing long-range dependencies
        in time series data with its exponentially increasing receptive field.
        
        Args:
            input_shape: Tuple of (sequence_length, n_features)
            
        Returns:
            The built Keras model
        """
        # Extract configuration parameters
        sequence_length, n_features = input_shape
        dropout_rate = self.config.get("dropout_rate", 0.2)
        l1_reg = self.config.get("l1_regularization", 0.0)
        l2_reg = self.config.get("l2_regularization", 0.0)
        n_filters = self.config.get("n_filters", 32)
        n_layers = self.config.get("n_wavenet_layers", 5)
        dense_units = self.config.get("dense_units", [64, 32])
        
        # Input layer
        inputs = Input(shape=input_shape, name='input')
        
        # Initial projection
        x = Conv1D(
            filters=n_filters,
            kernel_size=1,
            padding='causal',
            name='initial_projection'
        )(inputs)
        
        # WaveNet blocks with dilated convolutions
        skip_connections = []
        for i in range(n_layers):
            # Dilated causal convolution
            dilation_rate = 2 ** i
            conv = Conv1D(
                filters=n_filters,
                kernel_size=2,
                padding='causal',
                dilation_rate=dilation_rate,
                activation=None,
                kernel_regularizer=l1_l2(l1=l1_reg, l2=l2_reg),
                name=f'dilated_conv_{i+1}'
            )(x)
            
            # Gated activation unit (as in original WaveNet)
            tanh_out = Conv1D(
                filters=n_filters,
                kernel_size=1,
                activation='tanh',
                name=f'tanh_conv_{i+1}'
            )(conv)
            sigmoid_out = Conv1D(
                filters=n_filters,
                kernel_size=1,
                activation='sigmoid',
                name=f'sigmoid_conv_{i+1}'
            )(conv)
            
            gated_activation = Multiply(name=f'gated_activation_{i+1}')([tanh_out, sigmoid_out])
            
            # 1x1 convolution for residual and skip connections
            res = Conv1D(
                filters=n_filters,
                kernel_size=1,
                name=f'residual_projection_{i+1}'
            )(gated_activation)
            
            # Residual connection
            x = Add(name=f'residual_add_{i+1}')([res, x])
            
            # Skip connection
            skip = Conv1D(
                filters=n_filters,
                kernel_size=1,
                name=f'skip_projection_{i+1}'
            )(gated_activation)
            skip_connections.append(skip)
        
        # Combine skip connections
        if skip_connections:
            x = Add(name='add_skip_connections')(skip_connections)
        
        # Apply ReLU activation
        x = Activation('relu', name='activation_after_skip')(x)
        
        # 1x1 convolution
        x = Conv1D(
            filters=n_filters,
            kernel_size=1,
            activation='relu',
            name='conv_after_skip'
        )(x)
        
        # Global pooling
        x = GlobalAveragePooling1D(name='global_avg_pooling')(x)
        
        # Dense layers
        for i, units in enumerate(dense_units):
            x = Dense(
                units=units,
                kernel_regularizer=l1_l2(l1=l1_reg, l2=l2_reg),
                name=f'dense_{i+1}'
            )(x)
            x = BatchNormalization(name=f'batch_norm_dense_{i+1}')(x)
            x = LeakyReLU(alpha=0.1, name=f'leaky_relu_dense_{i+1}')(x)
            x = Dropout(dropout_rate, name=f'dropout_{i+1}')(x)
        
        # Output layer
        if self.mode == "regression":
            outputs = Dense(self.n_targets, activation='linear', name='output')(x)
        elif self.mode == "classification":
            n_classes = self.config.get("n_classes", 2)
            if n_classes == 2:
                outputs = Dense(1, activation='sigmoid', name='output')(x)
            else:
                outputs = Dense(n_classes, activation='softmax', name='output')(x)
        elif self.mode == "multi_task":
            price_output = Dense(self.n_targets, activation='linear', name='price_prediction')(x)
            signal_output = Dense(1, activation='sigmoid', name='signal_classification')(x)
            outputs = [price_output, signal_output]
        
        # Create the model
        model = Model(inputs=inputs, outputs=outputs, name='ariadne_wavenet')
        return model
    
    def _build_tcn_model(self, input_shape: Tuple[int, int]) -> tf.keras.Model:
        """
        Build a Temporal Convolutional Network (TCN) model.
        
        TCNs are specialized 1D convolutional architectures that 
        outperform LSTMs on many time series tasks.
        
        Args:
            input_shape: Tuple of (sequence_length, n_features)
            
        Returns:
            The built Keras model
        """
        # Extract configuration parameters
        sequence_length, n_features = input_shape
        dropout_rate = self.config.get("dropout_rate", 0.2)
        l1_reg = self.config.get("l1_regularization", 0.0)
        l2_reg = self.config.get("l2_regularization", 0.0)
        n_filters = self.config.get("n_filters", [64, 128, 128])
        kernel_size = self.config.get("tcn_kernel_size", 3)
        n_tcn_stacks = self.config.get("n_tcn_stacks", 3)
        dense_units = self.config.get("dense_units", [64, 32])
        
        # Input layer
        inputs = Input(shape=input_shape, name='input')
        
        # Build TCN model
        x = inputs
        
        # Initial projection
        x = Conv1D(
            filters=n_filters[0],
            kernel_size=1,
            padding='causal',
            name='initial_projection'
        )(x)
        
        # TCN stacks
        for s in range(n_tcn_stacks):
            # Residual connection
            res_x = x
            
            # If the number of filters changes, we need to project the residual
            if s < len(n_filters) - 1 and n_filters[s] != n_filters[s+1]:
                res_x = Conv1D(
                    filters=n_filters[s+1],
                    kernel_size=1,
                    padding='same',
                    name=f'residual_projection_{s+1}'
                )(res_x)
            
            # Dilated causal convolutions with increasing dilation rates
            for i in range(4):  # 4 layers per stack
                dilation_rate = 2 ** i
                
                # Dilated causal convolution
                conv = Conv1D(
                    filters=n_filters[min(s, len(n_filters)-1)],
                    kernel_size=kernel_size,
                    padding='causal',
                    dilation_rate=dilation_rate,
                    activation=None,
                    kernel_regularizer=l1_l2(l1=l1_reg, l2=l2_reg),
                    name=f'tcn_s{s+1}_d{dilation_rate}'
                )(x)
                
                # Normalization and activation
                conv = LayerNormalization(name=f'layer_norm_s{s+1}_d{dilation_rate}')(conv)
                conv = Activation('relu', name=f'relu_s{s+1}_d{dilation_rate}')(conv)
                conv = Dropout(dropout_rate, name=f'dropout_s{s+1}_d{dilation_rate}')(conv)
                
                # Second convolution in the block
                conv = Conv1D(
                    filters=n_filters[min(s, len(n_filters)-1)],
                    kernel_size=kernel_size,
                    padding='causal',
                    dilation_rate=dilation_rate,
                    activation=None,
                    kernel_regularizer=l1_l2(l1=l1_reg, l2=l2_reg),
                    name=f'tcn_s{s+1}_d{dilation_rate}_2'
                )(conv)
                
                # Normalization and activation
                conv = LayerNormalization(name=f'layer_norm_s{s+1}_d{dilation_rate}_2')(conv)
                conv = Activation('relu', name=f'relu_s{s+1}_d{dilation_rate}_2')(conv)
                conv = Dropout(dropout_rate, name=f'dropout_s{s+1}_d{dilation_rate}_2')(conv)
                
                # Skip connection
                x = Add(name=f'skip_add_s{s+1}_d{dilation_rate}')([x, conv])
            
            # Add residual connection for the stack
            x = Add(name=f'residual_add_stack_{s+1}')([x, res_x])
        
        # Global pooling
        x = GlobalAveragePooling1D(name='global_avg_pooling')(x)
        
        # Dense layers
        for i, units in enumerate(dense_units):
            x = Dense(
                units=units,
                kernel_regularizer=l1_l2(l1=l1_reg, l2=l2_reg),
                name=f'dense_{i+1}'
            )(x)
            x = BatchNormalization(name=f'batch_norm_dense_{i+1}')(x)
            x = LeakyReLU(alpha=0.1, name=f'leaky_relu_dense_{i+1}')(x)
            x = Dropout(dropout_rate, name=f'dropout_{i+1}')(x)
        
        # Output layer
        if self.mode == "regression":
            outputs = Dense(self.n_targets, activation='linear', name='output')(x)
        elif self.mode == "classification":
            n_classes = self.config.get("n_classes", 2)
            if n_classes == 2:
                outputs = Dense(1, activation='sigmoid', name='output')(x)
            else:
                outputs = Dense(n_classes, activation='softmax', name='output')(x)
        elif self.mode == "multi_task":
            price_output = Dense(self.n_targets, activation='linear', name='price_prediction')(x)
            signal_output = Dense(1, activation='sigmoid', name='signal_classification')(x)
            outputs = [price_output, signal_output]
        
        # Create the model
        model = Model(inputs=inputs, outputs=outputs, name='ariadne_tcn')
        return model
    
    def _create_ensemble_models(self) -> None:
        """
        Create an ensemble of models for improved prediction accuracy.
        
        This method implements various ensemble techniques like bagging,
        boosting, and stacking based on the configuration.
        """
        if not self.ensemble_enabled:
            return
            
        logger.info(f"Creating ensemble with {self.ensemble_size} models using {self.ensemble_method} method")
        
        # Create ensemble models
        self.ensemble_models = []
        
        base_model = self.model
        
        for i in range(self.ensemble_size):
            # Clone the base model
            ensemble_model = clone_model(base_model)
            
            # Recompile the model
            self._compile_model(ensemble_model)
            
            # Store the model
            self.ensemble_models.append(ensemble_model)
            
        logger.info(f"Successfully created {len(self.ensemble_models)} ensemble models")
    
    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        batch_size: Optional[int] = None,
        epochs: Optional[int] = None,
        callbacks: Optional[List[tf.keras.callbacks.Callback]] = None,
        sample_weight: Optional[np.ndarray] = None,
        class_weight: Optional[Dict[int, float]] = None,
        validation_freq: int = 1,
        use_multiprocessing: bool = False,
        workers: int = 1,
        **kwargs
    ) -> Union[tf.keras.callbacks.History, List[tf.keras.callbacks.History]]:
        """
        Train the model on the provided data.
        
        Args:
            X_train: Training data of shape (samples, sequence_length, features)
            y_train: Training labels of shape (samples, n_targets) for regression
                    or (samples, 1) for binary classification
                    or (samples, n_classes) for multi-class classification
            X_val: Validation data with same shape as X_train
            y_val: Validation labels with same shape as y_train
            batch_size: Batch size for training
            epochs: Number of epochs to train
            callbacks: List of Keras callbacks
            sample_weight: Optional array of weights for training samples
            class_weight: Optional dictionary mapping class indices to weights
            validation_freq: Frequency (in epochs) at which validation is performed
            use_multiprocessing: Whether to use multiprocessing during training
            workers: Number of worker processes for multiprocessing
            **kwargs: Additional arguments to pass to model.fit()
            
        Returns:
            History object(s) with training metrics
            
        Raises:
            ValueError: If model is not built or input shapes are incorrect
            RuntimeError: If training fails
        """
        if self.model is None:
            logger.error("Model not built. Call build_model() before fit()")
            raise ValueError("Model not built. Call build_model() before fit()")
            
        # If input shape not set, determine from training data
        if self.n_features == 0:
            if len(X_train.shape) != 3:
                logger.error(f"Expected 3D input (samples, sequence_length, features), got shape {X_train.shape}")
                raise ValueError(f"Expected 3D input (samples, sequence_length, features), got shape {X_train.shape}")
                
            self.sequence_length, self.n_features = X_train.shape[1:]
            logger.info(f"Determined input shape: ({self.sequence_length}, {self.n_features})")
            
        # Get configuration parameters or use defaults
        batch_size = batch_size or self.config.get("batch_size", 32)
        epochs = epochs or self.config.get("epochs", 100)
        
        # Record the training start time
        training_start_time = datetime.datetime.now()
        
        # Setup callbacks
        if callbacks is None:
            callbacks = self._get_default_callbacks()
        
        try:
            logger.info(f"Starting model training for {epochs} epochs with batch size {batch_size}")
            
            # Train the main model
            if self.ensemble_enabled and self.ensemble_method == "bagging":
                # Bagging: Train each model on a bootstrap sample
                histories = []
                
                for i, model in enumerate(self.ensemble_models):
                    logger.info(f"Training ensemble model {i+1}/{len(self.ensemble_models)}")
                    
                    # Create bootstrap sample
                    n_samples = X_train.shape[0]
                    bootstrap_indices = np.random.choice(n_samples, size=n_samples, replace=True)
                    X_bootstrap = X_train[bootstrap_indices]
                    y_bootstrap = y_train[bootstrap_indices]
                    
                    # Create model-specific callbacks
                    model_callbacks = self._get_default_callbacks(suffix=f"_ensemble_{i+1}")
                    
                    # Train the model
                    history = model.fit(
                        X_bootstrap, y_bootstrap,
                        validation_data=(X_val, y_val) if X_val is not None and y_val is not None else None,
                        batch_size=batch_size,
                        epochs=epochs,
                        callbacks=model_callbacks,
                        verbose=1 if self.verbose else 0,
                        sample_weight=sample_weight[bootstrap_indices] if sample_weight is not None else None,
                        **kwargs
                    )
                    
                    histories.append(history)
                
                # Store the histories
                self.training_history = histories
                training_result = histories
                
            else:
                # Standard training for the main model
                history = self.model.fit(
                    X_train, y_train,
                    validation_data=(X_val, y_val) if X_val is not None and y_val is not None else None,
                    batch_size=batch_size,
                    epochs=epochs,
                    callbacks=callbacks,
                    verbose=1 if self.verbose else 0,
                    sample_weight=sample_weight,
                    **kwargs
                )
                
                # Store the history
                self.training_history = history
                training_result = history
                
                # For other ensemble methods, we might need to train after the main model
                if self.ensemble_enabled:
                    if self.ensemble_method == "snapshot":
                        # Snapshot ensemble: Use checkpoints from training
                        for i, weight_path in enumerate(sorted(self.model_dir.glob("*snapshot*.h5"))):
                            if i < self.ensemble_size:
                                self.ensemble_models[i].load_weights(str(weight_path))
                    
                    elif self.ensemble_method == "boosting":
                        # Simple boosting implementation
                        for i, model in enumerate(self.ensemble_models):
                            logger.info(f"Training boosting ensemble model {i+1}/{len(self.ensemble_models)}")
                            
                            # Predict with current model
                            preds = self.predict(X_train)
                            
                            # Compute errors
                            if self.mode == "regression":
                                errors = np.abs(y_train - preds)
                            else:
                                errors = (y_train != (preds > 0.5).astype(int)).astype(float)
                            
                            # Create sample weights emphasizing errors
                            sample_weights = errors / np.sum(errors)
                            
                            # Create model-specific callbacks
                            model_callbacks = self._get_default_callbacks(suffix=f"_ensemble_{i+1}")
                            
                            # Train the model with updated weights
                            model.fit(
                                X_train, y_train,
                                validation_data=(X_val, y_val) if X_val is not None and y_val is not None else None,
                                batch_size=batch_size,
                                epochs=epochs // 2,  # Shorter training for ensemble models
                                callbacks=model_callbacks,
                                verbose=0,
                                sample_weight=sample_weights,
                                **kwargs
                            )
            
            # Record the training end time
            self.last_training_time = datetime.datetime.now() - training_start_time
            logger.info(f"Model training completed in {self.last_training_time}")
            
            return training_result
            
        except Exception as e:
            logger.error(f"Training failed: {e}")
            raise RuntimeError(f"Training failed: {e}")
    
    def _get_default_callbacks(self, suffix: str = "") -> List[tf.keras.callbacks.Callback]:
        """
        Create default callbacks for model training.
        
        Args:
            suffix: Optional suffix for callback names and file paths
            
        Returns:
            List of Keras callbacks
        """
        callbacks = []
        
        # Extract configuration parameters
        patience = self.config.get("early_stopping_patience", 10)
        min_delta = self.config.get("early_stopping_min_delta", 0.001)
        reduce_lr_factor = self.config.get("reduce_lr_factor", 0.5)
        reduce_lr_patience = self.config.get("reduce_lr_patience", 5)
        
        # Create model checkpoint callback
        checkpoint_path = str(self.model_dir / f"model_checkpoint{suffix}.h5")
        checkpoint_callback = ModelCheckpoint(
            filepath=checkpoint_path,
            monitor='val_loss' if self.mode != 'multi_task' else 'val_loss',
            save_best_only=True,
            save_weights_only=True,
            verbose=1 if self.verbose else 0,
            mode='min'
        )
        callbacks.append(checkpoint_callback)
        
        # Create early stopping callback
        early_stopping = EarlyStopping(
            monitor='val_loss' if self.mode != 'multi_task' else 'val_loss',
            patience=patience,
            min_delta=min_delta,
            verbose=1 if self.verbose else 0,
            mode='min',
            restore_best_weights=True
        )
        callbacks.append(early_stopping)
        
        # Create learning rate scheduler callback
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss' if self.mode != 'multi_task' else 'val_loss',
            factor=reduce_lr_factor,
            patience=reduce_lr_patience,
            min_lr=1e-6,
            verbose=1 if self.verbose else 0,
            mode='min'
        )
        callbacks.append(reduce_lr)
        
        # Create TensorBoard callback
        log_dir = str(self.model_dir / f"logs{suffix}")
        tensorboard_callback = TensorBoard(
            log_dir=log_dir,
            histogram_freq=1,
            write_graph=True,
            update_freq='epoch'
        )
        callbacks.append(tensorboard_callback)
        
        # Create CSV logger callback
        csv_path = str(self.model_dir / f"training_log{suffix}.csv")
        csv_logger = CSVLogger(
            filename=csv_path,
            separator=',',
            append=True
        )
        callbacks.append(csv_logger)
        
        # Add snapshot ensemble callback if enabled
        if self.ensemble_enabled and self.ensemble_method == "snapshot":
            snapshot_dir = str(self.model_dir / "snapshots")
            Path(snapshot_dir).mkdir(exist_ok=True)
            
            # Custom callback for snapshot ensemble
            class SnapshotEnsembleCallback(tf.keras.callbacks.Callback):
                def __init__(self, n_snapshots, snapshot_dir):
                    super().__init__()
                    self.n_snapshots = n_snapshots
                    self.snapshot_dir = snapshot_dir
                    
                def on_epoch_end(self, epoch, logs=None):
                    # Save snapshot at regular intervals
                    epochs_per_snapshot = self.params['epochs'] // self.n_snapshots
                    if (epoch + 1) % epochs_per_snapshot == 0 or epoch == self.params['epochs'] - 1:
                        snapshot_path = f"{self.snapshot_dir}/snapshot_epoch_{epoch+1}.h5"
                        self.model.save_weights(snapshot_path)
                        logger.info(f"Saved snapshot at epoch {epoch+1}")
            
            snapshot_callback = SnapshotEnsembleCallback(
                n_snapshots=self.ensemble_size,
                snapshot_dir=snapshot_dir
            )
            callbacks.append(snapshot_callback)
        
        return callbacks
    
    def evaluate(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray,
        batch_size: Optional[int] = None,
        **kwargs
    ) -> Union[Dict[str, float], List[Dict[str, float]]]:
        """
        Evaluate the model on test data.
        
        Args:
            X_test: Test data
            y_test: Test labels
            batch_size: Batch size for evaluation
            **kwargs: Additional arguments to pass to model.evaluate()
            
        Returns:
            Dictionary mapping metric names to values
            
        Raises:
            ValueError: If model is not built
            RuntimeError: If evaluation fails
        """
        if self.model is None:
            logger.error("Model not built. Call build_model() before evaluate()")
            raise ValueError("Model not built. Call build_model() before evaluate()")
        
        # Use default batch size if not provided
        batch_size = batch_size or self.config.get("batch_size", 32)
        
        try:
            logger.info("Evaluating model on test data")
            
            if self.ensemble_enabled and len(self.ensemble_models) > 0:
                # Evaluate each ensemble model
                ensemble_metrics = []
                
                for i, model in enumerate(self.ensemble_models):
                    logger.info(f"Evaluating ensemble model {i+1}/{len(self.ensemble_models)}")
                    metrics = model.evaluate(
                        X_test, y_test,
                        batch_size=batch_size,
                        verbose=0,
                        **kwargs
                    )
                    
                    # Convert metrics to dictionary
                    metric_names = model.metrics_names
                    metrics_dict = {name: value for name, value in zip(metric_names, metrics)}
                    ensemble_metrics.append(metrics_dict)
                
                # Calculate ensemble average metrics
                avg_metrics = {}
                for metric in ensemble_metrics[0].keys():
                    avg_metrics[metric] = np.mean([m[metric] for m in ensemble_metrics])
                    avg_metrics[f"{metric}_std"] = np.std([m[metric] for m in ensemble_metrics])
                
                # Store evaluation results
                self.model_evaluation = {
                    "ensemble_models": ensemble_metrics,
                    "ensemble_average": avg_metrics
                }
                
                if self.verbose:
                    logger.info("Ensemble evaluation results:")
                    for metric, value in avg_metrics.items():
                        logger.info(f"{metric}: {value:.4f}")
                
                return self.model_evaluation
            else:
                # Evaluate the main model
                metrics = self.model.evaluate(
                    X_test, y_test,
                    batch_size=batch_size,
                    verbose=1 if self.verbose else 0,
                    **kwargs
                )
                
                # Convert metrics to dictionary
                metric_names = self.model.metrics_names
                metrics_dict = {name: value for name, value in zip(metric_names, metrics)}
                
                # Store evaluation results
                self.model_evaluation = metrics_dict
                
                if self.verbose:
                    logger.info("Evaluation results:")
                    for metric, value in metrics_dict.items():
                        logger.info(f"{metric}: {value:.4f}")
                
                return metrics_dict
                
        except Exception as e:
            logger.error(f"Evaluation failed: {e}")
            raise RuntimeError(f"Evaluation failed: {e}")
    
    def predict(
        self,
        X: np.ndarray,
        batch_size: Optional[int] = None,
        return_uncertainty: bool = False,
        **kwargs
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Generate predictions for the input data.
        
        Args:
            X: Input data
            batch_size: Batch size for prediction
            return_uncertainty: If True, return uncertainty estimates
            **kwargs: Additional arguments to pass to model.predict()
            
        Returns:
            Predictions, or tuple of (predictions, uncertainty) if return_uncertainty=True
            
        Raises:
            ValueError: If model is not built
            RuntimeError: If prediction fails
        """
        if self.model is None:
            logger.error("Model not built. Call build_model() before predict()")
            raise ValueError("Model not built. Call build_model() before predict()")
        
        # Use default batch size if not provided
        batch_size = batch_size or self.config.get("batch_size", 32)
        
        try:
            # If using ensemble, perform ensemble prediction
            if self.ensemble_enabled and len(self.ensemble_models) > 0:
                ensemble_preds = []
                
                for i, model in enumerate(self.ensemble_models):
                    preds = model.predict(X, batch_size=batch_size, verbose=0, **kwargs)
                    ensemble_preds.append(preds)
                
                # Calculate ensemble mean and standard deviation
                ensemble_preds = np.array(ensemble_preds)
                mean_preds = np.mean(ensemble_preds, axis=0)
                std_preds = np.std(ensemble_preds, axis=0)
                
                if return_uncertainty:
                    return mean_preds, std_preds
                else:
                    return mean_preds
            
            # If using MC Dropout for uncertainty estimation
            elif self.uncertainty_enabled and return_uncertainty and self.config.get("uncertainty_method", "mc_dropout") == "mc_dropout":
                mc_samples = self.config.get("mc_samples", 30)
                
                # Function to get model predictions with dropout enabled
                def predict_with_dropout():
                    return self.model(X, training=True)
                
                # Collect MC Dropout samples
                mc_preds = []
                for _ in range(mc_samples):
                    preds = predict_with_dropout()
                    mc_preds.append(preds)
                
                # Calculate mean and standard deviation
                mc_preds = np.array(mc_preds)
                mean_preds = np.mean(mc_preds, axis=0)
                std_preds = np.std(mc_preds, axis=0)
                
                return mean_preds, std_preds
            
            # If using evidential networks for uncertainty estimation
            elif self.uncertainty_enabled and return_uncertainty and self.config.get("uncertainty_method", "mc_dropout") == "evidential":
                # Evidential networks output distribution parameters directly
                params = self.model.predict(X, batch_size=batch_size, verbose=0, **kwargs)
                
                # Extract mean and uncertainty (variance or standard deviation)
                if self.mode == "regression":
                    mean = params[:, :self.n_targets]
                    variance = params[:, 2*self.n_targets:3*self.n_targets] / (params[:, 3*self.n_targets:] - 1)
                    std = np.sqrt(variance)
                    
                    return mean, std
                else:
                    # For classification, use predictive entropy as uncertainty
                    mean = params
                    entropy = -np.sum(mean * np.log(mean + 1e-10), axis=-1, keepdims=True)
                    
                    return mean, entropy
            
            # Standard prediction
            else:
                preds = self.model.predict(X, batch_size=batch_size, verbose=0, **kwargs)
                return preds
                
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            raise RuntimeError(f"Prediction failed: {e}")
    
    def save(self, model_path: Optional[str] = None) -> str:
        """
        Save the model weights and configuration.
        
        Args:
            model_path: Path to save the model (without extension)
                       If None, a path is generated based on timestamp
        
        Returns:
            Path to the saved model weights
            
        Raises:
            ValueError: If model is not built
            RuntimeError: If saving fails
        """
        if self.model is None:
            logger.error("Model not built. Call build_model() before save()")
            raise ValueError("Model not built. Call build_model() before save()")
        
        try:
            # Generate path if not provided
            if model_path is None:
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                model_path = str(self.model_dir / f"ariadne_model_{timestamp}")
            
            # Save model weights
            weights_path = f"{model_path}.h5"
            self.model.save_weights(weights_path)
            
            # Save configuration
            config_path = f"{model_path}_config.json"
            with open(config_path, 'w') as f:
                # Add model info to config
                save_config = self.config.copy()
                save_config["model_version"] = self.MODEL_VERSION
                save_config["model_architecture"] = self.architecture
                save_config["mode"] = self.mode
                save_config["feature_names"] = self.feature_names
                save_config["n_features"] = self.n_features
                save_config["n_targets"] = self.n_targets
                save_config["sequence_length"] = self.sequence_length
                
                # Save metrics if available
                if self.model_evaluation:
                    save_config["evaluation_metrics"] = self.model_evaluation
                
                # Save training time if available
                if self.last_training_time:
                    save_config["training_time_seconds"] = self.last_training_time.total_seconds()
                
                json.dump(save_config, f, indent=4)
            
            # Save ensemble models if enabled
            if self.ensemble_enabled and len(self.ensemble_models) > 0:
                ensemble_dir = Path(f"{model_path}_ensemble")
                ensemble_dir.mkdir(exist_ok=True)
                
                for i, model in enumerate(self.ensemble_models):
                    ensemble_weights_path = str(ensemble_dir / f"ensemble_model_{i+1}.h5")
                    model.save_weights(ensemble_weights_path)
                
                logger.info(f"Saved {len(self.ensemble_models)} ensemble models to {ensemble_dir}")
            
            logger.info(f"Model saved to {weights_path}")
            logger.info(f"Configuration saved to {config_path}")
            
            return weights_path
            
        except Exception as e:
            logger.error(f"Failed to save model: {e}")
            raise RuntimeError(f"Failed to save model: {e}")
    
    def load(
        self,
        model_path: str,
        load_weights: bool = True,
        custom_objects: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Load the model weights and configuration.
        
        Args:
            model_path: Path to the saved model (without extension)
            load_weights: Whether to load model weights
            custom_objects: Dictionary mapping names to custom classes or functions
            
        Raises:
            FileNotFoundError: If model files are not found
            RuntimeError: If loading fails
        """
        try:
            # Load configuration
            config_path = f"{model_path}_config.json"
            if not os.path.exists(config_path):
                logger.error(f"Configuration file not found: {config_path}")
                raise FileNotFoundError(f"Configuration file not found: {config_path}")
            
            with open(config_path, 'r') as f:
                loaded_config = json.load(f)
            
            # Update instance configuration
            self.config.update(loaded_config)
            
            # Extract model parameters
            self.architecture = loaded_config.get("model_architecture", self.architecture)
            self.mode = loaded_config.get("mode", self.mode)
            self.feature_names = loaded_config.get("feature_names", self.feature_names)
            self.n_features = loaded_config.get("n_features", self.n_features)
            self.n_targets = loaded_config.get("n_targets", self.n_targets)
            self.sequence_length = loaded_config.get("sequence_length", self.sequence_length)
            
            # Rebuild the model
            if self.n_features > 0 and self.sequence_length > 0:
                input_shape = (self.sequence_length, self.n_features)
                self.build_model(input_shape)
            
                # Load weights if requested
                if load_weights:
                    weights_path = f"{model_path}.h5"
                    if not os.path.exists(weights_path):
                        logger.error(f"Weights file not found: {weights_path}")
                        raise FileNotFoundError(f"Weights file not found: {weights_path}")
                    
                    self.model.load_weights(weights_path)
                    logger.info(f"Loaded model weights from {weights_path}")
                    
                    # Load ensemble models if available
                    ensemble_dir = Path(f"{model_path}_ensemble")
                    if ensemble_dir.exists() and self.ensemble_enabled:
                        self._create_ensemble_models()
                        
                        for i, model in enumerate(self.ensemble_models):
                            ensemble_weights_path = str(ensemble_dir / f"ensemble_model_{i+1}.h5")
                            if os.path.exists(ensemble_weights_path):
                                model.load_weights(ensemble_weights_path)
                                logger.info(f"Loaded ensemble model {i+1} weights from {ensemble_weights_path}")
            else:
                logger.warning("Could not rebuild model: missing input shape parameters")
            
            logger.info(f"Model configuration loaded from {config_path}")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise RuntimeError(f"Failed to load model: {e}")
    
    def feature_importance(
        self,
        X: np.ndarray,
        method: str = "permutation",
        n_repeats: int = 10,
        **kwargs
    ) -> Dict[str, np.ndarray]:
        """
        Calculate feature importance scores.
        
        Args:
            X: Input data for calculating importance
            method: Method for calculating importance
                   Options: "permutation", "shap", "integrated_gradients"
            n_repeats: Number of times to repeat permutation importance
            **kwargs: Additional method-specific arguments
            
        Returns:
            Dictionary mapping feature names to importance scores
            
        Raises:
            ValueError: If method is not supported or model is not built
            RuntimeError: If calculation fails
        """
        if self.model is None:
            logger.error("Model not built. Call build_model() before feature_importance()")
            raise ValueError("Model not built. Call build_model() before feature_importance()")
        
        # Ensure we have feature names
        if not self.feature_names:
            self.feature_names = [f"feature_{i+1}" for i in range(self.n_features)]
        
        try:
            logger.info(f"Calculating feature importance using {method} method")
            
            if method == "permutation":
                # Permutation importance
                y_pred = self.predict(X)
                baseline_score = np.mean((y_pred - kwargs.get("y_true", y_pred)) ** 2)
                
                importance_scores = np.zeros((n_repeats, self.n_features))
                
                for r in range(n_repeats):
                    for i in range(self.n_features):
                        # Create a copy and permute the feature
                        X_permuted = X.copy()
                        
                        # Permute the feature across all time steps
                        permuted_idx = np.random.permutation(X.shape[0])
                        X_permuted[:, :, i] = X[permuted_idx, :, i]
                        
                        # Predict with permuted feature
                        y_pred_permuted = self.predict(X_permuted)
                        permuted_score = np.mean((y_pred_permuted - kwargs.get("y_true", y_pred)) ** 2)
                        
                        # Calculate importance as the difference in scores
                        importance_scores[r, i] = permuted_score - baseline_score
                
                # Average across repeats
                mean_importance = np.mean(importance_scores, axis=0)
                std_importance = np.std(importance_scores, axis=0)
                
                # Normalize to sum to 1
                mean_importance = mean_importance / np.sum(np.abs(mean_importance))
                
                # Create dictionary of feature importances
                result = {
                    "importance_mean": mean_importance,
                    "importance_std": std_importance,
                    "feature_names": self.feature_names,
                    "method": method
                }
                
                return result
                
            elif method == "shap":
                try:
                    import shap
                except ImportError:
                    logger.error("SHAP package not installed. Install with: pip install shap")
                    raise ImportError("SHAP package not installed. Install with: pip install shap")
                
                # Create explainer
                explainer = shap.DeepExplainer(self.model, X[:kwargs.get("background_samples", 100)])
                
                # Calculate SHAP values
                shap_values = explainer.shap_values(X[:kwargs.get("explanation_samples", 1000)])
                
                # Average SHAP values across samples and time steps
                if isinstance(shap_values, list):
                    # For multi-output models
                    mean_importance = np.mean([np.abs(sv).mean(axis=(0, 1)) for sv in shap_values], axis=0)
                else:
                    mean_importance = np.abs(shap_values).mean(axis=(0, 1))
                
                # Normalize to sum to 1
                mean_importance = mean_importance / np.sum(mean_importance)
                
                # Create dictionary of feature importances
                result = {
                    "importance_mean": mean_importance,
                    "importance_std": np.zeros_like(mean_importance),  # Not available for SHAP
                    "feature_names": self.feature_names,
                    "method": method,
                    "shap_values": shap_values
                }
                
                return result
                
            elif method == "integrated_gradients":
                # Implement integrated gradients method
                # This is a simplified version; a proper implementation would use TF's GradientTape
                baseline = np.zeros_like(X[:1])
                steps = kwargs.get("steps", 50)
                
                # Interpolate between baseline and input
                alphas = np.linspace(0, 1, steps+1)
                interpolated = np.array([baseline + alpha * (X[:1] - baseline) for alpha in alphas])
                interpolated = interpolated.reshape(-1, X.shape[1], X.shape[2])
                
                # Calculate gradients
                with tf.GradientTape() as tape:
                    inputs = tf.convert_to_tensor(interpolated)
                    tape.watch(inputs)
                    outputs = self.model(inputs)
                    
                gradients = tape.gradient(outputs, inputs)
                gradients = gradients.numpy()
                
                # Reshape gradients and calculate integrated gradients
                gradients = gradients.reshape(steps+1, -1, X.shape[1], X.shape[2])
                avg_gradients = (gradients[:-1] + gradients[1:]) / 2
                integrated_gradients = avg_gradients.mean(axis=0) * (X[:1] - baseline)
                
                # Average across samples and time steps
                mean_importance = np.abs(integrated_gradients).mean(axis=(0, 1))
                
                # Normalize to sum to 1
                mean_importance = mean_importance / np.sum(mean_importance)
                
                # Create dictionary of feature importances
                result = {
                    "importance_mean": mean_importance,
                    "importance_std": np.zeros_like(mean_importance),  # Not available for this method
                    "feature_names": self.feature_names,
                    "method": method
                }
                
                return result
            
            else:
                logger.error(f"Unsupported feature importance method: {method}")
                raise ValueError(f"Unsupported feature importance method: {method}")
                
        except Exception as e:
            logger.error(f"Feature importance calculation failed: {e}")
            raise RuntimeError(f"Feature importance calculation failed: {e}")
    
    def visualize_feature_importance(
        self,
        importance_data: Dict[str, np.ndarray],
        top_n: int = 10,
        save_path: Optional[str] = None
    ) -> None:
        """
        Visualize feature importance scores.
        
        Args:
            importance_data: Dictionary from feature_importance() method
            top_n: Number of top features to display
            save_path: Optional path to save the visualization
            
        Raises:
            ValueError: If importance_data format is invalid
            RuntimeError: If visualization fails
        """
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            # Extract data
            importance_mean = importance_data.get("importance_mean")
            importance_std = importance_data.get("importance_std")
            feature_names = importance_data.get("feature_names", self.feature_names)
            method = importance_data.get("method", "unknown")
            
            if importance_mean is None or feature_names is None:
                logger.error("Invalid importance data format")
                raise ValueError("Invalid importance data format")
            
            # Sort features by importance
            indices = np.argsort(importance_mean)[-top_n:]
            
            plt.figure(figsize=(10, 6))
            
            # Plot horizontal bar chart
            plt.barh(
                range(len(indices)),
                importance_mean[indices],
                xerr=importance_std[indices] if importance_std is not None else None,
                align='center',
                alpha=0.8,
                capsize=5
            )
            
            # Add feature names to y-axis
            plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
            
            # Add labels and title
            plt.xlabel('Importance Score')
            plt.ylabel('Feature')
            plt.title(f'Top {top_n} Features by {method.capitalize()} Importance')
            
            # Add grid lines
            plt.grid(axis='x', linestyle='--', alpha=0.7)
            
            # Tight layout
            plt.tight_layout()
            
            # Save figure if path provided
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"Feature importance visualization saved to {save_path}")
            
            # Show plot
            plt.show()
            
        except ImportError:
            logger.warning("Matplotlib or seaborn not installed. Install with: pip install matplotlib seaborn")
        except Exception as e:
            logger.error(f"Visualization failed: {e}")
            raise RuntimeError(f"Visualization failed: {e}")
    
    def visualize_attention(
        self,
        X: np.ndarray,
        sample_idx: int = 0,
        layer_name: Optional[str] = None,
        head_idx: Optional[int] = None,
        save_path: Optional[str] = None
    ) -> None:
        """
        Visualize attention weights for a specific sample.
        
        Args:
            X: Input data
            sample_idx: Index of the sample to visualize
            layer_name: Name of the attention layer to visualize
                       If None, the first attention layer is used
            head_idx: Index of the attention head to visualize
                     If None, all heads are averaged
            save_path: Optional path to save the visualization
            
        Raises:
            ValueError: If no attention layer found or sample_idx is invalid
            RuntimeError: If visualization fails
        """
        if not self.attention_enabled:
            logger.warning("Attention visualization requested but attention is not enabled in the model")
            return
        
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            # Check sample index
            if sample_idx >= X.shape[0]:
                logger.error(f"Sample index {sample_idx} out of range (0-{X.shape[0]-1})")
                raise ValueError(f"Sample index {sample_idx} out of range (0-{X.shape[0]-1})")
            
            # Find attention layers
            attention_layers = [layer for layer in self.model.layers if 'attention' in layer.name.lower()]
            
            if not attention_layers:
                logger.error("No attention layer found in the model")
                raise ValueError("No attention layer found in the model")
            
            # Use specified layer or first attention layer
            if layer_name is not None:
                target_layer = next((layer for layer in attention_layers if layer.name == layer_name), None)
                if target_layer is None:
                    logger.error(f"Attention layer '{layer_name}' not found")
                    raise ValueError(f"Attention layer '{layer_name}' not found")
            else:
                target_layer = attention_layers[0]
                layer_name = target_layer.name
            
            # Create a new model to extract attention weights
            attention_model = Model(
                inputs=self.model.input,
                outputs=target_layer.output
            )
            
            # Get attention weights
            attention_weights = attention_model.predict(X[sample_idx:sample_idx+1])
            
            # Process attention weights based on layer type
            if isinstance(target_layer, MultiHeadAttention):
                # MultiHeadAttention returns attention scores as the second output
                if isinstance(attention_weights, tuple) and len(attention_weights) > 1:
                    attention_weights = attention_weights[1]
                
                # Select head if specified, otherwise average across heads
                if head_idx is not None:
                    if head_idx >= attention_weights.shape[1]:
                        logger.error(f"Head index {head_idx} out of range (0-{attention_weights.shape[1]-1})")
                        raise ValueError(f"Head index {head_idx} out of range (0-{attention_weights.shape[1]-1})")
                    
                    attention_matrix = attention_weights[0, head_idx]
                    title = f"Attention Weights - Layer: {layer_name}, Head: {head_idx}"
                else:
                    attention_matrix = np.mean(attention_weights[0], axis=0)
                    title = f"Attention Weights - Layer: {layer_name}, All Heads (Averaged)"
            else:
                # For custom attention implementations
                attention_matrix = attention_weights[0]
                title = f"Attention Weights - Layer: {layer_name}"
            
            # Visualize attention matrix
            plt.figure(figsize=(10, 8))
            ax = sns.heatmap(
                attention_matrix,
                cmap='viridis',
                annot=False,
                square=True
            )
            
            # Add labels and title
            plt.xlabel('Key Position')
            plt.ylabel('Query Position')
            plt.title(title)
            
            # Save figure if path provided
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"Attention visualization saved to {save_path}")
            
            # Show plot
            plt.show()
            
        except ImportError:
            logger.warning("Matplotlib or seaborn not installed. Install with: pip install matplotlib seaborn")
        except Exception as e:
            logger.error(f"Attention visualization failed: {e}")
            raise RuntimeError(f"Attention visualization failed: {e}")
    
    def export_to_onnx(self, export_path: str, **kwargs) -> str:
        """
        Export the model to ONNX format for deployment.
        
        Args:
            export_path: Path to save the ONNX model (without extension)
            **kwargs: Additional arguments to pass to tf.keras.Model.save()
            
        Returns:
            Path to the exported ONNX model
            
        Raises:
            ValueError: If model is not built
            RuntimeError: If export fails
        """
        if self.model is None:
            logger.error("Model not built. Call build_model() before export_to_onnx()")
            raise ValueError("Model not built. Call build_model() before export_to_onnx()")
        
        try:
            import tf2onnx
            import onnx
        except ImportError:
            logger.error("tf2onnx or onnx package not installed. Install with: pip install tf2onnx onnx")
            raise ImportError("tf2onnx or onnx package not installed. Install with: pip install tf2onnx onnx")
        
        try:
            # Generate ONNX path
            onnx_path = f"{export_path}.onnx"
            
            # Convert Keras model to ONNX
            spec = (tf.TensorSpec((None, self.sequence_length, self.n_features), tf.float32, name="input"),)
            model_proto, _ = tf2onnx.convert.from_keras(self.model, input_signature=spec, opset=13)
            
            # Save ONNX model
            onnx.save_model(model_proto, onnx_path)
            
            logger.info(f"Model exported to ONNX format: {onnx_path}")
            
            return onnx_path
            
        except Exception as e:
            logger.error(f"Failed to export model to ONNX: {e}")
            raise RuntimeError(f"Failed to export model to ONNX: {e}")
    
    def export_to_tensorflow_serving(self, export_dir: str, version: int = 1) -> str:
        """
        Export the model for TensorFlow Serving deployment.
        
        Args:
            export_dir: Directory to save the model
            version: Model version number
            
        Returns:
            Path to the exported model directory
            
        Raises:
            ValueError: If model is not built
            RuntimeError: If export fails
        """
        if self.model is None:
            logger.error("Model not built. Call build_model() before export_to_tensorflow_serving()")
            raise ValueError("Model not built. Call build_model() before export_to_tensorflow_serving()")
        
        try:
            # Create versioned export directory
            versioned_dir = os.path.join(export_dir, str(version))
            os.makedirs(export_dir, exist_ok=True)
            
            # Save model in SavedModel format
            self.model.save(
                versioned_dir,
                include_optimizer=False,
                save_format='tf',
                signatures={
                    'serving_default': self._get_serving_signature()
                }
            )
            
            logger.info(f"Model exported for TensorFlow Serving: {versioned_dir}")
            
            return versioned_dir
            
        except Exception as e:
            logger.error(f"Failed to export model for TensorFlow Serving: {e}")
            raise RuntimeError(f"Failed to export model for TensorFlow Serving: {e}")
    
    def _get_serving_signature(self) -> Callable:
        """
        Create a serving signature for TensorFlow Serving.
        
        Returns:
            Serving signature function
        """
        @tf.function(input_signature=[tf.TensorSpec(shape=(None, self.sequence_length, self.n_features), dtype=tf.float32)])
        def serving_fn(inputs):
            return {'output': self.model(inputs, training=False)}
        
        return serving_fn
    
    def create_tensorboard_projector(
        self,
        X: np.ndarray,
        metadata: Optional[List[str]] = None,
        layer_name: Optional[str] = None,
        projector_dir: Optional[str] = None
    ) -> str:
        """
        Create a TensorBoard Projector for visualizing embeddings.
        
        Args:
            X: Input data
            metadata: Optional list of strings with metadata for each sample
            layer_name: Name of the layer to extract embeddings from
                      If None, the last layer before the output is used
            projector_dir: Directory to save the projector files
                          If None, a directory within model_dir is used
            
        Returns:
            Path to the projector directory
            
        Raises:
            ValueError: If model is not built or layer_name is invalid
            RuntimeError: If projector creation fails
        """
        if self.model is None:
            logger.error("Model not built. Call build_model() before create_tensorboard_projector()")
            raise ValueError("Model not built. Call build_model() before create_tensorboard_projector()")
        
        try:
            from tensorboard.plugins import projector
        except ImportError:
            logger.error("TensorBoard package not installed. Install with: pip install tensorboard")
            raise ImportError("TensorBoard package not installed. Install with: pip install tensorboard")
        
        try:
            # Set projector directory
            if projector_dir is None:
                projector_dir = os.path.join(str(self.model_dir), "projector")
            
            os.makedirs(projector_dir, exist_ok=True)
            
            # Find the target layer
            if layer_name is not None:
                target_layer = self.model.get_layer(layer_name)
            else:
                # Use the last layer before the output
                output_layer_names = ['output', 'price_prediction', 'signal_classification']
                eligible_layers = [l for l in self.model.layers if l.name not in output_layer_names]
                
                if not eligible_layers:
                    logger.error("No eligible layers found for embedding extraction")
                    raise ValueError("No eligible layers found for embedding extraction")
                
                target_layer = eligible_layers[-1]
                layer_name = target_layer.name
            
            # Create a model to extract embeddings
            embedding_model = Model(
                inputs=self.model.input,
                outputs=target_layer.output
            )
            
            # Extract embeddings
            embeddings = embedding_model.predict(X)
            
            # If the embeddings are multi-dimensional, flatten them
            if len(embeddings.shape) > 2:
                embeddings = embeddings.reshape(embeddings.shape[0], -1)
            
            # Save embeddings as a checkpoint
            checkpoint_path = os.path.join(projector_dir, "embeddings.ckpt")
            embedding_var = tf.Variable(embeddings, name='embeddings')
            checkpoint = tf.train.Checkpoint(embedding=embedding_var)
            checkpoint.save(checkpoint_path)
            
            # Set up config
            config = projector.ProjectorConfig()
            embedding = config.embeddings.add()
            embedding.tensor_name = "embedding/.ATTRIBUTES/VARIABLE_VALUE"
            
            # Add metadata if provided
            if metadata is not None:
                metadata_path = os.path.join(projector_dir, "metadata.tsv")
                with open(metadata_path, 'w') as f:
                    for sample_metadata in metadata:
                        f.write(f"{sample_metadata}\n")
                
                embedding.metadata_path = "metadata.tsv"
            
            # Write projector config
            projector.visualize_embeddings(projector_dir, config)
            
            logger.info(f"TensorBoard Projector created at {projector_dir}")
            logger.info(f"To view projector, run: tensorboard --logdir={projector_dir}")
            
            return projector_dir
            
        except Exception as e:
            logger.error(f"Failed to create TensorBoard Projector: {e}")
            raise RuntimeError(f"Failed to create TensorBoard Projector: {e}")
    
    def __str__(self) -> str:
        """
        Return a string representation of the model.
        
        Returns:
            String description of the model
        """
        if self.model is None:
            return "AriadneModel (not built)"
        
        parts = [
            f"AriadneModel (Version: {self.MODEL_VERSION})",
            f"Mode: {self.mode}",
            f"Architecture: {self.architecture}",
            f"Input Shape: ({self.sequence_length}, {self.n_features})",
            f"Output Shape: {self.n_targets if self.mode == 'regression' else self.config.get('n_classes', 2)}",
            f"Trainable Parameters: {self.model.count_params():,}",
            f"Attention Enabled: {self.attention_enabled}",
            f"Uncertainty Estimation: {self.uncertainty_enabled}",
        ]
        
        if self.ensemble_enabled:
            parts.append(f"Ensemble Method: {self.ensemble_method} (Size: {self.ensemble_size})")
        
        if self.last_training_time:
            parts.append(f"Last Training Time: {self.last_training_time}")
        
        return "\n".join(parts)
    
    def __repr__(self) -> str:
        """
        Return a string representation of the model.
        
        Returns:
            String description of the model
        """
        return self.__str__()


if __name__ == "__main__":
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser(description='Ariadne Model CLI')
    parser.add_argument('--config', type=str, help='Path to configuration file')
    parser.add_argument('--mode', type=str, choices=['train', 'predict', 'evaluate', 'export'], help='Operation mode')
    parser.add_argument('--data', type=str, help='Path to data file')
    parser.add_argument('--output', type=str, help='Output path')
    
    args = parser.parse_args()
    
    if args.config and args.mode:
        # Initialize model
        model = AriadneModel(config_path=args.config)
        
        # Perform requested operation
        if args.mode == 'train':
            # Load data and train model (implementation depends on data format)
            print("Training mode not implemented in CLI")
        elif args.mode == 'predict':
            # Load data and generate predictions
            print("Prediction mode not implemented in CLI")
        elif args.mode == 'evaluate':
            # Load data and evaluate model
            print("Evaluation mode not implemented in CLI")
        elif args.mode == 'export':
            # Build model and export
            if args.output:
                model.build_model()
                model.export_to_onnx(args.output)
                print(f"Model exported to {args.output}.onnx")
    else:
        print("Usage: python model.py --config CONFIG_PATH --mode {train,predict,evaluate,export} [--data DATA_PATH] [--output OUTPUT_PATH]")

    price_prediction')(x)
            signal_output = Dense(1, activation='sigmoid', name='signal_classification')(x)
            outputs = [price_output, signal_output]
        
        # Create the model
        model = Model(inputs=inputs, outputs=outputs, name='ariadne_cnn_lstm')
        return model
    
    def _build_transformer_model(self, input_shape: Tuple[int, int]) -> tf.keras.Model:
        """
        Build a Transformer model for time series prediction.
        
        Args:
            input_shape: Tuple of (sequence_length, n_features)
            
        Returns:
            The built Keras model
        """
        # Extract configuration parameters
        sequence_length, n_features = input_shape
        dropout_rate = self.config.get("dropout_rate", 0.1)
        l1_reg = self.config.get("l1_regularization", 0.0)
        l2_reg = self.config.get("l2_regularization", 0.0)
        n_heads = self.config.get("n_heads", 8)
        d_model = self.config.get("d_model", 256)
        dff = self.config.get("dff", 512)
        n_encoder_layers = self.config.get("n_encoder_layers", 4)
        dense_units = self.config.get("dense_units", [128, 64])
        
        # Input layer
        inputs = Input(shape=input_shape, name='input')
        
        # Position encoding
        position_encoding = self._positional_encoding(sequence_length, d_model)
        
        # Input projection
        x = Dense(d_model, name='input_projection')(inputs)
        
        # Add positional encoding
        pos_encoding = tf.cast(position_encoding, dtype=x.dtype)
        x = x + pos_encoding[:, :sequence_length, :]
        
        # Dropout for regularization
        x = Dropout(dropout_rate, name='dropout_pos_encoding')(x)
        
        # Transformer encoder layers
        for i in range(n_encoder_layers):
            x = self._transformer_encoder_layer(
                x, d_model, n_heads, dff, dropout_rate, l1_reg, l2_reg, i+1
            )
        
        # Global average pooling
        x = GlobalAveragePooling1D(name='global_avg_pooling')(x)
        
        # Dense layers
        for i, units in enumerate(dense_units):
            x = Dense(
                units=units,
                kernel_regularizer=l1_l2(l1=l1_reg, l2=l2_reg),
                name=f'dense_{i+1}'
            )(x)
            x = LayerNormalization(name=f'layer_norm_dense_{i+1}')(x)
            x = LeakyReLU(alpha=0.1, name=f'leaky_relu_dense_{i+1}')(x)
            x = Dropout(dropout_rate, name=f'dropout_dense_{i+1}')(x)
        
        # Output layer
        if self.mode == "regression":
            outputs = Dense(self.n_targets, activation='linear', name='output')(x)
        elif self.mode == "classification":
            n_classes = self.config.get("n_classes", 2)
            if n_classes == 2:
                outputs = Dense(1, activation='sigmoid', name='output')(x)
            else:
                outputs = Dense(n_classes, activation='softmax', name='output')(x)
        elif self.mode == "multi_task":
            # Multiple outputs for different tasks
            price_output = Dense(self.n_targets, activation='linear', name='