"""
Androgeus Model - Technical Analysis AI Agent

This module defines the core model architecture for the Androgeus technical analysis
AI agent, which is designed to analyze market data using various technical indicators
and machine learning techniques to predict market movements and generate trading signals.

The model uses a hybrid approach combining LSTM networks for time series analysis with
attention mechanisms to identify relevant patterns in financial data.

Author: Minos-AI Team
Date: January 8, 2025
"""

import os
import json
import logging
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Dense, LSTM, Dropout, Input, Concatenate,
    BatchNormalization, Attention, TimeDistributed
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.regularizers import l1_l2
from typing import Dict, List, Optional, Tuple, Union, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class AndrogeusModel:
    """
    Androgeus Technical Analysis AI Model

    A deep learning model specialized in technical analysis for financial markets,
    combining historical price data, volume information, and various technical
    indicators to predict future price movements and generate trading signals.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the Androgeus model with the specified configuration.
        
        Args:
            config_path: Path to the configuration JSON file. If not provided,
                         the default config.json in the same directory will be used.
        
        Raises:
            FileNotFoundError: If the configuration file cannot be found.
            json.JSONDecodeError: If the configuration file is not valid JSON.
        """
        self.model = None
        self.config = None
        self.is_compiled = False
        self.is_fitted = False
        
        # Load configuration
        if config_path is None:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            config_path = os.path.join(current_dir, 'config.json')
        
        try:
            with open(config_path, 'r') as f:
                self.config = json.load(f)
            logger.info(f"Configuration loaded from {config_path}")
        except FileNotFoundError:
            logger.error(f"Configuration file not found at {config_path}")
            raise
        except json.JSONDecodeError:
            logger.error(f"Invalid JSON in configuration file at {config_path}")
            raise
            
        # Set up TensorFlow memory growth to avoid GPU memory issues
        self._configure_gpu_memory()
        
        # Initialize model architecture
        self._build_model()
        
    def _configure_gpu_memory(self) -> None:
        """
        Configure TensorFlow to grow GPU memory usage as needed.
        This helps prevent TensorFlow from allocating all GPU memory at once.
        """
        try:
            gpus = tf.config.experimental.list_physical_devices('GPU')
            if gpus:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                logger.info(f"Found {len(gpus)} GPU(s). Memory growth enabled.")
            else:
                logger.info("No GPUs found. Running on CPU.")
        except Exception as e:
            logger.warning(f"Error configuring GPU memory: {str(e)}")
    
    def _build_model(self) -> None:
        """
        Build the neural network architecture based on the configuration.
        
        The model architecture consists of:
        1. Price sequence input branch with LSTM layers
        2. Technical indicators input branch with dense layers
        3. Attention mechanism to focus on important time steps
        4. Merged layers for final prediction outputs
        
        Raises:
            ValueError: If required configuration parameters are missing or invalid.
        """
        try:
            # Extract model parameters from config
            sequence_length = self.config.get('sequence_length', 50)
            price_features = self.config.get('price_features', 5)  # OHLCV
            indicator_features = self.config.get('indicator_features', 20)
            lstm_units = self.config.get('lstm_units', [128, 64])
            dense_units = self.config.get('dense_units', [128, 64, 32])
            dropout_rate = self.config.get('dropout_rate', 0.2)
            learning_rate = self.config.get('learning_rate', 0.001)
            l1_reg = self.config.get('l1_reg', 0.0001)
            l2_reg = self.config.get('l2_reg', 0.0001)
            
            # Price sequence input - shape: (sequence_length, price_features)
            price_input = Input(shape=(sequence_length, price_features), name='price_input')
            
            # Technical indicators input - shape: (indicator_features,)
            indicator_input = Input(shape=(indicator_features,), name='indicator_input')
            
            # Market metadata input (optional) - e.g., market cap, exchange info, etc.
            metadata_input = None
            if self.config.get('use_metadata', False):
                metadata_features = self.config.get('metadata_features', 10)
                metadata_input = Input(shape=(metadata_features,), name='metadata_input')
            
            # Process price sequence through LSTM layers
            x_price = price_input
            for i, units in enumerate(lstm_units):
                return_sequences = (i < len(lstm_units) - 1) or self.config.get('use_attention', True)
                x_price = LSTM(
                    units, 
                    return_sequences=return_sequences,
                    kernel_regularizer=l1_l2(l1=l1_reg, l2=l2_reg),
                    recurrent_regularizer=l1_l2(l1=l1_reg, l2=l2_reg),
                    name=f'lstm_{i}'
                )(x_price)
                x_price = BatchNormalization(name=f'bn_lstm_{i}')(x_price)
                x_price = Dropout(dropout_rate, name=f'dropout_lstm_{i}')(x_price)
            
            # Apply attention mechanism if enabled
            if self.config.get('use_attention', True):
                # Self-attention on LSTM output sequence
                query_value_attention = Attention(name='self_attention')([x_price, x_price])
                x_price = query_value_attention
                
                # Convert sequences to single vector if needed
                if x_price.shape.ndims > 2:
                    # Use global average pooling on the sequence dimension
                    x_price = tf.reduce_mean(x_price, axis=1)
            
            # Process technical indicators through dense layers
            x_indicator = indicator_input
            for i, units in enumerate(dense_units[:2]):  # Use first few layers for indicators
                x_indicator = Dense(
                    units, 
                    activation='relu',
                    kernel_regularizer=l1_l2(l1=l1_reg, l2=l2_reg),
                    name=f'dense_indicator_{i}'
                )(x_indicator)
                x_indicator = BatchNormalization(name=f'bn_indicator_{i}')(x_indicator)
                x_indicator = Dropout(dropout_rate, name=f'dropout_indicator_{i}')(x_indicator)
            
            # Merge price and indicator branches
            merged_inputs = [x_price, x_indicator]
            
            # Add metadata input if enabled
            if metadata_input is not None:
                x_metadata = Dense(
                    dense_units[0]//2, 
                    activation='relu',
                    kernel_regularizer=l1_l2(l1=l1_reg, l2=l2_reg),
                    name='dense_metadata'
                )(metadata_input)
                x_metadata = BatchNormalization(name='bn_metadata')(x_metadata)
                x_metadata = Dropout(dropout_rate, name='dropout_metadata')(x_metadata)
                merged_inputs.append(x_metadata)
            
            # Concatenate all input branches
            if len(merged_inputs) > 1:
                x = Concatenate(name='concatenate')(merged_inputs)
            else:
                x = merged_inputs[0]
            
            # Final dense layers
            for i, units in enumerate(dense_units[1:], 1):  # Skip first layer already used
                x = Dense(
                    units, 
                    activation='relu',
                    kernel_regularizer=l1_l2(l1=l1_reg, l2=l2_reg),
                    name=f'dense_merged_{i}'
                )(x)
                x = BatchNormalization(name=f'bn_merged_{i}')(x)
                x = Dropout(dropout_rate, name=f'dropout_merged_{i}')(x)
            
            # Output layers based on configuration
            outputs = []
            
            # Price direction prediction (classification: up/down)
            if self.config.get('predict_direction', True):
                direction_output = Dense(
                    1, activation='sigmoid', name='direction_output'
                )(x)
                outputs.append(direction_output)
            
            # Price movement magnitude prediction (regression)
            if self.config.get('predict_magnitude', True):
                magnitude_output = Dense(
                    1, activation='linear', name='magnitude_output'
                )(x)
                outputs.append(magnitude_output)
                
            # Price target points prediction (regression)
            if self.config.get('predict_price_targets', False):
                num_targets = self.config.get('num_price_targets', 3)
                targets_output = Dense(
                    num_targets, activation='linear', name='targets_output'
                )(x)
                outputs.append(targets_output)
                
            # Volatility prediction (regression)
            if self.config.get('predict_volatility', False):
                volatility_output = Dense(
                    1, activation='softplus', name='volatility_output'
                )(x)
                outputs.append(volatility_output)
                
            # Create and compile model
            model_inputs = [price_input, indicator_input]
            if metadata_input is not None:
                model_inputs.append(metadata_input)
                
            self.model = Model(inputs=model_inputs, outputs=outputs, name="androgeus_model")
            
            logger.info(f"Model built with architecture: {len(lstm_units)} LSTM layers, "
                       f"{len(dense_units)} dense layers, {len(outputs)} outputs")
                       
        except Exception as e:
            logger.error(f"Error building model: {str(e)}")
            raise
    
    def compile(self, learning_rate: Optional[float] = None) -> None:
        """
        Compile the model with appropriate loss functions and metrics.
        
        Args:
            learning_rate: Optional learning rate override. If None, uses the
                          value from the configuration.
        
        Raises:
            ValueError: If the model has not been built yet.
        """
        if self.model is None:
            logger.error("Cannot compile model: Model has not been built")
            raise ValueError("Model has not been built. Call build_model first.")
        
        if learning_rate is None:
            learning_rate = self.config.get('learning_rate', 0.001)
        
        # Define losses and metrics for each output
        losses = {}
        metrics = {}
        loss_weights = {}
        
        # Configure outputs based on enabled predictions
        if self.config.get('predict_direction', True):
            losses['direction_output'] = 'binary_crossentropy'
            metrics['direction_output'] = ['accuracy', tf.keras.metrics.AUC()]
            loss_weights['direction_output'] = self.config.get('direction_weight', 1.0)
            
        if self.config.get('predict_magnitude', True):
            losses['magnitude_output'] = 'mean_squared_error'
            metrics['magnitude_output'] = ['mean_absolute_error']
            loss_weights['magnitude_output'] = self.config.get('magnitude_weight', 1.0)
            
        if self.config.get('predict_price_targets', False):
            losses['targets_output'] = 'mean_squared_error'
            metrics['targets_output'] = ['mean_absolute_error']
            loss_weights['targets_output'] = self.config.get('targets_weight', 1.0)
            
        if self.config.get('predict_volatility', False):
            # Custom loss for volatility prediction
            losses['volatility_output'] = 'mean_squared_error'
            metrics['volatility_output'] = ['mean_absolute_error']
            loss_weights['volatility_output'] = self.config.get('volatility_weight', 0.5)
        
        # Compile the model
        self.model.compile(
            optimizer=Adam(learning_rate=learning_rate),
            loss=losses,
            metrics=metrics,
            loss_weights=loss_weights
        )
        
        self.is_compiled = True
        logger.info(f"Model compiled with learning rate {learning_rate}")
        logger.info(f"Loss functions: {losses}")
    
    def fit(self, 
            train_data: Tuple[List[np.ndarray], List[np.ndarray]],
            validation_data: Optional[Tuple[List[np.ndarray], List[np.ndarray]]] = None,
            epochs: Optional[int] = None,
            batch_size: Optional[int] = None,
            callbacks: Optional[List[tf.keras.callbacks.Callback]] = None) -> tf.keras.callbacks.History:
        """
        Train the model on the provided data.
        
        Args:
            train_data: Tuple of (inputs, targets) for training. 
                       Inputs should be a list of numpy arrays matching the model inputs.
                       Targets should be a list of numpy arrays matching the model outputs.
            validation_data: Optional tuple of (inputs, targets) for validation.
            epochs: Number of epochs to train. If None, uses the value from configuration.
            batch_size: Batch size for training. If None, uses the value from configuration.
            callbacks: Optional list of Keras callbacks for training.
            
        Returns:
            A Keras History object containing the training history.
            
        Raises:
            ValueError: If the model has not been compiled yet.
        """
        if not self.is_compiled:
            logger.error("Cannot train model: Model has not been compiled")
            raise ValueError("Model has not been compiled. Call compile first.")
        
        # Use configuration values if not provided
        if epochs is None:
            epochs = self.config.get('epochs', 100)
            
        if batch_size is None:
            batch_size = self.config.get('batch_size', 64)
        
        # Default callbacks if none provided
        if callbacks is None:
            callbacks = self._get_default_callbacks()
        
        logger.info(f"Starting model training for {epochs} epochs with batch size {batch_size}")
        
        try:
            history = self.model.fit(
                train_data[0],
                train_data[1],
                epochs=epochs,
                batch_size=batch_size,
                validation_data=validation_data,
                callbacks=callbacks,
                verbose=1
            )
            
            self.is_fitted = True
            logger.info("Model training completed successfully")
            
            return history
            
        except Exception as e:
            logger.error(f"Error during model training: {str(e)}")
            raise
    
    def _get_default_callbacks(self) -> List[tf.keras.callbacks.Callback]:
        """
        Create default callbacks for model training.
        
        Returns:
            A list of Keras callbacks including early stopping and model checkpointing.
        """
        current_dir = os.path.dirname(os.path.abspath(__file__))
        checkpoints_dir = os.path.join(current_dir, 'checkpoints')
        
        # Create checkpoints directory if it doesn't exist
        os.makedirs(checkpoints_dir, exist_ok=True)
        
        callbacks = [
            # Early stopping to prevent overfitting
            EarlyStopping(
                monitor='val_loss' if self.config.get('use_validation', True) else 'loss',
                patience=self.config.get('early_stopping_patience', 10),
                restore_best_weights=True,
                verbose=1
            ),
            
            # Model checkpointing to save best models
            ModelCheckpoint(
                filepath=os.path.join(checkpoints_dir, 'androgeus_model_{epoch:02d}_{val_loss:.4f}.h5'),
                monitor='val_loss' if self.config.get('use_validation', True) else 'loss',
                save_best_only=True,
                save_weights_only=False,
                verbose=1
            ),
            
            # Learning rate scheduler (optional)
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss' if self.config.get('use_validation', True) else 'loss',
                factor=0.5,
                patience=5,
                min_lr=1e-6,
                verbose=1
            ) if self.config.get('use_lr_scheduler', True) else None
        ]
        
        # Remove None values
        return [cb for cb in callbacks if cb is not None]
    
    def predict(self, inputs: List[np.ndarray]) -> List[np.ndarray]:
        """
        Generate predictions using the trained model.
        
        Args:
            inputs: List of numpy arrays matching the model inputs.
            
        Returns:
            List of numpy arrays containing the model's predictions.
            
        Raises:
            ValueError: If the model has not been fitted yet.
        """
        if not self.is_fitted:
            logger.warning("Model has not been fitted. Predictions may be unreliable.")
        
        try:
            predictions = self.model.predict(inputs)
            
            # Convert single output to list for consistent handling
            if not isinstance(predictions, list):
                predictions = [predictions]
                
            return predictions
            
        except Exception as e:
            logger.error(f"Error during prediction: {str(e)}")
            raise
    
    def evaluate(self, 
                 inputs: List[np.ndarray], 
                 targets: List[np.ndarray]) -> Dict[str, float]:
        """
        Evaluate the model on test data.
        
        Args:
            inputs: List of numpy arrays matching the model inputs.
            targets: List of numpy arrays matching the model outputs.
            
        Returns:
            Dictionary of metric names and values.
            
        Raises:
            ValueError: If the model has not been fitted yet.
        """
        if not self.is_fitted:
            logger.error("Cannot evaluate model: Model has not been fitted")
            raise ValueError("Model has not been fitted. Call fit first.")
        
        try:
            results = self.model.evaluate(inputs, targets, verbose=1)
            
            # Create dictionary of metrics based on model.metrics_names
            metrics_dict = {}
            for i, metric_name in enumerate(self.model.metrics_names):
                metrics_dict[metric_name] = results[i]
                
            logger.info(f"Model evaluation results: {metrics_dict}")
            return metrics_dict
            
        except Exception as e:
            logger.error(f"Error during model evaluation: {str(e)}")
            raise
    
    def save(self, path: Optional[str] = None) -> str:
        """
        Save the model to disk.
        
        Args:
            path: Path where to save the model. If None, uses default location
                 based on the model name.
                 
        Returns:
            Path where the model was saved.
            
        Raises:
            ValueError: If the model has not been built yet.
        """
        if self.model is None:
            logger.error("Cannot save model: Model has not been built")
            raise ValueError("Model has not been built.")
        
        if path is None:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            path = os.path.join(current_dir, 'saved_models', 'androgeus_model.h5')
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        try:
            self.model.save(path)
            logger.info(f"Model saved to {path}")
            
            # Also save the configuration
            config_path = os.path.join(os.path.dirname(path), 'androgeus_config.json')
            with open(config_path, 'w') as f:
                json.dump(self.config, f, indent=2)
            logger.info(f"Model configuration saved to {config_path}")
            
            return path
            
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
            raise
    
    @classmethod
    def load(cls, model_path: str, config_path: Optional[str] = None) -> 'AndrogeusModel':
        """
        Load a saved model from disk.
        
        Args:
            model_path: Path to the saved model file.
            config_path: Optional path to the configuration file. If None,
                         attempts to find a config file in the same directory.
                         
        Returns:
            An instance of AndrogeusModel with the loaded model.
            
        Raises:
            FileNotFoundError: If the model file cannot be found.
        """
        # Try to find config in the same directory if not provided
        if config_path is None:
            possible_config = os.path.join(
                os.path.dirname(model_path),
                'androgeus_config.json'
            )
            if os.path.exists(possible_config):
                config_path = possible_config
        
        # Create instance with config
        instance = cls(config_path=config_path)
        
        try:
            # Load the model
            instance.model = tf.keras.models.load_model(model_path)
            instance.is_compiled = True
            instance.is_fitted = True
            logger.info(f"Model loaded from {model_path}")
            return instance
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise
    
    def summary(self) -> None:
        """
        Print a summary of the model architecture.
        
        Raises:
            ValueError: If the model has not been built yet.
        """
        if self.model is None:
            logger.error("Cannot show summary: Model has not been built")
            raise ValueError("Model has not been built.")
        
        self.model.summary()
        
    def get_feature_importance(self, 
                               inputs: List[np.ndarray], 
                               targets: List[np.ndarray],
                               output_idx: int = 0,
                               method: str = 'permutation') -> Dict[str, float]:
        """
        Calculate feature importance using either permutation importance
        or integrated gradients.
        
        Args:
            inputs: List of numpy arrays matching the model inputs.
            targets: List of numpy arrays matching the model outputs.
            output_idx: Index of the output to calculate importance for.
            method: Method to use for feature importance, either 'permutation'
                    or 'gradients'.
                    
        Returns:
            Dictionary mapping feature names to importance scores.
            
        Raises:
            ValueError: If the model has not been fitted yet or if an invalid
                       method is specified.
        """
        if not self.is_fitted:
            logger.error("Cannot calculate feature importance: Model has not been fitted")
            raise ValueError("Model has not been fitted. Call fit first.")
        
        if method not in ['permutation', 'gradients']:
            logger.error(f"Invalid feature importance method: {method}")
            raise ValueError("Method must be either 'permutation' or 'gradients'")
        
        feature_names = {
            'price_input': [],
            'indicator_input': []
        }
        
        # Get feature names from config if available
        if 'price_feature_names' in self.config:
            feature_names['price_input'] = self.config['price_feature_names']
        else:
            # Default OHLCV names
            feature_names['price_input'] = ['open', 'high', 'low', 'close', 'volume']
            
        if 'indicator_feature_names' in self.config:
            feature_names['indicator_input'] = self.config['indicator_feature_names']
        else:
            # Generate generic names for indicators
            num_indicators = self.config.get('indicator_features', 20)
            feature_names['indicator_input'] = [f'indicator_{i}' for i in range(num_indicators)]
            
        if method == 'permutation':
            return self._permutation_importance(inputs, targets, output_idx, feature_names)
        else:
            return self._gradient_importance(inputs, output_idx, feature_names)
    
    def _permutation_importance(self, 
                                inputs: List[np.ndarray], 
                                targets: List[np.ndarray],
                                output_idx: int,
                                feature_names: Dict[str, List[str]]) -> Dict[str, float]:
        """
        Calculate permutation feature importance.
        
        Args:
            inputs: List of numpy arrays matching the model inputs.
            targets: List of numpy arrays matching the model outputs.
            output_idx: Index of the output to calculate importance for.
            feature_names: Dictionary mapping input names to feature names.
            
        Returns:
            Dictionary mapping feature names to importance scores.
        """
        # Implementation of permutation importance
        logger.info("Calculating permutation feature importance")
        
        # Make a copy of inputs to avoid modifying the original
        inputs_copy = [np.copy(inp) for inp in inputs]
        
        # Get baseline performance
        baseline_metrics = self.evaluate(inputs_copy, targets)
        baseline_loss = baseline_metrics['loss']
        
        importance_scores = {}
        
        # For each input and each feature
        for input_idx, input_data in enumerate(inputs_copy):
            input_name = self.model.input_names[input_idx]
            
            # Skip inputs not in feature_names
            if input_name not in feature_names:
                continue
                
            # For price input, we need to handle the time dimension
            if input_name == 'price_input':
                # For each price feature (across all time steps)
                for feat_idx in range(input_data.shape[2]):
                    if feat_idx >= len(feature_names[input_name]):
                        feat_name = f"{input_name}_unknown_{feat_idx}"
                    else:
                        feat_name = f"{input_name}_{feature_names[input_name][feat_idx]}"
                    
                    # Save original values
                    original_values = np.copy(input_data[:, :, feat_idx])
                    
                    # Permute this feature across all time steps
                    np.random.shuffle(input_data[:, :, feat_idx])
                    
                    # Evaluate with permuted feature
                    perm_metrics = self.evaluate(inputs_copy, targets)
                    perm_loss = perm_metrics['loss']
                    
                    # Importance is the increase in loss
                    importance = perm_loss - baseline_loss
                    importance_scores[feat_name] = float(importance)
                    
                    # Restore original values
                    input_data[:, :, feat_idx] = original_values
            else:
                # For other inputs (without time dimension)
                for feat_idx in range(input_data.shape[1]):
                    if feat_idx >= len(feature_names[input_name]):
                        feat_name = f"{input_name}_unknown_{feat_idx}"
                    else:
                        feat_name = f"{input_name}_{feature_names[input_name][feat_idx]}"
                    
                    # Save original values
                    original_values = np.copy(input_data[:, feat_idx])
                    
                    # Permute this feature
                    np.random.shuffle(input_data[:, feat_idx])
                    
                    # Evaluate with permuted feature
                    perm_metrics = self.evaluate(inputs_copy, targets)
                    perm_loss = perm_metrics['loss']
                    
                    # Importance is the increase in loss
                    importance = perm_loss - baseline_loss
                    importance_scores[feat_name] = float(importance)
                    
                    # Restore original values
                    input_data[:, feat_idx] = original_values
        
        # Normalize importance scores
        max_importance = max(abs(score) for score in importance_scores.values())
        if max_importance > 0:
            for feat_name in importance_scores:
                importance_scores[feat_name] /= max_importance
                
        return importance_scores
    
    def _gradient_importance(self, 
                             inputs: List[np.ndarray], 
                             output_idx: int,
                             feature_names: Dict[str, List[str]]) -> Dict[str, float]:
        """
        Calculate gradient-based feature importance.
        
        Args:
            inputs: List of numpy arrays matching the model inputs.
            output_idx: Index of the output to calculate importance for.
            feature_names: Dictionary mapping input names to feature names.
            
        Returns:
            Dictionary mapping feature names to importance scores.
        """
        logger.info("Calculating gradient-based feature importance")
        
        # Define a GradientTape function to compute gradients
        @tf.function
        def get_gradients(input_tensors):
            with tf.GradientTape() as tape:
                for tensor in input_tensors:
                    tape.watch(tensor)
                predictions = self.model(input_tensors)
                output = predictions[output_idx] if isinstance(predictions, list) else predictions
                # Use mean of the output for gradient calculation
                target = tf.reduce_mean(output)
            
            # Get gradients with respect to inputs
            gradients = tape.gradient(target, input_tensors)
            return gradients
        
        # Convert inputs to tensors
        input_tensors = [tf.convert_to_tensor(inp) for inp in inputs]
        
        # Calculate gradients
        gradients = get_gradients(input_tensors)
        
        importance_scores = {}
        
        # Process gradients for each input
        for input_idx, (grad, input_data) in enumerate(zip(gradients, inputs)):
            input_name = self.model.input_names[input_idx]
            
            # Skip inputs not in feature_names
            if input_name not in feature_names:
                continue
                
            # For price input with time dimension
            if input_name == 'price_input':
                # Get absolute gradients and average across batch and time
                abs_grad = tf.abs(grad)
                avg_grad = tf.reduce_mean(abs_grad, axis=[0, 1])  # Average across batch and time
                
                # For each price feature
                for feat_idx in range(avg_grad.shape[0]):
                    if feat_idx >= len(feature_names[input_name]):
                        feat_name = f"{input_name}_unknown_{feat_idx}"
                    else:
                        feat_name = f"{input_name}_{feature_names[input_name][feat_idx]}"
                    
                    importance_scores[feat_name] = float(avg_grad[feat_idx])
            else:
                # For other inputs (without time dimension)
                abs_grad = tf.abs(grad)
                avg_grad = tf.reduce_mean(abs_grad, axis=0)  # Average across batch
                
                # For each feature
                for feat_idx in range(avg_grad.shape[0]):
                    if feat_idx >= len(feature_names[input_name]):
                        feat_name = f"{input_name}_unknown_{feat_idx}"
                    else:
                        feat_name = f"{input_name}_{feature_names[input_name][feat_idx]}"
                    
                    importance_scores[feat_name] = float(avg_grad[feat_idx])
        
        # Normalize importance scores
        max_importance = max(abs(score) for score in importance_scores.values())
        if max_importance > 0:
            for feat_name in importance_scores:
                importance_scores[feat_name] /= max_importance
                
        return importance_scores
    
    def visualize_feature_importance(self, importance_scores: Dict[str, float], top_n: int = 10) -> None:
        """
        Visualize feature importance scores.
        
        Args:
            importance_scores: Dictionary mapping feature names to importance scores.
            top_n: Number of top features to visualize.
            
        Raises:
            ImportError: If matplotlib is not installed.
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            logger.error("Matplotlib is required for visualization")
            raise ImportError("Matplotlib is required for visualization. Install it using 'pip install matplotlib'")
        
        # Sort features by importance
        sorted_features = sorted(
            importance_scores.items(), 
            key=lambda x: abs(x[1]), 
            reverse=True
        )[:top_n]
        
        # Unpack names and scores
        names = [item[0] for item in sorted_features]
        scores = [item[1] for item in sorted_features]
        
        # Create horizontal bar plot
        plt.figure(figsize=(10, 8))
        colors = ['#3498db' if score > 0 else '#e74c3c' for score in scores]
        bars = plt.barh(range(len(names)), [abs(score) for score in scores], color=colors)
        
        # Customize plot
        plt.yticks(range(len(names)), names)
        plt.xlabel('Normalized Importance')
        plt.title('Feature Importance')
        plt.tight_layout()
        
        # Add a legend for positive and negative importance
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='#3498db', label='Positive Impact'),
            Patch(facecolor='#e74c3c', label='Negative Impact')
        ]
        plt.legend(handles=legend_elements, loc='lower right')
        
        # Show plot
        plt.show()
    
    def debug_model(self, inputs: List[np.ndarray]) -> None:
        """
        Debug the model by examining activations of intermediate layers.
        
        Args:
            inputs: List of numpy arrays matching the model inputs.
            
        Raises:
            ValueError: If the model has not been built yet.
        """
        if self.model is None:
            logger.error("Cannot debug model: Model has not been built")
            raise ValueError("Model has not been built.")
        
        try:
            # Create a list of all layer names
            layer_names = [layer.name for layer in self.model.layers]
            logger.info(f"Model contains {len(layer_names)} layers")
            
            # Create models that output intermediate activations
            activation_models = {}
            for layer_name in layer_names:
                layer = self.model.get_layer(layer_name)
                activation_models[layer_name] = Model(
                    inputs=self.model.inputs,
                    outputs=layer.output
                )
            
            # Get activations for each layer
            activations = {}
            for layer_name, activation_model in activation_models.items():
                try:
                    activation = activation_model.predict(inputs)
                    
                    # Calculate activation statistics
                    if isinstance(activation, np.ndarray):
                        stats = {
                            'shape': activation.shape,
                            'min': float(np.min(activation)),
                            'max': float(np.max(activation)),
                            'mean': float(np.mean(activation)),
                            'std': float(np.std(activation)),
                            'sparsity': float(np.mean(activation == 0))
                        }
                        activations[layer_name] = stats
                        logger.info(f"Layer {layer_name}: {stats}")
                    else:
                        logger.warning(f"Layer {layer_name} output is not a numpy array")
                        
                except Exception as e:
                    logger.warning(f"Could not get activations for layer {layer_name}: {str(e)}")
            
            return activations
            
        except Exception as e:
            logger.error(f"Error during model debugging: {str(e)}")
            raise
    
    def export_for_production(self, export_dir: str, optimize: bool = True) -> str:
        """
        Export the model for production use, optionally optimizing it.
        
        Args:
            export_dir: Directory where to export the model.
            optimize: Whether to optimize the model for inference.
            
        Returns:
            Path to the exported model.
            
        Raises:
            ValueError: If the model has not been built yet.
        """
        if self.model is None:
            logger.error("Cannot export model: Model has not been built")
            raise ValueError("Model has not been built.")
        
        # Create export directory if it doesn't exist
        os.makedirs(export_dir, exist_ok=True)
        
        try:
            # Save model architecture and weights in the TensorFlow SavedModel format
            model_path = os.path.join(export_dir, 'androgeus_model')
            
            # Clone the model to avoid modifying the original
            export_model = tf.keras.models.clone_model(self.model)
            export_model.set_weights(self.model.get_weights())
            
            # Optimize for inference if requested
            if optimize:
                try:
                    # Convert to TensorFlow Lite for mobile/edge deployment
                    converter = tf.lite.TFLiteConverter.from_keras_model(export_model)
                    converter.optimizations = [tf.lite.Optimize.DEFAULT]
                    tflite_model = converter.convert()
                    
                    # Save TFLite model
                    tflite_path = os.path.join(export_dir, 'androgeus_model.tflite')
                    with open(tflite_path, 'wb') as f:
                        f.write(tflite_model)
                    logger.info(f"Optimized TFLite model saved to {tflite_path}")
                    
                    # For TensorFlow SavedModel with optimizations for CPU/GPU
                    export_model = tf.keras.models.clone_model(self.model)
                    export_model.set_weights(self.model.get_weights())
                    
                    # Apply optimization
                    logger.info("Applying TensorFlow optimizations")
                    # Use mixed precision
                    if tf.config.list_physical_devices('GPU'):
                        tf.keras.mixed_precision.set_global_policy('mixed_float16')
                        
                except Exception as e:
                    logger.warning(f"Model optimization failed: {str(e)}")
                    logger.warning("Proceeding with standard export")
            
            # Save the model in SavedModel format
            export_model.save(model_path, save_format='tf')
            logger.info(f"Model exported to {model_path}")
            
            # Export config alongside the model
            config_path = os.path.join(export_dir, 'androgeus_config.json')
            with open(config_path, 'w') as f:
                json.dump(self.config, f, indent=2)
            logger.info(f"Model configuration exported to {config_path}")
            
            # Create a metadata file with model information
            metadata = {
                'model_name': 'androgeus',
                'version': self.config.get('model_version', '1.0.0'),
                'description': 'Technical Analysis AI Agent',
                'features': {
                    'price_features': self.config.get('price_features', 5),
                    'indicator_features': self.config.get('indicator_features', 20),
                    'sequence_length': self.config.get('sequence_length', 50)
                },
                'outputs': [],
                'export_time': str(tf.timestamp())
            }
            
            # Add output information
            if self.config.get('predict_direction', True):
                metadata['outputs'].append({
                    'name': 'direction_output',
                    'type': 'classification',
                    'description': 'Price movement direction (up/down)'
                })
                
            if self.config.get('predict_magnitude', True):
                metadata['outputs'].append({
                    'name': 'magnitude_output',
                    'type': 'regression',
                    'description': 'Price movement magnitude'
                })
                
            if self.config.get('predict_price_targets', False):
                metadata['outputs'].append({
                    'name': 'targets_output',
                    'type': 'regression',
                    'description': 'Price targets'
                })
                
            if self.config.get('predict_volatility', False):
                metadata['outputs'].append({
                    'name': 'volatility_output',
                    'type': 'regression',
                    'description': 'Price volatility'
                })
            
            # Save metadata
            metadata_path = os.path.join(export_dir, 'androgeus_metadata.json')
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            logger.info(f"Model metadata exported to {metadata_path}")
            
            return model_path
            
        except Exception as e:
            logger.error(f"Error exporting model: {str(e)}")
            raise
            
    def __str__(self) -> str:
        """
        Return a string representation of the model.
        
        Returns:
            A string containing model information.
        """
        if self.model is None:
            return "AndrogeusModel (not built)"
        
        # Collect model information
        info = [
            "AndrogeusModel:",
            f"  Compiled: {self.is_compiled}",
            f"  Fitted: {self.is_fitted}",
            f"  Architecture: {len(self.config.get('lstm_units', [128, 64]))} LSTM layers, "
            f"{len(self.config.get('dense_units', [128, 64, 32]))} dense layers"
        ]
        
        # Add input information
        info.append("  Inputs:")
        info.append(f"    Price sequence: {self.config.get('sequence_length', 50)} timesteps, "
                    f"{self.config.get('price_features', 5)} features")
        info.append(f"    Technical indicators: {self.config.get('indicator_features', 20)} features")
        if self.config.get('use_metadata', False):
            info.append(f"    Market metadata: {self.config.get('metadata_features', 10)} features")
        
        # Add output information
        info.append("  Outputs:")
        if self.config.get('predict_direction', True):
            info.append("    Direction prediction (classification)")
        if self.config.get('predict_magnitude', True):
            info.append("    Magnitude prediction (regression)")
        if self.config.get('predict_price_targets', False):
            info.append(f"    Price targets: {self.config.get('num_price_targets', 3)} targets (regression)")
        if self.config.get('predict_volatility', False):
            info.append("    Volatility prediction (regression)")
        
        return "\n".join(info)


if __name__ == "__main__":
    """
    Simple example of using the AndrogeusModel.
    """
    import numpy as np
    
    # Create a sample configuration
    sample_config = {
        "sequence_length": 30,
        "price_features": 5,       # OHLCV
        "indicator_features": 15,  # Technical indicators
        "lstm_units": [64, 32],
        "dense_units": [64, 32, 16],
        "dropout_rate": 0.3,
        "learning_rate": 0.001,
        "batch_size": 32,
        "epochs": 50,
        "predict_direction": True,
        "predict_magnitude": True,
        "predict_price_targets": False,
        "predict_volatility": False,
        "use_attention": True,
        "early_stopping_patience": 5
    }
    
    # Save temporary config for testing
    import tempfile
    with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
        json.dump(sample_config, f)
        config_path = f.name
    
    try:
        # Create model
        print("Creating model...")
        model = AndrogeusModel(config_path)
        print(model)
        
        # Compile model
        print("\nCompiling model...")
        model.compile()
        
        # Generate some random data for demonstration
        print("\nGenerating sample data...")
        batch_size = 32
        
        # Create price sequence data: (batch_size, sequence_length, price_features)
        price_data = np.random.random((batch_size, sample_config["sequence_length"], sample_config["price_features"]))
        
        # Create indicator data: (batch_size, indicator_features)
        indicator_data = np.random.random((batch_size, sample_config["indicator_features"]))
        
        # Create sample targets
        direction_targets = np.random.randint(0, 2, (batch_size, 1)).astype(np.float32)
        magnitude_targets = np.random.random((batch_size, 1)) * 0.1
        
        # Example prediction
        print("\nMaking a prediction with random data...")
        model.is_fitted = True  # Hack for demonstration only
        predictions = model.predict([price_data, indicator_data])
        
        print(f"Direction prediction shape: {predictions[0].shape}")
        print(f"Magnitude prediction shape: {predictions[1].shape}")
        
        print("\nModel summary:")
        model.summary()
        
    finally:
        # Clean up temporary file
        os.unlink(config_path)