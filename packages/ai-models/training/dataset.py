"""
Dataset Preparation Module for Minos-AI DeFi Strategy Platform

This module provides utilities for preparing datasets for machine learning models,
with a focus on time series data processing for financial applications. It includes:
- Time series dataset creation with sliding windows
- Feature engineering and transformation
- Data normalization and scaling
- Train/val/test splitting with time-aware logic
- Dataset augmentation techniques
- PyTorch and TensorFlow dataset abstractions

Integration Points:
- Used by train.py for dataset preparation
- Consumes data from data_loader.py
- Provides datasets to models in standardized format
- Handles preprocessing consistently between training and inference

Author: Minos-AI Team
"""

import os
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Union, Optional, Any, Callable
from datetime import datetime, timedelta
import logging
from enum import Enum
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
import torch
from torch.utils.data import Dataset, DataLoader
import tensorflow as tf

# Configure logging
logger = logging.getLogger(__name__)


class ScalingMethod(Enum):
    """Enumeration of supported scaling methods."""
    STANDARD = "standard"
    MINMAX = "minmax"
    ROBUST = "robust"
    NONE = "none"
    PERCENTAGE_CHANGE = "percentage_change"
    LOG_TRANSFORM = "log_transform"


class FeatureEngineering:
    """
    Feature engineering utility for time series financial data.
    
    This class provides methods for creating features from raw financial data,
    including technical indicators, statistical features, and temporal features.
    """
    
    def __init__(self, 
                feature_columns: List[str],
                target_column: str,
                config: Dict[str, Any] = None):
        """
        Initialize feature engineering processor.
        
        Args:
            feature_columns: List of column names to use as features
            target_column: Name of target column
            config: Configuration options for feature engineering
        """
        self.feature_columns = feature_columns
        self.target_column = target_column
        self.config = config or {}
        
        # Technical indicators to compute (if not already in data)
        self.technical_indicators = self.config.get("technical_indicators", [])
        
        # Temporal features to add
        self.temporal_features = self.config.get("temporal_features", [])
        
        # Statistical features to compute
        self.statistical_features = self.config.get("statistical_features", [])
        
        # Lag features
        self.lag_features = self.config.get("lag_features", {})
        
        # Difference features
        self.diff_features = self.config.get("diff_features", {})
        
        # Custom features (calculated using lambda functions)
        self.custom_features = self.config.get("custom_features", {})
        
        # Features to drop after engineering
        self.drop_features = self.config.get("drop_features", [])
        
        logger.info(f"Initialized feature engineering with {len(feature_columns)} base features")
    
    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Apply feature engineering to input data.
        
        Args:
            data: Input DataFrame
            
        Returns:
            DataFrame with engineered features
        """
        # Create a copy to avoid modifying the original
        df = data.copy()
        
        # Ensure timestamp column is datetime
        if 'timestamp' in df.columns and not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Only keep necessary base columns and timestamp (if available)
        keep_columns = list(set(self.feature_columns + [self.target_column] + 
                               (['timestamp'] if 'timestamp' in df.columns else [])))
        
        # Add any extra columns required for feature calculation
        if self.technical_indicators:
            for indicator in self.technical_indicators:
                if indicator.startswith('rsi_') and 'close' not in keep_columns and 'close' in df.columns:
                    keep_columns.append('close')
                elif indicator.startswith('macd_') and 'close' not in keep_columns and 'close' in df.columns:
                    keep_columns.append('close')
                elif indicator.startswith('bb_') and 'close' not in keep_columns and 'close' in df.columns:
                    keep_columns.append('close')
        
        # Filter columns, keeping only what we need
        available_columns = [col for col in keep_columns if col in df.columns]
        df = df[available_columns]
        
        # Store original columns for logging
        original_columns = list(df.columns)
        
        # 1. Add technical indicators (if they don't already exist)
        df = self._add_technical_indicators(df)
        
        # 2. Add temporal features
        df = self._add_temporal_features(df)
        
        # 3. Add statistical features
        df = self._add_statistical_features(df)
        
        # 4. Add lag features
        df = self._add_lag_features(df)
        
        # 5. Add difference features
        df = self._add_diff_features(df)
        
        # 6. Add custom calculated features
        df = self._add_custom_features(df)
        
        # 7. Drop unnecessary features
        if self.drop_features:
            df = df.drop(columns=[f for f in self.drop_features if f in df.columns])
        
        # 8. Add previous value for the target (useful for directional metrics)
        if self.target_column in df.columns:
            df[f'prev_{self.target_column}'] = df[self.target_column].shift(1)
        
        # 9. Remove rows with NaN values (typically just the first few due to lag features)
        df = df.dropna()
        
        # Log feature engineering summary
        new_columns = set(df.columns) - set(original_columns)
        dropped_columns = set(original_columns) - set(df.columns)
        
        logger.info(f"Feature engineering completed: {len(new_columns)} new features added, "
                   f"{len(dropped_columns)} features dropped, {len(df)} rows remaining")
        
        return df
    
    def _add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add technical indicators to the dataset.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with added technical indicators
        """
        if not self.technical_indicators:
            return df
        
        try:
            import ta
            
            for indicator in self.technical_indicators:
                # Skip if indicator already exists
                if indicator in df.columns:
                    continue
                
                # RSI (Relative Strength Index)
                if indicator.startswith('rsi_'):
                    window = int(indicator.split('_')[1])
                    if 'close' in df.columns and indicator not in df.columns:
                        df[indicator] = ta.momentum.RSIIndicator(df['close'], window=window).rsi()
                
                # SMA (Simple Moving Average)
                elif indicator.startswith('sma_'):
                    window = int(indicator.split('_')[1])
                    if 'close' in df.columns and indicator not in df.columns:
                        df[indicator] = ta.trend.SMAIndicator(df['close'], window=window).sma_indicator()
                
                # EMA (Exponential Moving Average)
                elif indicator.startswith('ema_'):
                    window = int(indicator.split('_')[1])
                    if 'close' in df.columns and indicator not in df.columns:
                        df[indicator] = ta.trend.EMAIndicator(df['close'], window=window).ema_indicator()
                
                # MACD (Moving Average Convergence Divergence)
                elif indicator == 'macd':
                    if 'close' in df.columns and indicator not in df.columns:
                        macd = ta.trend.MACD(df['close'])
                        df['macd'] = macd.macd()
                        df['macd_signal'] = macd.macd_signal()
                        df['macd_diff'] = macd.macd_diff()
                
                # Bollinger Bands
                elif indicator.startswith('bb_'):
                    if 'close' in df.columns and 'bb_upper' not in df.columns:
                        bollinger = ta.volatility.BollingerBands(df['close'])
                        df['bb_upper'] = bollinger.bollinger_hband()
                        df['bb_middle'] = bollinger.bollinger_mavg()
                        df['bb_lower'] = bollinger.bollinger_lband()
                        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
                
                # ATR (Average True Range)
                elif indicator == 'atr':
                    if all(col in df.columns for col in ['high', 'low', 'close']) and indicator not in df.columns:
                        df['atr'] = ta.volatility.AverageTrueRange(
                            df['high'], df['low'], df['close']
                        ).average_true_range()
                
        except ImportError:
            logger.warning("Could not import TA-Lib for technical indicators")
        except Exception as e:
            logger.warning(f"Error adding technical indicators: {str(e)}")
        
        return df
    
    def _add_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add temporal features (time-based) to the dataset.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with added temporal features
        """
        if not self.temporal_features or 'timestamp' not in df.columns:
            return df
        
        for feature in self.temporal_features:
            if feature == 'hour_of_day':
                df['hour_of_day'] = df['timestamp'].dt.hour
                # Normalize to [0, 1]
                df['hour_of_day'] = df['hour_of_day'] / 23.0
                
            elif feature == 'day_of_week':
                df['day_of_week'] = df['timestamp'].dt.dayofweek
                # Normalize to [0, 1]
                df['day_of_week'] = df['day_of_week'] / 6.0
                
            elif feature == 'day_of_month':
                df['day_of_month'] = df['timestamp'].dt.day
                # Normalize to [0, 1]
                df['day_of_month'] = df['day_of_month'] / 31.0
                
            elif feature == 'month_of_year':
                df['month_of_year'] = df['timestamp'].dt.month
                # Normalize to [0, 1]
                df['month_of_year'] = df['month_of_year'] / 12.0
                
            elif feature == 'week_of_year':
                df['week_of_year'] = df['timestamp'].dt.isocalendar().week
                # Normalize to [0, 1]
                df['week_of_year'] = df['week_of_year'] / 53.0
                
            elif feature == 'is_weekend':
                df['is_weekend'] = (df['timestamp'].dt.dayofweek >= 5).astype(float)
                
            elif feature == 'sin_hour':
                # Cyclical encoding of hour using sine
                df['sin_hour'] = np.sin(2 * np.pi * df['timestamp'].dt.hour / 24)
                
            elif feature == 'cos_hour':
                # Cyclical encoding of hour using cosine
                df['cos_hour'] = np.cos(2 * np.pi * df['timestamp'].dt.hour / 24)
                
            elif feature == 'sin_day':
                # Cyclical encoding of day using sine
                df['sin_day'] = np.sin(2 * np.pi * df['timestamp'].dt.dayofweek / 7)
                
            elif feature == 'cos_day':
                # Cyclical encoding of day using cosine
                df['cos_day'] = np.cos(2 * np.pi * df['timestamp'].dt.dayofweek / 7)
                
            elif feature == 'sin_month':
                # Cyclical encoding of month using sine
                df['sin_month'] = np.sin(2 * np.pi * df['timestamp'].dt.month / 12)
                
            elif feature == 'cos_month':
                # Cyclical encoding of month using cosine
                df['cos_month'] = np.cos(2 * np.pi * df['timestamp'].dt.month / 12)
        
        return df
    
    def _add_statistical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add statistical features to the dataset.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with added statistical features
        """
        if not self.statistical_features:
            return df
        
        for feature_config in self.statistical_features:
            # Parse configuration
            column = feature_config.get('column')
            window = feature_config.get('window')
            metric = feature_config.get('metric')
            
            if not column or not window or not metric or column not in df.columns:
                continue
            
            feature_name = f"{column}_{metric}_{window}"
            
            # Calculate rolling statistic
            if metric == 'mean':
                df[feature_name] = df[column].rolling(window=window).mean()
            elif metric == 'std':
                df[feature_name] = df[column].rolling(window=window).std()
            elif metric == 'var':
                df[feature_name] = df[column].rolling(window=window).var()
            elif metric == 'min':
                df[feature_name] = df[column].rolling(window=window).min()
            elif metric == 'max':
                df[feature_name] = df[column].rolling(window=window).max()
            elif metric == 'median':
                df[feature_name] = df[column].rolling(window=window).median()
            elif metric == 'skew':
                df[feature_name] = df[column].rolling(window=window).skew()
            elif metric == 'kurt':
                df[feature_name] = df[column].rolling(window=window).kurt()
            elif metric == 'q25':
                df[feature_name] = df[column].rolling(window=window).quantile(0.25)
            elif metric == 'q75':
                df[feature_name] = df[column].rolling(window=window).quantile(0.75)
        
        return df
    
    def _add_lag_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add lagged features to the dataset.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with added lag features
        """
        if not self.lag_features:
            return df
        
        for column, lags in self.lag_features.items():
            if column not in df.columns:
                continue
                
            for lag in lags:
                df[f"{column}_lag_{lag}"] = df[column].shift(lag)
        
        return df
    
    def _add_diff_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add differenced features to the dataset.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with added difference features
        """
        if not self.diff_features:
            return df
        
        for column, diffs in self.diff_features.items():
            if column not in df.columns:
                continue
                
            for diff in diffs:
                # First-order difference
                if isinstance(diff, int):
                    df[f"{column}_diff_{diff}"] = df[column].diff(diff)
                
                # Percentage change
                elif diff.startswith('pct_'):
                    period = int(diff.split('_')[1])
                    df[f"{column}_pct_{period}"] = df[column].pct_change(period)
                
                # Log difference
                elif diff.startswith('log_'):
                    period = int(diff.split('_')[1])
                    df[f"{column}_logdiff_{period}"] = np.log(df[column]).diff(period)
        
        return df
    
    def _add_custom_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add custom calculated features to the dataset.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with added custom features
        """
        if not self.custom_features:
            return df
        
        # Dictionary of common functions for convenience
        functions = {
            'log': np.log,
            'sqrt': np.sqrt,
            'square': lambda x: x**2,
            'cube': lambda x: x**3,
            'abs': np.abs,
            'sign': np.sign,
            'sigmoid': lambda x: 1 / (1 + np.exp(-x)),
            'tanh': np.tanh
        }
        
        for feature_name, config in self.custom_features.items():
            try:
                # Get formula type
                formula_type = config.get('type')
                
                if formula_type == 'arithmetic':
                    # Arithmetic formula (e.g., "A + B * C")
                    formula = config.get('formula')
                    variables = config.get('variables', {})
                    
                    # Replace variable placeholders with actual column names
                    for var_name, column in variables.items():
                        if column not in df.columns:
                            logger.warning(f"Column {column} for variable {var_name} not found")
                            break
                        
                        formula = formula.replace(var_name, f"df['{column}']")
                    
                    # Evaluate formula
                    df[feature_name] = eval(formula)
                    
                elif formula_type == 'function':
                    # Function application (e.g., "log(A / B)")
                    function_name = config.get('function')
                    column = config.get('column')
                    
                    if column not in df.columns:
                        logger.warning(f"Column {column} not found")
                        continue
                    
                    if function_name in functions:
                        df[feature_name] = functions[function_name](df[column])
                    else:
                        logger.warning(f"Unknown function {function_name}")
                
                elif formula_type == 'expression':
                    # Custom expression with eval
                    expression = config.get('expression')
                    df[feature_name] = eval(expression)
                
            except Exception as e:
                logger.warning(f"Error adding custom feature {feature_name}: {str(e)}")
        
        return df
    
    def get_config(self) -> Dict[str, Any]:
        """
        Get feature engineering configuration.
        
        Returns:
            Configuration dictionary
        """
        return {
            "feature_columns": self.feature_columns,
            "target_column": self.target_column,
            "technical_indicators": self.technical_indicators,
            "temporal_features": self.temporal_features,
            "statistical_features": self.statistical_features,
            "lag_features": self.lag_features,
            "diff_features": self.diff_features,
            "custom_features": {k: v for k, v in self.custom_features.items() if isinstance(v, dict)},
            "drop_features": self.drop_features
        }


class TimeSeriesTransformer:
    """
    Transformer for time series data preparation.
    
    This class handles the preparation of time series datasets, including:
    - Creating sliding window sequences
    - Scaling/normalizing data
    - Splitting into train/validation/test sets
    - Creating TensorFlow or PyTorch-compatible datasets
    """
    
    def __init__(self, 
                lookback_window: int,
                forecast_horizon: int = 1,
                scaling_method: str = "standard",
                target_column: str = None,
                sequence_stride: int = 1,
                batch_size: int = 32):
        """
        Initialize time series transformer.
        
        Args:
            lookback_window: Number of time steps to look back
            forecast_horizon: Number of time steps to forecast
            scaling_method: Method for scaling features
            target_column: Name of target column (if None, use last column)
            sequence_stride: Stride for sliding window
            batch_size: Batch size for datasets
        """
        self.lookback_window = lookback_window
        self.forecast_horizon = forecast_horizon
        self.scaling_method = ScalingMethod(scaling_method) if isinstance(scaling_method, str) else scaling_method
        self.target_column = target_column
        self.sequence_stride = sequence_stride
        self.batch_size = batch_size
        
        # Scalers to be fit during transform
        self.feature_scaler = None
        self.target_scaler = None
        
        # Store original test data for inverse transform
        self.original_test_data = None
        
        logger.info(f"Initialized TimeSeriesTransformer with lookback={lookback_window}, "
                  f"horizon={forecast_horizon}, scaling={self.scaling_method.value}")
    
    def create_train_val_test_datasets(self, 
                                     train_data: pd.DataFrame,
                                     val_data: pd.DataFrame,
                                     test_data: pd.DataFrame) -> Tuple:
        """
        Create training, validation, and test datasets.
        
        Args:
            train_data: Training data
            val_data: Validation data
            test_data: Test data
            
        Returns:
            Tuple of (X_train, y_train, X_val, y_val, X_test, y_test)
        """
        # Store original test data for evaluation metrics
        if 'timestamp' in test_data.columns:
            timestamp_col = test_data['timestamp']
            test_data = test_data.drop(columns=['timestamp'])
            self.original_test_data = test_data.copy()
            self.original_test_data['timestamp'] = timestamp_col
        else:
            self.original_test_data = test_data.copy()
        
        # Determine target column index
        if self.target_column is not None and self.target_column in train_data.columns:
            target_idx = train_data.columns.get_loc(self.target_column)
        else:
            # Default to last column
            target_idx = -1
            self.target_column = train_data.columns[target_idx]
        
        # Apply scaling
        if 'timestamp' in train_data.columns:
            train_data = train_data.drop(columns=['timestamp'])
        if 'timestamp' in val_data.columns:
            val_data = val_data.drop(columns=['timestamp'])
        if 'timestamp' in test_data.columns:
            test_data = test_data.drop(columns=['timestamp'])
        
        # Scale features
        if self.scaling_method != ScalingMethod.NONE:
            train_data, val_data, test_data = self._apply_scaling(
                train_data, val_data, test_data, target_idx
            )
        
        # Convert to numpy arrays for sequence creation
        train_array = train_data.values
        val_array = val_data.values
        test_array = test_data.values
        
        # Create sequences
        X_train, y_train = self._create_sequences(train_array, target_idx)
        X_val, y_val = self._create_sequences(val_array, target_idx)
        X_test, y_test = self._create_sequences(test_array, target_idx)
        
        # Add column to original test data with previous target value
        if self.target_column in self.original_test_data.columns:
            self.original_test_data[f'prev_{self.target_column}'] = self.original_test_data[self.target_column].shift(1)
            self.original_test_data = self.original_test_data.dropna()
            
            # Adjust length to match X_test
            if len(self.original_test_data) > len(X_test):
                self.original_test_data = self.original_test_data.tail(len(X_test))
        
        logger.info(f"Created sequences - X_train: {X_train.shape}, y_train: {y_train.shape}, "
                  f"X_val: {X_val.shape}, y_val: {y_val.shape}, X_test: {X_test.shape}, y_test: {y_test.shape}")
        
        return X_train, y_train, X_val, y_val, X_test, y_test
    
    def _apply_scaling(self, 
                     train_data: pd.DataFrame,
                     val_data: pd.DataFrame,
                     test_data: pd.DataFrame,
                     target_idx: int) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Apply scaling to datasets.
        
        Args:
            train_data: Training data
            val_data: Validation data
            test_data: Test data
            target_idx: Index of target column
            
        Returns:
            Tuple of scaled (train_data, val_data, test_data)
        """
        # Extract target column
        target_train = train_data.iloc[:, target_idx].values.reshape(-1, 1)
        target_val = val_data.iloc[:, target_idx].values.reshape(-1, 1)
        target_test = test_data.iloc[:, target_idx].values.reshape(-1, 1)
        
        # Prepare feature data
        feature_columns = list(train_data.columns)
        feature_columns.pop(target_idx)
        
        features_train = train_data[feature_columns]
        features_val = val_data[feature_columns]
        features_test = test_data[feature_columns]
        
        # Initialize and fit scalers
        if self.scaling_method == ScalingMethod.STANDARD:
            self.feature_scaler = StandardScaler()
            self.target_scaler = StandardScaler()
        elif self.scaling_method == ScalingMethod.MINMAX:
            self.feature_scaler = MinMaxScaler()
            self.target_scaler = MinMaxScaler()
        elif self.scaling_method == ScalingMethod.ROBUST:
            self.feature_scaler = RobustScaler()
            self.target_scaler = RobustScaler()
        elif self.scaling_method == ScalingMethod.PERCENTAGE_CHANGE:
            # No scaler needed, will apply pct_change directly
            pass
        elif self.scaling_method == ScalingMethod.LOG_TRANSFORM:
            # No scaler needed, will apply log transform directly
            pass
        
        # Apply scaling based on method
        if self.scaling_method in [ScalingMethod.STANDARD, ScalingMethod.MINMAX, ScalingMethod.ROBUST]:
            # Fit and transform features
            features_train_scaled = self.feature_scaler.fit_transform(features_train)
            features_val_scaled = self.feature_scaler.transform(features_val)
            features_test_scaled = self.feature_scaler.transform(features_test)
            
            # Fit and transform target
            target_train_scaled = self.target_scaler.fit_transform(target_train)
            target_val_scaled = self.target_scaler.transform(target_val)
            target_test_scaled = self.target_scaler.transform(target_test)
            
            # Create scaled DataFrames
            train_data_scaled = pd.DataFrame(
                np.hstack([features_train_scaled, target_train_scaled]),
                columns=list(feature_columns) + [self.target_column]
            )
            
            val_data_scaled = pd.DataFrame(
                np.hstack([features_val_scaled, target_val_scaled]),
                columns=list(feature_columns) + [self.target_column]
            )
            
            test_data_scaled = pd.DataFrame(
                np.hstack([features_test_scaled, target_test_scaled]),
                columns=list(feature_columns) + [self.target_column]
            )
            
            return train_data_scaled, val_data_scaled, test_data_scaled
            
        elif self.scaling_method == ScalingMethod.PERCENTAGE_CHANGE:
            # Apply percentage change
            train_pct = train_data.pct_change().fillna(0)
            val_pct = val_data.pct_change().fillna(0)
            test_pct = test_data.pct_change().fillna(0)
            
            return train_pct, val_pct, test_pct
            
        elif self.scaling_method == ScalingMethod.LOG_TRANSFORM:
            # Apply log transform (with offset for zero/negative values)
            train_log = np.log(train_data + 1e-8)
            val_log = np.log(val_data + 1e-8)
            test_log = np.log(test_data + 1e-8)
            
            return train_log, val_log, test_log
            
        else:
            # No scaling
            return train_data, val_data, test_data
    
    def _create_sequences(self, data: np.ndarray, target_idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sliding window sequences from data.
        
        Args:
            data: Input data array
            target_idx: Index of target column
            
        Returns:
            Tuple of (X, y) arrays
        """
        X, y = [], []
        
        for i in range(0, len(data) - self.lookback_window - self.forecast_horizon + 1, self.sequence_stride):
            # Extract features sequence
            X.append(data[i:i+self.lookback_window])
            
            # Extract target
            if self.forecast_horizon == 1:
                y.append(data[i+self.lookback_window, target_idx])
            else:
                y.append(data[i+self.lookback_window:i+self.lookback_window+self.forecast_horizon, target_idx])
        
        return np.array(X), np.array(y)
    
    def inverse_transform_y(self, y: np.ndarray) -> np.ndarray:
        """
        Apply inverse transform to target values.
        
        Args:
            y: Scaled target values
            
        Returns:
            Original-scale target values
        """
        if self.target_scaler is not None:
            # Reshape for scaler if needed
            if len(y.shape) == 1:
                y_reshaped = y.reshape(-1, 1)
            else:
                y_reshaped = y
            
            # Apply inverse transform
            return self.target_scaler.inverse_transform(y_reshaped).flatten()
        else:
            # No scaling was applied
            return y
    
    def get_original_test_data(self) -> pd.DataFrame:
        """
        Get original test data for evaluation.
        
        Returns:
            Original test data
        """
        return self.original_test_data
    
    def create_tf_dataset(self, 
                        X: np.ndarray, 
                        y: np.ndarray, 
                        batch_size: int = None,
                        shuffle: bool = False,
                        cache: bool = False) -> tf.data.Dataset:
        """
        Create TensorFlow dataset from X and y arrays.
        
        Args:
            X: Features array
            y: Target array
            batch_size: Batch size (None to use default)
            shuffle: Whether to shuffle the dataset
            cache: Whether to cache the dataset
            
        Returns:
            TensorFlow Dataset
        """
        batch_size = batch_size or self.batch_size
        
        # Create dataset
        dataset = tf.data.Dataset.from_tensor_slices((X, y))
        
        # Cache if requested
        if cache:
            dataset = dataset.cache()
        
        # Shuffle if requested
        if shuffle:
            dataset = dataset.shuffle(buffer_size=len(X))
        
        # Batch
        dataset = dataset.batch(batch_size)
        
        # Prefetch for performance
        dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
        
        return dataset
    
    def create_torch_dataloader(self,
                              X: np.ndarray,
                              y: np.ndarray,
                              batch_size: int = None,
                              shuffle: bool = False) -> DataLoader:
        """
        Create PyTorch DataLoader from X and y arrays.
        
        Args:
            X: Features array
            y: Target array
            batch_size: Batch size (None to use default)
            shuffle: Whether to shuffle the dataset
            
        Returns:
            PyTorch DataLoader
        """
        batch_size = batch_size or self.batch_size
        
        # Convert numpy arrays to PyTorch tensors
        X_tensor = torch.tensor(X, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.float32)
        
        # Create dataset
        dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
        
        # Create dataloader
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=0,  # No multiprocessing for now
            pin_memory=torch.cuda.is_available()  # Pin memory if CUDA is available
        )
        
        return dataloader


class SequenceDataset(Dataset):
    """
    PyTorch Dataset for time series sequences.
    
    This class provides a PyTorch Dataset implementation for handling
    time series data with lookback windows and forecast horizons.
    """
    
    def __init__(self, 
                data: pd.DataFrame,
                lookback_window: int,
                forecast_horizon: int = 1,
                target_column: str = None,
                feature_columns: List[str] = None,
                scaling_method: str = "standard",
                sequence_stride: int = 1):
        """
        Initialize sequence dataset.
        
        Args:
            data: Input DataFrame
            lookback_window: Number of time steps to look back
            forecast_horizon: Number of time steps to forecast
            target_column: Name of target column (if None, use last column)
            feature_columns: List of feature columns (if None, use all except target)
            scaling_method: Method for scaling features
            sequence_stride: Stride for sliding window
        """
        self.data = data.copy()
        self.lookback_window = lookback_window
        self.forecast_horizon = forecast_horizon
        self.sequence_stride = sequence_stride
        
        # Determine target column
        if target_column is None:
            target_column = data.columns[-1]
        self.target_column = target_column
        
        # Determine feature columns
        if feature_columns is None:
            feature_columns = [col for col in data.columns if col != target_column]
        self.feature_columns = feature_columns
        
        # Extract features and target
        self.features = data[feature_columns].values
        self.targets = data[target_column].values
        
        # Apply scaling
        self.scaling_method = ScalingMethod(scaling_method) if isinstance(scaling_method, str) else scaling_method
        self._apply_scaling()
        
        # Create sequence indices
        self.sequence_indices = self._create_sequence_indices()
        
        logger.info(f"Created SequenceDataset with {len(self.sequence_indices)} sequences")
    
    def _apply_scaling(self):
        """Apply scaling to features and target."""
        if self.scaling_method == ScalingMethod.STANDARD:
            self.feature_scaler = StandardScaler()
            self.target_scaler = StandardScaler()
            
            self.features = self.feature_scaler.fit_transform(self.features)
            self.targets = self.target_scaler.fit_transform(self.targets.reshape(-1, 1)).flatten()
            
        elif self.scaling_method == ScalingMethod.MINMAX:
            self.feature_scaler = MinMaxScaler()
            self.target_scaler = MinMaxScaler()
            
            self.features = self.feature_scaler.fit_transform(self.features)
            self.targets = self.target_scaler.fit_transform(self.targets.reshape(-1, 1)).flatten()
            
        elif self.scaling_method == ScalingMethod.ROBUST:
            self.feature_scaler = RobustScaler()
            self.target_scaler = RobustScaler()
            
            self.features = self.feature_scaler.fit_transform(self.features)
            self.targets = self.target_scaler.fit_transform(self.targets.reshape(-1, 1)).flatten()
            
        elif self.scaling_method == ScalingMethod.PERCENTAGE_CHANGE:
            # Apply percentage change calculation
            features_df = pd.DataFrame(self.features)
            targets_series = pd.Series(self.targets)
            
            self.features = features_df.pct_change().fillna(0).values
            self.targets = targets_series.pct_change().fillna(0).values
            
        elif self.scaling_method == ScalingMethod.LOG_TRANSFORM:
            # Apply log transform with offset for zero/negative values
            self.features = np.log(np.abs(self.features) + 1e-8)
            self.targets = np.log(np.abs(self.targets) + 1e-8)
    
    def _create_sequence_indices(self) -> List[int]:
        """
        Create list of valid sequence start indices.
        
        Returns:
            List of sequence start indices
        """
        indices = []
        
        for i in range(0, len(self.features) - self.lookback_window - self.forecast_horizon + 1, self.sequence_stride):
            indices.append(i)
        
        return indices
    
    def __len__(self) -> int:
        """
        Get dataset length.
        
        Returns:
            Number of sequences in dataset
        """
        return len(self.sequence_indices)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get sequence by index.
        
        Args:
            idx: Sequence index
            
        Returns:
            Tuple of (feature sequence, target)
        """
        # Get sequence start index
        seq_idx = self.sequence_indices[idx]
        
        # Extract feature sequence
        X = self.features[seq_idx:seq_idx+self.lookback_window]
        
        # Extract target
        if self.forecast_horizon == 1:
            y = self.targets[seq_idx+self.lookback_window]
        else:
            y = self.targets[seq_idx+self.lookback_window:seq_idx+self.lookback_window+self.forecast_horizon]
        
        # Convert to PyTorch tensors
        X_tensor = torch.FloatTensor(X)
        y_tensor = torch.FloatTensor(y)
        
        return X_tensor, y_tensor
    
    def inverse_transform_y(self, y: torch.Tensor) -> np.ndarray:
        """
        Apply inverse transform to target values.
        
        Args:
            y: Scaled target tensor
            
        Returns:
            Original-scale target values
        """
        # Convert to numpy
        y_np = y.detach().cpu().numpy()
        
        if hasattr(self, 'target_scaler'):
            # Reshape for scaler if needed
            if len(y_np.shape) == 1:
                y_reshaped = y_np.reshape(-1, 1)
            else:
                y_reshaped = y_np
            
            # Apply inverse transform
            return self.target_scaler.inverse_transform(y_reshaped).flatten()
        else:
            # No scaling was applied or it can't be inverted
            return y_np


def prepare_time_series_dataset(
    data: pd.DataFrame,
    target_column: str,
    feature_columns: List[str] = None,
    lookback_window: int = 24,
    forecast_horizon: int = 1,
    test_size: float = 0.2,
    val_size: float = 0.1,
    scaling_method: str = "standard"
) -> Dict[str, Union[np.ndarray, pd.DataFrame]]:
    """
    Prepare time series dataset for training.
    
    Args:
        data: Input DataFrame
        target_column: Name of target column
        feature_columns: List of feature columns
        lookback_window: Number of time steps to look back
        forecast_horizon: Number of time steps to forecast
        test_size: Fraction of data to use for testing
        val_size: Fraction of training data to use for validation
        scaling_method: Method for scaling features
        
    Returns:
        Dictionary containing dataset components
    """
    # Ensure all data is numeric
    if feature_columns is None:
        numeric_columns = data.select_dtypes(include=[np.number]).columns.tolist()
        feature_columns = [col for col in numeric_columns if col != target_column]
    
    # Select relevant columns
    cols_to_use = feature_columns + [target_column]
    data = data[cols_to_use].copy()
    
    # Check for NaN values
    if data.isna().any().any():
        logger.warning(f"Data contains NaN values. Filling with forward fill then backward fill.")
        data = data.ffill().bfill()
    
    # Split data chronologically
    n = len(data)
    test_idx = int(n * (1 - test_size))
    
    train_val_data = data.iloc[:test_idx].copy()
    test_data = data.iloc[test_idx:].copy()
    
    # Further split training data into train and validation
    train_idx = int(len(train_val_data) * (1 - val_size))
    
    train_data = train_val_data.iloc[:train_idx].copy()
    val_data = train_val_data.iloc[train_idx:].copy()
    
    logger.info(f"Data split: train={len(train_data)}, validation={len(val_data)}, test={len(test_data)}")
    
    # Create transformer
    transformer = TimeSeriesTransformer(
        lookback_window=lookback_window,
        forecast_horizon=forecast_horizon,
        scaling_method=scaling_method,
        target_column=target_column
    )
    
    # Create datasets
    X_train, y_train, X_val, y_val, X_test, y_test = transformer.create_train_val_test_datasets(
        train_data, val_data, test_data
    )
    
    # Return as dictionary
    return {
        "X_train": X_train,
        "y_train": y_train,
        "X_val": X_val,
        "y_val": y_val,
        "X_test": X_test,
        "y_test": y_test,
        "train_data": train_data,
        "val_data": val_data,
        "test_data": test_data,
        "transformer": transformer
    }


def create_sliding_window_dataset(
    data: np.ndarray,
    lookback_window: int,
    forecast_horizon: int = 1,
    target_idx: int = -1,
    sequence_stride: int = 1
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create sliding window sequences from data.
    
    Args:
        data: Input data array
        lookback_window: Number of time steps to look back
        forecast_horizon: Number of time steps to forecast
        target_idx: Index of target column
        sequence_stride: Stride for sliding window
        
    Returns:
        Tuple of (X, y) arrays
    """
    X, y = [], []
    
    for i in range(0, len(data) - lookback_window - forecast_horizon + 1, sequence_stride):
        # Extract features sequence
        X.append(data[i:i+lookback_window])
        
        # Extract target
        if forecast_horizon == 1:
            y.append(data[i+lookback_window, target_idx])
        else:
            y.append(data[i+lookback_window:i+lookback_window+forecast_horizon, target_idx])
    
    return np.array(X), np.array(y)


def apply_data_augmentation(
    X: np.ndarray,
    y: np.ndarray,
    augmentation_methods: List[str] = None,
    augmentation_ratio: float = 0.5
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply data augmentation to time series dataset.
    
    Args:
        X: Features sequences array of shape (n_samples, lookback, n_features)
        y: Target array
        augmentation_methods: List of augmentation methods to apply
        augmentation_ratio: Ratio of augmented samples to original samples
        
    Returns:
        Tuple of augmented (X, y) arrays
    """
    if augmentation_methods is None:
        augmentation_methods = ["jitter", "scaling", "time_warp", "window_slice"]
    
    n_samples = X.shape[0]
    n_augmented = int(n_samples * augmentation_ratio)
    
    if n_augmented <= 0:
        return X, y
    
    # Sample indices to augment
    indices = np.random.choice(n_samples, n_augmented, replace=True)
    
    # Create arrays for augmented data
    X_augmented = np.zeros((n_augmented, X.shape[1], X.shape[2]), dtype=X.dtype)
    y_augmented = np.zeros(n_augmented, dtype=y.dtype) if len(y.shape) == 1 else np.zeros((n_augmented, y.shape[1]), dtype=y.dtype)
    
    # Apply augmentation
    for i, idx in enumerate(indices):
        # Choose a random augmentation method
        method = np.random.choice(augmentation_methods)
        
        # Get sample to augment
        X_sample = X[idx].copy()
        y_sample = y[idx].copy()
        
        # Apply augmentation
        if method == "jitter":
            # Add random noise
            noise_level = np.random.uniform(0.01, 0.05)
            noise = np.random.normal(0, noise_level, X_sample.shape)
            X_augmented[i] = X_sample + noise
            y_augmented[i] = y_sample
            
        elif method == "scaling":
            # Apply random scaling
            scale_factor = np.random.uniform(0.8, 1.2)
            X_augmented[i] = X_sample * scale_factor
            
            # Scale target proportionally if it's a price forecast
            if len(y.shape) == 1:
                y_augmented[i] = y_sample * scale_factor
            else:
                y_augmented[i] = y_sample
                
        elif method == "time_warp":
            # Apply time warping (stretching/compressing sections of the sequence)
            lookback = X_sample.shape[0]
            warp_idx = np.random.randint(1, lookback - 1)
            warp_factor = np.random.uniform(0.5, 1.5)
            
            # Create warped sequence
            X_warped = np.zeros_like(X_sample)
            X_warped[:warp_idx] = X_sample[:warp_idx]
            
            # Stretch or compress remaining part
            orig_indices = np.linspace(warp_idx, lookback - 1, lookback - warp_idx)
            warped_indices = np.linspace(warp_idx, lookback - 1, int((lookback - warp_idx) * warp_factor))
            
            if len(warped_indices) < len(orig_indices):
                # Stretch (fewer warped indices)
                for j in range(warp_idx, lookback):
                    if j - warp_idx < len(warped_indices):
                        warped_idx = int(warped_indices[j - warp_idx])
                        if warped_idx < lookback:
                            X_warped[j] = X_sample[warped_idx]
                        else:
                            X_warped[j] = X_sample[-1]
                    else:
                        X_warped[j] = X_sample[-1]
            else:
                # Compress (more warped indices)
                for j in range(warp_idx, lookback):
                    orig_idx = int(orig_indices[j - warp_idx]) if j - warp_idx < len(orig_indices) else lookback - 1
                    X_warped[j] = X_sample[orig_idx]
            
            X_augmented[i] = X_warped
            y_augmented[i] = y_sample
            
        elif method == "window_slice":
            # Extract a random slice and resize
            lookback = X_sample.shape[0]
            slice_ratio = np.random.uniform(0.7, 0.9)
            slice_length = int(lookback * slice_ratio)
            start_idx = np.random.randint(0, lookback - slice_length)
            
            # Extract slice
            X_slice = X_sample[start_idx:start_idx+slice_length]
            
            # Resize to original length
            X_augmented[i] = np.array([
                np.interp(
                    np.linspace(0, 1, lookback),
                    np.linspace(0, 1, slice_length),
                    X_slice[:, j]
                ) for j in range(X_sample.shape[1])
            ]).T
            
            y_augmented[i] = y_sample
    
    # Combine original and augmented data
    X_combined = np.vstack([X, X_augmented])
    
    if len(y.shape) == 1:
        y_combined = np.concatenate([y, y_augmented])
    else:
        y_combined = np.vstack([y, y_augmented])
    
    logger.info(f"Applied data augmentation: original={X.shape}, augmented={X_augmented.shape}, combined={X_combined.shape}")
    
    return X_combined, y_combined


def create_multivariate_time_series_dataset(
    data: Dict[str, pd.DataFrame],
    target_column: str,
    common_index_column: str = "timestamp",
    lookback_window: int = 24,
    forecast_horizon: int = 1,
    test_size: float = 0.2,
    val_size: float = 0.1,
    scaling_method: str = "standard"
) -> Dict[str, Union[np.ndarray, pd.DataFrame]]:
    """
    Create multivariate time series dataset from multiple data sources.
    
    Args:
        data: Dictionary of DataFrames from different sources
        target_column: Name of target column
        common_index_column: Name of common index column
        lookback_window: Number of time steps to look back
        forecast_horizon: Number of time steps to forecast
        test_size: Fraction of data to use for testing
        val_size: Fraction of training data to use for validation
        scaling_method: Method for scaling features
        
    Returns:
        Dictionary containing dataset components
    """
    # Merge data sources on common index
    data_keys = list(data.keys())
    
    # Start with first data source
    merged_data = data[data_keys[0]].copy()
    merged_data.set_index(common_index_column, inplace=True)
    
    # Add suffixes to column names to avoid duplicates
    for k, df in list(data.items())[1:]:
        temp_df = df.copy()
        temp_df.set_index(common_index_column, inplace=True)
        
        # Rename columns to avoid duplicates
        temp_df.columns = [f"{col}_{k}" for col in temp_df.columns]
        
        # Merge with existing data
        merged_data = merged_data.join(temp_df, how='outer')
    
    # Reset index
    merged_data.reset_index(inplace=True)
    
    # Sort by time
    merged_data.sort_values(by=common_index_column, inplace=True)
    
    # Handle missing values
    merged_data = merged_data.ffill().bfill()
    
    # Prepare dataset
    return prepare_time_series_dataset(
        data=merged_data,
        target_column=target_column,
        lookback_window=lookback_window,
        forecast_horizon=forecast_horizon,
        test_size=test_size,
        val_size=val_size,
        scaling_method=scaling_method
    )


# Example usage
if __name__ == "__main__":
    # Create sample data
    dates = pd.date_range(start='2020-01-01', periods=1000, freq='1H')
    df = pd.DataFrame({
        'timestamp': dates,
        'price': np.sin(np.linspace(0, 15, 1000)) * 100 + 100 + np.random.normal(0, 5, 1000),
        'volume': np.random.exponential(1, 1000) * 1000,
        'feature1': np.random.normal(0, 1, 1000),
        'feature2': np.random.normal(0, 1, 1000)
    })
    
    # Create time series dataset
    dataset = prepare_time_series_dataset(
        data=df,
        target_column='price',
        feature_columns=['price', 'volume', 'feature1', 'feature2'],
        lookback_window=24,
        forecast_horizon=1,
        test_size=0.2,
        val_size=0.1
    )
    
    # Print dataset info
    for k, v in dataset.items():
        if isinstance(v, np.ndarray):
            print(f"{k}: shape={v.shape}")
        elif isinstance(v, pd.DataFrame):
            print(f"{k}: shape={v.shape}")