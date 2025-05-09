if n_samples <= 0:
            logger.warning(f"Invalid number of augmented samples: {n_samples}, no augmentation performed")
            return X, y
        
        # Prepare arrays for augmented data
        X_aug = []
        y_aug = []
        
        # Generate augmented samples
        for _ in range(n_samples):
            # Select a random sample to augment
            idx = np.random.randint(0, len(X_orig))
            X_sample = X_orig[idx].copy()
            y_sample = y_orig[idx].copy()
            
            # Select a random augmentation method
            method = np.random.choice(augmentation_methods)
            
            if method == "jitter":
                # Add random noise to the sequence
                noise_level = self.config.get("jitter_noise_level", 0.01)
                X_sample += np.random.normal(0, noise_level, X_sample.shape)
            
            elif method == "scaling":
                # Apply random scaling to the sequence
                scaling_factor = np.random.uniform(0.9, 1.1)
                X_sample *= scaling_factor
            
            elif method == "permutation":
                # Permute small segments within the sequence
                segment_size = self.config.get("permutation_segment_size", 5)
                
                if segment_size < X_sample.shape[0]:
                    # Select random segments and permute
                    for _ in range(self.config.get("permutation_count", 2)):
                        segment_start = np.random.randint(0, X_sample.shape[0] - segment_size)
                        segment_end = segment_start + segment_size
                        
                        # Permute the segment (shuffle rows)
                        permutation = np.random.permutation(segment_size)
                        X_sample[segment_start:segment_end] = X_sample[segment_start + permutation]
            
            elif method == "time_warp":
                # Time warping (stretch or compress parts of the sequence)
                if X_sample.shape[0] > 10:
                    # Create warping function
                    time_indices = np.arange(X_sample.shape[0])
                    warp_points = np.random.choice(
                        time_indices[5:-5], 
                        size=self.config.get("warp_points", 3), 
                        replace=False
                    )
                    warp_offsets = np.random.uniform(-0.2, 0.2, size=len(warp_points))
                    
                    # Create warped time indices
                    warped_indices = time_indices.copy().astype(float)
                    
                    for point, offset in zip(warp_points, warp_offsets):
                        warped_indices[point:] += offset
                    
                    # Ensure indices remain in bounds
                    warped_indices = np.clip(warped_indices, 0, X_sample.shape[0] - 1)
                    
                    # Interpolate using warped indices
                    for i in range(X_sample.shape[1]):
                        X_sample[:, i] = np.interp(time_indices, warped_indices, X_sample[:, i])
            
            elif method == "magnitude_warp":
                # Magnitude warping (change magnitude of features)
                # Create warping function using a smooth curve
                warp_peaks = self.config.get("magnitude_warp_peaks", 2)
                knot_points = np.random.choice(
                    np.arange(1, X_sample.shape[0] - 1), 
                    size=warp_peaks, 
                    replace=False
                )
                knot_points = np.sort(np.concatenate([[0], knot_points, [X_sample.shape[0] - 1]]))
                
                # Generate random magnitudes at knot points
                magnitudes = np.random.uniform(0.9, 1.1, size=len(knot_points))
                
                # Create spline interpolation
                time_indices = np.arange(X_sample.shape[0])
                warp_curve = np.interp(time_indices, knot_points, magnitudes)
                
                # Apply warping
                for i in range(X_sample.shape[1]):
                    X_sample[:, i] *= warp_curve
            
            elif method == "window_slice":
                # Slice a moving window through the sequence
                window_size = self.config.get("window_slice_size", int(X_sample.shape[0] * 0.9))
                
                if window_size < X_sample.shape[0]:
                    # Select random start point
                    start = np.random.randint(0, X_sample.shape[0] - window_size)
                    
                    # Extract window and resize back to original length
                    window = X_sample[start:start + window_size].copy()
                    X_sample = np.zeros_like(X_sample)
                    
                    # Simple linear interpolation for resizing
                    for i in range(X_sample.shape[1]):
                        X_sample[:, i] = np.interp(
                            np.linspace(0, 1, X_sample.shape[0]),
                            np.linspace(0, 1, window_size),
                            window[:, i]
                        )
            
            elif method == "window_warp":
                # Warp specific windows within the sequence
                window_size = self.config.get("window_warp_size", int(X_sample.shape[0] * 0.2))
                
                if window_size < X_sample.shape[0] // 2:
                    # Select random window
                    start = np.random.randint(0, X_sample.shape[0] - window_size)
                    end = start + window_size
                    
                    # Apply warping factor to window
                    warp_factor = np.random.uniform(0.8, 1.2)
                    X_sample[start:end] *= warp_factor
            
            else:
                logger.warning(f"Unknown augmentation method: {method}")
                continue
            
            # Add augmented sample to results
            X_aug.append(X_sample)
            y_aug.append(y_sample)
        
        # Combine original and augmented data
        X_combined = np.vstack([X_orig, np.array(X_aug)])
        
        # Handle different shapes of y
        if len(y_orig.shape) == 1:
            y_combined = np.concatenate([y_orig, np.array(y_aug)])
        else:
            y_combined = np.vstack([y_orig, np.array(y_aug)])
        
        logger.info(f"Data augmentation complete: {len(X_orig)} original samples + {len(X_aug)} augmented samples = {len(X_combined)} total samples")
        
        return X_combined, y_combined
    
    def prepare_data_pipeline(
        self,
        data_path: str,
        additional_data_paths: Optional[Dict[str, str]] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        save_processed_data: bool = True
    ) -> Dict[str, Any]:
        """
        Full data preparation pipeline from raw data to model inputs.
        
        Args:
            data_path: Path to the main price data file
            additional_data_paths: Dictionary mapping data type to file path for additional data
            start_date: Optional start date for filtering data
            end_date: Optional end date for filtering data
            save_processed_data: Whether to save processed data and transformers
            
        Returns:
            Dictionary containing processed data and metadata
        """
        logger.info("Starting data preparation pipeline")
        
        # Initialize results dictionary
        result = {}
        
        try:
            # 1. Load data
            df = self.load_data(data_path, additional_data_paths, start_date, end_date)
            result['raw_data'] = df
            
            # 2. Add technical indicators
            df = self.add_technical_indicators(df)
            
            # 3. Add market microstructure features
            df = self.add_market_microstructure_features(df)
            
            # 4. Add on-chain features (if available)
            if additional_data_paths and 'on_chain' in additional_data_paths:
                df = self.add_on_chain_features(df)
            
            # 5. Add sentiment features (if available)
            if additional_data_paths and 'sentiment' in additional_data_paths:
                df = self.add_sentiment_features(df)
            
            # 6. Add global market features (if available)
            if additional_data_paths and 'market' in additional_data_paths:
                df = self.add_global_market_features(df)
            
            # 7. Handle missing values
            df = self.handle_missing_values(df)
            
            # 8. Detect and handle outliers
            df = self.detect_and_handle_outliers(df)
            
            # 9. Split the data
            split_method = self.config.get("split_method", "time")
            train_df, val_df, test_df = self.train_val_test_split(
                df,
                split_method=split_method,
                train_ratio=self.train_val_test_split[0],
                val_ratio=self.train_val_test_split[1],
                test_ratio=self.train_val_test_split[2]
            )
            
            # 10. Normalize features
            train_df_norm = self.normalize_features(train_df, is_train=True)
            val_df_norm = self.normalize_features(val_df, is_train=False)
            test_df_norm = self.normalize_features(test_df, is_train=False)
            
            # 11. Apply dimension reduction
            apply_dim_reduction = self.config.get("apply_dimension_reduction", False)
            if apply_dim_reduction:
                train_df_norm = self.apply_dimension_reduction(train_df_norm, is_train=True)
                val_df_norm = self.apply_dimension_reduction(val_df_norm, is_train=False)
                test_df_norm = self.apply_dimension_reduction(test_df_norm, is_train=False)
            
            # Store preprocessed DataFrames
            result['train_df'] = train_df_norm
            result['val_df'] = val_df_norm
            result['test_df'] = test_df_norm
            
            # 12. Create sequences or prepare for classification
            model_mode = self.config.get("mode", "regression")
            
            if model_mode == "regression":
                # Create sequences for regression
                X_train, y_train = self.create_sequences(
                    train_df_norm,
                    sequence_length=self.sequence_length,
                    prediction_horizon=self.prediction_horizon,
                    target_column=self.target_column
                )
                
                X_val, y_val = self.create_sequences(
                    val_df_norm,
                    sequence_length=self.sequence_length,
                    prediction_horizon=self.prediction_horizon,
                    target_column=self.target_column
                )
                
                X_test, y_test = self.create_sequences(
                    test_df_norm,
                    sequence_length=self.sequence_length,
                    prediction_horizon=self.prediction_horizon,
                    target_column=self.target_column
                )
                
            elif model_mode == "classification":
                # Prepare data for classification
                label_type = self.config.get("label_type", "binary")
                threshold = self.config.get("classification_threshold", 0.0)
                
                # Create classification labels
                y_train_labels, train_valid_indices = self.create_classification_labels(
                    train_df_norm,
                    target_column=self.target_column,
                    threshold=threshold,
                    horizon=self.prediction_horizon,
                    label_type=label_type
                )
                
                y_val_labels, val_valid_indices = self.create_classification_labels(
                    val_df_norm,
                    target_column=self.target_column,
                    threshold=threshold,
                    horizon=self.prediction_horizon,
                    label_type=label_type
                )
                
                y_test_labels, test_valid_indices = self.create_classification_labels(
                    test_df_norm,
                    target_column=self.target_column,
                    threshold=threshold,
                    horizon=self.prediction_horizon,
                    label_type=label_type
                )
                
                # Filter data frames to valid indices
                train_df_valid = train_df_norm[train_valid_indices].copy()
                val_df_valid = val_df_norm[val_valid_indices].copy()
                test_df_valid = test_df_norm[test_valid_indices].copy()
                
                # Create sequences
                X_train, _ = self.create_sequences(
                    train_df_valid,
                    sequence_length=self.sequence_length,
                    prediction_horizon=1,  # For classification, prediction horizon is handled by labels
                    target_column=self.target_column
                )
                
                X_val, _ = self.create_sequences(
                    val_df_valid,
                    sequence_length=self.sequence_length,
                    prediction_horizon=1,
                    target_column=self.target_column
                )
                
                X_test, _ = self.create_sequences(
                    test_df_valid,
                    sequence_length=self.sequence_length,
                    prediction_horizon=1,
                    target_column=self.target_column
                )
                
                # Use the labels as targets
                y_train = y_train_labels
                y_val = y_val_labels
                y_test = y_test_labels
                
            else:
                # Multi-task or custom mode
                logger.warning(f"Unsupported model mode: {model_mode}, falling back to regression")
                
                # Create sequences for regression
                X_train, y_train = self.create_sequences(
                    train_df_norm,
                    sequence_length=self.sequence_length,
                    prediction_horizon=self.prediction_horizon,
                    target_column=self.target_column
                )
                
                X_val, y_val = self.create_sequences(
                    val_df_norm,
                    sequence_length=self.sequence_length,
                    prediction_horizon=self.prediction_horizon,
                    target_column=self.target_column
                )
                
                X_test, y_test = self.create_sequences(
                    test_df_norm,
                    sequence_length=self.sequence_length,
                    prediction_horizon=self.prediction_horizon,
                    target_column=self.target_column
                )
            
            # 13. Apply data augmentation (to training data only)
            apply_augmentation = self.config.get("apply_data_augmentation", False)
            if apply_augmentation:
                augmentation_factor = self.config.get("augmentation_factor", 1.5)
                X_train, y_train = self.apply_data_augmentation(
                    X_train, y_train, augmentation_factor
                )
            
            # Store processed sequences
            result['X_train'] = X_train
            result['y_train'] = y_train
            result['X_val'] = X_val
            result['y_val'] = y_val
            result['X_test'] = X_test
            result['y_test'] = y_test
            
            # 14. Save processed data and transformers if requested
            if save_processed_data:
                self.save_processed_data(result)
            
            # Include metadata in result
            result['feature_names'] = list(train_df_norm.columns)
            result['sequence_length'] = self.sequence_length
            result['prediction_horizon'] = self.prediction_horizon
            result['feature_scaler_type'] = self.feature_scaler_type
            result['label_scaler_type'] = self.label_scaler_type
            result['target_column'] = self.target_column
            
            logger.info("Data preparation pipeline completed successfully")
            
            return result
            
        except Exception as e:
            logger.error(f"Error in data preparation pipeline: {e}")
            raise
    
    def save_processed_data(self, data_dict: Dict[str, Any], path: Optional[str] = None) -> None:
        """
        Save processed data and transformers to disk.
        
        Args:
            data_dict: Dictionary containing processed data and metadata
            path: Optional custom path to save to
        """
        logger.info("Saving processed data and transformers")
        
        # Use provided path or default
        save_path = Path(path) if path else self.output_dir
        save_path.mkdir(parents=True, exist_ok=True)
        
        try:
            # Save numpy arrays
            for key in ['X_train', 'y_train', 'X_val', 'y_val', 'X_test', 'y_test']:
                if key in data_dict:
                    np.save(save_path / f"{key}.npy", data_dict[key])
            
            # Save DataFrames
            for key in ['train_df', 'val_df', 'test_df']:
                if key in data_dict and isinstance(data_dict[key], pd.DataFrame):
                    data_dict[key].to_csv(save_path / f"{key}.csv")
            
            # Save feature scalers
            if self.feature_scalers:
                joblib.dump(self.feature_scalers, save_path / "feature_scalers.joblib")
            
            # Save label scalers
            if self.label_scalers:
                joblib.dump(self.label_scalers, save_path / "label_scalers.joblib")
            
            # Save imputers
            if self.imputers:
                joblib.dump(self.imputers, save_path / "imputers.joblib")
            
            # Save PCA transformers
            if self.pca_transformers:
                joblib.dump(self.pca_transformers, save_path / "pca_transformers.joblib")
            
            # Save feature selectors
            if self.feature_selectors:
                joblib.dump(self.feature_selectors, save_path / "feature_selectors.joblib")
            
            # Save outlier detectors
            if self.outlier_detectors:
                joblib.dump(self.outlier_detectors, save_path / "outlier_detectors.joblib")
            
            # Save configuration
            with open(save_path / "config.json", 'w') as f:
                json.dump(self.config, f, indent=4)
            
            # Save metadata
            metadata = {
                'feature_names': data_dict.get('feature_names', []),
                'sequence_length': self.sequence_length,
                'prediction_horizon': self.prediction_horizon,
                'feature_scaler_type': self.feature_scaler_type,
                'label_scaler_type': self.label_scaler_type,
                'target_column': self.target_column,
                'data_shapes': {
                    key: data_dict[key].shape for key in ['X_train', 'y_train', 'X_val', 'y_val', 'X_test', 'y_test']
                    if key in data_dict
                },
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'version': self.PREPROCESSOR_VERSION
            }
            
            with open(save_path / "metadata.json", 'w') as f:
                json.dump(metadata, f, indent=4)
            
            logger.info(f"Processed data and transformers saved to {save_path}")
            
        except Exception as e:
            logger.error(f"Error saving processed data: {e}")
            raise
    
    def load_processed_data(self, path: Optional[str] = None) -> Dict[str, Any]:
        """
        Load previously processed data and transformers from disk.
        
        Args:
            path: Optional custom path to load from
            
        Returns:
            Dictionary containing loaded data and metadata
        """
        logger.info("Loading processed data and transformers")
        
        # Use provided path or default
        load_path = Path(path) if path else self.output_dir
        
        if not load_path.exists():
            logger.error(f"Load path does not exist: {load_path}")
            raise FileNotFoundError(f"Load path does not exist: {load_path}")
        
        result = {}
        
        try:
            # Load numpy arrays
            for key in ['X_train', 'y_train', 'X_val', 'y_val', 'X_test', 'y_test']:
                file_path = load_path / f"{key}.npy"
                if file_path.exists():
                    result[key] = np.load(file_path, allow_pickle=True)
            
            # Load DataFrames
            for key in ['train_df', 'val_df', 'test_df']:
                file_path = load_path / f"{key}.csv"
                if file_path.exists():
                    result[key] = pd.read_csv(file_path, index_col=0)
            
            # Load feature scalers
            scaler_path = load_path / "feature_scalers.joblib"
            if scaler_path.exists():
                self.feature_scalers = joblib.load(scaler_path)
            
            # Load label scalers
            scaler_path = load_path / "label_scalers.joblib"
            if scaler_path.exists():
                self.label_scalers = joblib.load(scaler_path)
            
            # Load imputers
            imputer_path = load_path / "imputers.joblib"
            if imputer_path.exists():
                self.imputers = joblib.load(imputer_path)
            
            # Load PCA transformers
            pca_path = load_path / "pca_transformers.joblib"
            if pca_path.exists():
                self.pca_transformers = joblib.load(pca_path)
            
            # Load feature selectors
            selector_path = load_path / "feature_selectors.joblib"
            if selector_path.exists():
                self.feature_selectors = joblib.load(selector_path)
            
            # Load outlier detectors
            outlier_path = load_path / "outlier_detectors.joblib"
            if outlier_path.exists():
                self.outlier_detectors = joblib.load(outlier_path)
            
            # Load configuration
            config_path = load_path / "config.json"
            if config_path.exists():
                with open(config_path, 'r') as f:
                    self.config = json.load(f)
                    
                    # Update instance variables from config
                    self.sequence_length = self.config.get("sequence_length", self.sequence_length)
                    self.prediction_horizon = self.config.get("prediction_horizon", self.prediction_horizon)
                    self.feature_scaler_type = self.config.get("feature_scaler", self.feature_scaler_type)
                    self.label_scaler_type = self.config.get("label_scaler", self.label_scaler_type)
                    self.target_column = self.config.get("target_column", self.target_column)
            
            # Load metadata
            metadata_path = load_path / "metadata.json"
            if metadata_path.exists():
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                    result['metadata'] = metadata
                    
                    # Extract feature names
                    if 'feature_names' in metadata:
                        result['feature_names'] = metadata['feature_names']
            
            logger.info(f"Processed data and transformers loaded from {load_path}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error loading processed data: {e}")
            raise
    
    def inverse_transform_predictions(self, predictions: np.ndarray) -> np.ndarray:
        """
        Inverse transform scaled predictions back to original scale.
        
        Args:
            predictions: Scaled model predictions
            
        Returns:
            Predictions in original scale
        """
        logger.info("Inverse transforming predictions")
        
        try:
            # Check if label scalers exist
            if 'target' not in self.label_scalers:
                logger.warning("No label scaler found, returning predictions as is")
                return predictions
            
            # Reshape predictions for inverse transform if needed
            if len(predictions.shape) == 1:
                predictions = predictions.reshape(-1, 1)
            
            # Apply inverse transform
            original_predictions = self.label_scalers['target'].inverse_transform(predictions)
            
            # Convert back to original shape
            if original_predictions.shape[1] == 1:
                original_predictions = original_predictions.ravel()
            
            return original_predictions
            
        except Exception as e:
            logger.error(f"Error inverse transforming predictions: {e}")
            # Return original predictions if error
            return predictions
    
    def get_feature_names(self) -> List[str]:
        """
        Get the names of features in the processed data.
        
        Returns:
            List of feature names
        """
        if self.features_to_include:
            return self.features_to_include
        
        # Try to get from instance variables
        if hasattr(self, 'feature_names') and self.feature_names:
            return self.feature_names
        
        # Default feature names if nothing else available
        return ['feature_{}'.format(i+1) for i in range(self.n_features)]
    
    def __str__(self) -> str:
        """
        Return string representation of the preprocessor.
        
        Returns:
            String with preprocessor information
        """
        info = [
            f"AriadnePreprocessor (Version: {self.PREPROCESSOR_VERSION})",
            f"Sequence Length: {self.sequence_length}",
            f"Prediction Horizon: {self.prediction_horizon}",
            f"Feature Scaler: {self.feature_scaler_type}",
            f"Label Scaler: {self.label_scaler_type}",
            f"Target Column: {self.target_column}",
            f"Output Directory: {self.output_dir}",
        ]
        
        return "\n".join(info)
    
    def __repr__(self) -> str:
        """
        Return string representation of the preprocessor.
        
        Returns:
            String with preprocessor information
        """
        return self.__str__()


if __name__ == "__main__":
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser(description='Ariadne Preprocessor CLI')
    parser.add_argument('--config', type=str, help='Path to configuration file')
    parser.add_argument('--data', type=str, help='Path to data file')
    parser.add_argument('--output', type=str, help='Output directory')
    parser.add_argument('--mode', type=str, choices=['process', 'load'], help='Operation mode')
    
    args = parser.parse_args()
    
    if args.config and args.data and args.mode == 'process':
        preprocessor = AriadnePreprocessor(config_path=args.config, output_dir=args.output)
        
        # Process data
        result = preprocessor.prepare_data_pipeline(
            data_path=args.data,
            save_processed_data=True
        )
        
        print(f"Data processed and saved to {args.output}")
        print(f"X_train shape: {result['X_train'].shape}")
        print(f"y_train shape: {result['y_train'].shape}")
    
    elif args.output and args.mode == 'load':
        preprocessor = AriadnePreprocessor(output_dir=args.output)
        
        # Load processed data
        result = preprocessor.load_processed_data()
        
        print(f"Data loaded from {args.output}")
        print(f"X_train shape: {result['X_train'].shape}")
        print(f"y_train shape: {result['y_train'].shape}")
    
    else:
        print("Usage: python preprocessor.py --config CONFIG_PATH --data DATA_PATH --output OUTPUT_DIR --mode process")
        print("   or: python preprocessor.py --output OUTPUT_DIR --mode load")
                # Most frequent value imputation for categorical columns
                imputer = SimpleImputer(strategy='most_frequent')
                self.imputers['categorical'] = imputer
                df_imputed[categorical_cols] = imputer.fit_transform(df_imputed[categorical_cols])
                
            except Exception as e:
                logger.error(f"Error during categorical imputation: {e}")
                # Fallback to simpler imputation
                for col in categorical_cols:
                    df_imputed[col] = df_imputed[col].fillna(df_imputed[col].mode()[0])
        
        # Check if any missing values remain
        remaining_missing = df_imputed.isnull().sum().sum()
        if remaining_missing > 0:
            logger.warning(f"There are still {remaining_missing} missing values after imputation")
            
            # Last resort: drop rows with any remaining missing values
            df_imputed = df_imputed.dropna()
            logger.warning(f"Dropped rows with missing values, {len(df_imputed)} rows remaining")
        
        logger.info("Missing value handling completed")
        
        return df_imputed
    
    def detect_and_handle_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Detect and handle outliers in the data.
        
        Args:
            df: DataFrame with potentially outlier values
            
        Returns:
            DataFrame with outliers handled
        """
        logger.info("Detecting and handling outliers")
        
        # Make a copy to avoid modifying the original DataFrame
        df_cleaned = df.copy()
        
        # Get outlier detection method from config
        outlier_method = self.config.get("outlier_method", "zscore")
        
        # Get numeric columns
        numeric_cols = df_cleaned.select_dtypes(include=['int', 'float']).columns
        
        try:
            if outlier_method == "zscore":
                # Z-score method
                z_threshold = self.config.get("zscore_threshold", 3.0)
                
                for col in numeric_cols:
                    # Skip columns that are likely to be categorical or binary
                    if df_cleaned[col].nunique() < 5:
                        continue
                    
                    # Calculate z-scores
                    z_scores = np.abs((df_cleaned[col] - df_cleaned[col].mean()) / df_cleaned[col].std())
                    
                    # Identify outliers
                    outliers = z_scores > z_threshold
                    outlier_count = outliers.sum()
                    
                    if outlier_count > 0:
                        logger.info(f"Found {outlier_count} outliers in column {col} using Z-score method")
                        
                        # Replace outliers with column median
                        df_cleaned.loc[outliers, col] = df_cleaned[col].median()
            
            elif outlier_method == "iqr":
                # Interquartile Range method
                iqr_multiplier = self.config.get("iqr_multiplier", 1.5)
                
                for col in numeric_cols:
                    # Skip columns that are likely to be categorical or binary
                    if df_cleaned[col].nunique() < 5:
                        continue
                    
                    # Calculate IQR boundaries
                    q1 = df_cleaned[col].quantile(0.25)
                    q3 = df_cleaned[col].quantile(0.75)
                    iqr = q3 - q1
                    
                    lower_bound = q1 - (iqr_multiplier * iqr)
                    upper_bound = q3 + (iqr_multiplier * iqr)
                    
                    # Identify outliers
                    outliers = (df_cleaned[col] < lower_bound) | (df_cleaned[col] > upper_bound)
                    outlier_count = outliers.sum()
                    
                    if outlier_count > 0:
                        logger.info(f"Found {outlier_count} outliers in column {col} using IQR method")
                        
                        # Replace outliers with column median
                        df_cleaned.loc[outliers, col] = df_cleaned[col].median()
            
            elif outlier_method == "isolation_forest":
                # Isolation Forest method
                contamination = self.config.get("anomaly_contamination", 0.05)
                
                # Initialize Isolation Forest
                isolation_forest = IsolationForest(
                    contamination=contamination,
                    random_state=self.random_seed
                )
                
                # Fit the model on numeric columns
                isolation_forest.fit(df_cleaned[numeric_cols])
                
                # Predict outliers
                outlier_predictions = isolation_forest.predict(df_cleaned[numeric_cols])
                outliers = outlier_predictions == -1
                outlier_count = np.sum(outliers)
                
                if outlier_count > 0:
                    logger.info(f"Found {outlier_count} outlier rows using Isolation Forest method")
                    
                    # Store the model for later use
                    self.outlier_detectors['isolation_forest'] = isolation_forest
                    
                    # For each outlier row, replace values with median of non-outlier rows
                    for col in numeric_cols:
                        non_outlier_median = df_cleaned.loc[~outliers, col].median()
                        df_cleaned.loc[outliers, col] = non_outlier_median
            
            elif outlier_method == "winsorize":
                # Winsorization method (cap at percentiles)
                lower_percentile = self.config.get("winsorize_lower", 0.05)
                upper_percentile = self.config.get("winsorize_upper", 0.95)
                
                for col in numeric_cols:
                    # Skip columns that are likely to be categorical or binary
                    if df_cleaned[col].nunique() < 5:
                        continue
                    
                    # Calculate percentile values
                    lower_bound = df_cleaned[col].quantile(lower_percentile)
                    upper_bound = df_cleaned[col].quantile(upper_percentile)
                    
                    # Identify outliers
                    lower_outliers = df_cleaned[col] < lower_bound
                    upper_outliers = df_cleaned[col] > upper_bound
                    
                    outlier_count = lower_outliers.sum() + upper_outliers.sum()
                    
                    if outlier_count > 0:
                        logger.info(f"Found {outlier_count} outliers in column {col} using Winsorization method")
                        
                        # Cap values at percentile boundaries
                        df_cleaned.loc[lower_outliers, col] = lower_bound
                        df_cleaned.loc[upper_outliers, col] = upper_bound
            
            else:
                logger.warning(f"Unknown outlier detection method: {outlier_method}")
        
        except Exception as e:
            logger.error(f"Error during outlier detection: {e}")
        
        logger.info("Outlier detection and handling completed")
        
        return df_cleaned
    
    def normalize_features(self, df: pd.DataFrame, is_train: bool = True) -> pd.DataFrame:
        """
        Normalize features to improve model training.
        
        Args:
            df: DataFrame with features to normalize
            is_train: Whether this is training data (to fit scalers) or not
            
        Returns:
            DataFrame with normalized features
        """
        logger.info("Normalizing features")
        
        # Make a copy to avoid modifying the original DataFrame
        df_norm = df.copy()
        
        # Get numeric columns
        numeric_cols = df_norm.select_dtypes(include=['int', 'float']).columns.tolist()
        
        # If target_column is in numeric_cols, remove it (will handle separately)
        if self.target_column in numeric_cols:
            numeric_cols.remove(self.target_column)
        
        # Get feature scaler type from config
        feature_scaler_type = self.feature_scaler_type
        
        try:
            # Feature scaling
            if len(numeric_cols) > 0:
                # Get correct scaler
                if feature_scaler_type == "standard":
                    if is_train or 'feature' not in self.feature_scalers:
                        self.feature_scalers['feature'] = StandardScaler()
                
                elif feature_scaler_type == "minmax":
                    if is_train or 'feature' not in self.feature_scalers:
                        self.feature_scalers['feature'] = MinMaxScaler()
                
                elif feature_scaler_type == "robust":
                    if is_train or 'feature' not in self.feature_scalers:
                        self.feature_scalers['feature'] = RobustScaler()
                
                elif feature_scaler_type == "quantile":
                    if is_train or 'feature' not in self.feature_scalers:
                        self.feature_scalers['feature'] = QuantileTransformer(output_distribution='normal')
                
                elif feature_scaler_type == "timeseries":
                    if is_train or 'feature' not in self.feature_scalers:
                        self.feature_scalers['feature'] = TimeSeriesScalerMeanVariance()
                
                else:
                    logger.warning(f"Unknown feature scaler type: {feature_scaler_type}, using StandardScaler")
                    if is_train or 'feature' not in self.feature_scalers:
                        self.feature_scalers['feature'] = StandardScaler()
                
                # Apply scaling
                if feature_scaler_type != "timeseries":
                    if is_train:
                        df_norm[numeric_cols] = self.feature_scalers['feature'].fit_transform(df_norm[numeric_cols])
                    else:
                        df_norm[numeric_cols] = self.feature_scalers['feature'].transform(df_norm[numeric_cols])
                else:
                    # For time series scaler, need to handle differently
                    # Convert to numpy array with shape (n_samples, n_timesteps, n_features)
                    # For simplicity here, treat each row as a separate time series
                    data = df_norm[numeric_cols].values.reshape(-1, 1, len(numeric_cols))
                    
                    if is_train:
                        data_scaled = self.feature_scalers['feature'].fit_transform(data)
                    else:
                        data_scaled = self.feature_scalers['feature'].transform(data)
                    
                    # Convert back to DataFrame format
                    df_norm[numeric_cols] = data_scaled.reshape(-1, len(numeric_cols))
            
            # Target scaling (if regression)
            if self.target_column in df_norm.columns:
                label_scaler_type = self.label_scaler_type
                
                # Get correct scaler for target
                if label_scaler_type == "standard":
                    if is_train or 'target' not in self.label_scalers:
                        self.label_scalers['target'] = StandardScaler()
                
                elif label_scaler_type == "minmax":
                    if is_train or 'target' not in self.label_scalers:
                        self.label_scalers['target'] = MinMaxScaler()
                
                elif label_scaler_type == "robust":
                    if is_train or 'target' not in self.label_scalers:
                        self.label_scalers['target'] = RobustScaler()
                
                elif label_scaler_type == "quantile":
                    if is_train or 'target' not in self.label_scalers:
                        self.label_scalers['target'] = QuantileTransformer(output_distribution='normal')
                
                else:
                    logger.warning(f"Unknown label scaler type: {label_scaler_type}, using StandardScaler")
                    if is_train or 'target' not in self.label_scalers:
                        self.label_scalers['target'] = StandardScaler()
                
                # Reshape for scaler
                target_values = df_norm[self.target_column].values.reshape(-1, 1)
                
                # Apply scaling to target
                if is_train:
                    df_norm[self.target_column] = self.label_scalers['target'].fit_transform(target_values)
                else:
                    df_norm[self.target_column] = self.label_scalers['target'].transform(target_values)
        
        except Exception as e:
            logger.error(f"Error during normalization: {e}")
            # If error occurs, return original DataFrame
            return df
        
        logger.info("Feature normalization completed")
        
        return df_norm
    
    def apply_dimension_reduction(self, df: pd.DataFrame, is_train: bool = True) -> pd.DataFrame:
        """
        Apply dimensionality reduction to features.
        
        Args:
            df: DataFrame with features
            is_train: Whether this is training data (to fit transformers) or not
            
        Returns:
            DataFrame with reduced dimensions
        """
        # Check if dimension reduction is enabled
        if not self.config.get("apply_dimension_reduction", False):
            return df
        
        logger.info("Applying dimension reduction")
        
        # Make a copy to avoid modifying the original DataFrame
        df_reduced = df.copy()
        
        # Get numeric columns
        numeric_cols = df_reduced.select_dtypes(include=['int', 'float']).columns.tolist()
        
        # If target_column is in numeric_cols, remove it
        if self.target_column in numeric_cols:
            numeric_cols.remove(self.target_column)
        
        # Get dimension reduction method from config
        reduction_method = self.config.get("dimension_reduction_method", "pca")
        
        try:
            if reduction_method == "pca":
                # PCA dimension reduction
                n_components = self.config.get("pca_components", 0.95)
                
                if is_train or 'pca' not in self.pca_transformers:
                    pca = PCA(n_components=n_components, random_state=self.random_seed)
                    self.pca_transformers['pca'] = pca
                
                # Apply PCA
                if is_train:
                    pca_result = self.pca_transformers['pca'].fit_transform(df_reduced[numeric_cols])
                else:
                    pca_result = self.pca_transformers['pca'].transform(df_reduced[numeric_cols])
                
                # Create new DataFrame with PCA components
                pca_cols = [f'pca_{i+1}' for i in range(pca_result.shape[1])]
                pca_df = pd.DataFrame(pca_result, columns=pca_cols, index=df_reduced.index)
                
                # Drop original numeric columns and add PCA components
                df_reduced = df_reduced.drop(columns=numeric_cols)
                df_reduced = pd.concat([df_reduced, pca_df], axis=1)
                
                # Keep target column if it exists
                if self.target_column in df.columns and self.target_column not in df_reduced.columns:
                    df_reduced[self.target_column] = df[self.target_column].copy()
                
                logger.info(f"Reduced {len(numeric_cols)} dimensions to {len(pca_cols)} PCA components")
            
            elif reduction_method == "feature_selection":
                # Feature selection using various methods
                selection_method = self.config.get("feature_selection_method", "mutual_info")
                k_features = self.config.get("k_features", 10)
                
                if is_train or 'selector' not in self.feature_selectors:
                    if selection_method == "mutual_info":
                        selector = SelectKBest(mutual_info_regression, k=k_features)
                    elif selection_method == "f_regression":
                        selector = SelectKBest(f_regression, k=k_features)
                    else:
                        logger.warning(f"Unknown feature selection method: {selection_method}, using mutual_info")
                        selector = SelectKBest(mutual_info_regression, k=k_features)
                    
                    self.feature_selectors['selector'] = selector
                
                # Make sure target column exists for supervised feature selection
                if self.target_column not in df.columns:
                    logger.error("Target column required for supervised feature selection")
                    return df
                
                # Apply feature selection
                if is_train:
                    # Fit and transform
                    selected_features = self.feature_selectors['selector'].fit_transform(
                        df_reduced[numeric_cols], df_reduced[self.target_column]
                    )
                    
                    # Get selected feature names
                    selected_indices = self.feature_selectors['selector'].get_support(indices=True)
                    selected_names = [numeric_cols[i] for i in selected_indices]
                    
                    # Store selected feature names
                    self.feature_selectors['selected_features'] = selected_names
                    
                    logger.info(f"Selected {len(selected_names)} features: {selected_names}")
                    
                    # Keep only selected features plus non-numeric columns
                    non_numeric_cols = [col for col in df_reduced.columns if col not in numeric_cols]
                    df_reduced = df_reduced[non_numeric_cols + selected_names]
                
                else:
                    # Transform using previously selected features
                    if 'selected_features' in self.feature_selectors:
                        selected_names = self.feature_selectors['selected_features']
                        
                        # Keep only selected features plus non-numeric columns
                        non_numeric_cols = [col for col in df_reduced.columns if col not in numeric_cols]
                        df_reduced = df_reduced[non_numeric_cols + selected_names]
                    else:
                        logger.error("No selected features found for transform. Run fit first.")
            
            else:
                logger.warning(f"Unknown dimension reduction method: {reduction_method}")
                # Return original DataFrame if method not recognized
                return df
        
        except Exception as e:
            logger.error(f"Error during dimension reduction: {e}")
            # If error occurs, return original DataFrame
            return df
        
        logger.info("Dimension reduction completed")
        
        return df_reduced
    
    def create_sequences(
        self,
        df: pd.DataFrame,
        sequence_length: Optional[int] = None,
        prediction_horizon: Optional[int] = None,
        step_size: int = 1,
        target_column: Optional[str] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sequences for time series prediction.
        
        Args:
            df: DataFrame with time series data
            sequence_length: Number of time steps in each sequence
            prediction_horizon: Number of time steps to predict into the future
            step_size: Step size between sequences (for overlapping sequences)
            target_column: Column to use as prediction target
            
        Returns:
            Tuple of (X, y) where X is input sequences and y is target values
        """
        logger.info("Creating sequences for time series prediction")
        
        # Use parameters from config if not specified
        sequence_length = sequence_length or self.sequence_length
        prediction_horizon = prediction_horizon or self.prediction_horizon
        target_column = target_column or self.target_column
        
        # Check if target column exists
        if target_column not in df.columns:
            logger.error(f"Target column '{target_column}' not found in DataFrame")
            raise ValueError(f"Target column '{target_column}' not found in DataFrame")
        
        # Prepare data
        data = df.values
        X, y = [], []
        
        # Create sequences
        for i in range(0, len(data) - sequence_length - prediction_horizon + 1, step_size):
            X.append(data[i:i+sequence_length])
            
            # For regression, use the actual value at the future time step
            if prediction_horizon == 1:
                y.append(data[i+sequence_length, df.columns.get_loc(target_column)])
            else:
                # For multi-step prediction, extract target values for the entire horizon
                target_idx = df.columns.get_loc(target_column)
                y.append(data[i+sequence_length:i+sequence_length+prediction_horizon, target_idx])
        
        # Convert to numpy arrays
        X = np.array(X)
        y = np.array(y)
        
        logger.info(f"Created {len(X)} sequences with shape {X.shape}")
        
        return X, y
    
    def create_classification_labels(
        self,
        df: pd.DataFrame,
        target_column: Optional[str] = None,
        threshold: float = 0.0,
        horizon: Optional[int] = None,
        label_type: str = 'binary'
    ) -> np.ndarray:
        """
        Create classification labels for price movement prediction.
        
        Args:
            df: DataFrame with time series data
            target_column: Column to use for creating labels
            threshold: Minimum required return to be considered positive (for binary classification)
            horizon: Number of time steps to look ahead
            label_type: Type of labels ('binary', 'ternary', or 'multi')
            
        Returns:
            Array of classification labels
        """
        logger.info("Creating classification labels")
        
        # Use parameters from config if not specified
        target_column = target_column or self.target_column
        horizon = horizon or self.prediction_horizon
        
        # Check if target column exists
        if target_column not in df.columns:
            logger.error(f"Target column '{target_column}' not found in DataFrame")
            raise ValueError(f"Target column '{target_column}' not found in DataFrame")
        
        # Calculate future returns
        returns = df[target_column].pct_change(periods=horizon).shift(-horizon)
        
        # Create labels based on label_type
        if label_type == 'binary':
            # Binary classification: 1 for positive returns above threshold, 0 otherwise
            labels = (returns > threshold).astype(int)
            logger.info(f"Created binary labels with {labels.sum()} positive class samples")
        
        elif label_type == 'ternary':
            # Ternary classification: 1 for positive returns, 0 for neutral, -1 for negative
            lower_threshold = -threshold if threshold > 0 else -0.005
            upper_threshold = threshold if threshold > 0 else 0.005
            
            labels = pd.cut(
                returns,
                bins=[float('-inf'), lower_threshold, upper_threshold, float('inf')],
                labels=[-1, 0, 1]
            ).astype(int)
            
            # Convert to 0, 1, 2 for model compatibility
            labels = labels + 1
            
            logger.info(f"Created ternary labels with class distribution: {labels.value_counts().to_dict()}")
        
        elif label_type == 'multi':
            # Multi-class classification: Discretize returns into bins
            bins = self.config.get("multi_class_bins", 5)
            
            if isinstance(bins, int):
                # Create equal-frequency bins
                labels = pd.qcut(
                    returns,
                    q=bins,
                    labels=False,
                    duplicates='drop'
                )
            else:
                # Use specified bin edges
                labels = pd.cut(
                    returns,
                    bins=bins,
                    labels=False
                )
            
            logger.info(f"Created multi-class labels with class distribution: {labels.value_counts().to_dict()}")
        
        else:
            logger.error(f"Unknown label type: {label_type}")
            raise ValueError(f"Unknown label type: {label_type}")
        
        # Handle NaN values (at the end of the series)
        labels = labels.fillna(-999).astype(int)
        
        # Remove rows with NaN labels (marked as -999)
        valid_indices = labels != -999
        labels = labels[valid_indices].values
        
        return labels, valid_indices
    
    def train_val_test_split(
        self,
        df: pd.DataFrame,
        split_method: str = 'time',
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        purge_overlap: bool = True
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split data into training, validation, and test sets.
        
        Args:
            df: DataFrame to split
            split_method: Method to use for splitting ('time', 'random', or 'grouped')
            train_ratio: Ratio of data to use for training
            val_ratio: Ratio of data to use for validation
            test_ratio: Ratio of data to use for testing
            purge_overlap: Whether to remove overlapping sequences from different sets
            
        Returns:
            Tuple of (train_df, val_df, test_df)
        """
        logger.info(f"Splitting data using {split_method} method")
        
        # Check if ratios sum to 1
        if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
            logger.warning("Split ratios do not sum to 1.0, normalizing")
            total = train_ratio + val_ratio + test_ratio
            train_ratio /= total
            val_ratio /= total
            test_ratio /= total
        
        # Use parameters from config if not specified
        if not all([train_ratio, val_ratio, test_ratio]):
            splits = self.train_val_test_split
            train_ratio, val_ratio, test_ratio = splits
        
        # Make a copy of the DataFrame
        df_copy = df.copy()
        
        if split_method == 'time':
            # Time-based splitting (most appropriate for time series)
            n = len(df_copy)
            train_idx = int(n * train_ratio)
            val_idx = train_idx + int(n * val_ratio)
            
            train_df = df_copy.iloc[:train_idx].copy()
            val_df = df_copy.iloc[train_idx:val_idx].copy()
            test_df = df_copy.iloc[val_idx:].copy()
            
            logger.info(f"Time-based split: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")
            
            # Handle overlap purging if enabled
            if purge_overlap and self.sequence_length > 0:
                # Remove last `sequence_length` rows from train to avoid overlap with val
                if len(train_df) > self.sequence_length:
                    train_df = train_df.iloc[:-self.sequence_length].copy()
                
                # Remove last `sequence_length` rows from val to avoid overlap with test
                if len(val_df) > self.sequence_length:
                    val_df = val_df.iloc[:-self.sequence_length].copy()
                
                logger.info(f"After purging overlap: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")
        
        elif split_method == 'random':
            # Random splitting (not ideal for time series, but included for completeness)
            from sklearn.model_selection import train_test_split
            
            # First split train and test
            train_val_df, test_df = train_test_split(
                df_copy,
                test_size=test_ratio,
                random_state=self.random_seed
            )
            
            # Then split train and validation
            val_size = val_ratio / (train_ratio + val_ratio)
            train_df, val_df = train_test_split(
                train_val_df,
                test_size=val_size,
                random_state=self.random_seed
            )
            
            logger.info(f"Random split: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")
        
        elif split_method == 'grouped':
            # Grouped splitting (for time series with grouping variable)
            group_col = self.config.get("group_column")
            
            if group_col not in df_copy.columns:
                logger.error(f"Group column '{group_col}' not found, falling back to time-based split")
                return self.train_val_test_split(
                    df, 'time', train_ratio, val_ratio, test_ratio, purge_overlap
                )
            
            # Get unique groups
            groups = df_copy[group_col].unique()
            np.random.seed(self.random_seed)
            np.random.shuffle(groups)
            
            # Split groups
            n_groups = len(groups)
            train_groups = groups[:int(n_groups * train_ratio)]
            val_groups = groups[int(n_groups * train_ratio):int(n_groups * (train_ratio + val_ratio))]
            test_groups = groups[int(n_groups * (train_ratio + val_ratio)):]
            
            # Split data based on groups
            train_df = df_copy[df_copy[group_col].isin(train_groups)].copy()
            val_df = df_copy[df_copy[group_col].isin(val_groups)].copy()
            test_df = df_copy[df_copy[group_col].isin(test_groups)].copy()
            
            logger.info(f"Grouped split: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")
        
        else:
            logger.error(f"Unknown split method: {split_method}, falling back to time-based split")
            return self.train_val_test_split(
                df, 'time', train_ratio, val_ratio, test_ratio, purge_overlap
            )
        
        return train_df, val_df, test_df
    
    def apply_data_augmentation(
        self,
        X: np.ndarray,
        y: np.ndarray,
        augmentation_factor: float = 1.0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply data augmentation techniques specific to financial time series.
        
        Args:
            X: Input sequences
            y: Target values
            augmentation_factor: Factor by which to increase dataset size
            
        Returns:
            Tuple of (augmented_X, augmented_y)
        """
        # Check if data augmentation is enabled
        if not self.config.get("apply_data_augmentation", False):
            return X, y
        
        logger.info("Applying data augmentation")
        
        # Check if augmentation_factor is valid
        if augmentation_factor <= 0:
            logger.error("Invalid augmentation factor, must be positive")
            return X, y
        
        # No augmentation needed if factor is 1
        if abs(augmentation_factor - 1.0) < 1e-6:
            return X, y
        
        # Set random seed for reproducibility
        np.random.seed(self.random_seed)
        
        # Make copies to avoid modifying the original arrays
        X_orig = X.copy()
        y_orig = y.copy()
        
        # Get augmentation methods from config
        augmentation_methods = self.config.get("augmentation_methods", ["jitter", "scaling", "permutation"])
        
        # Calculate number of augmented samples to generate
        n_samples = int(len(X) * augmentation_factor) - len(X)
        
        if n_samples <= 0:                        # Volatility ratio (up/down)
                        df_with_features[f'vol_ratio_{window}'] = (
                            df_with_features[f'up_vol_{window}'] / (df_with_features[f'down_vol_{window}'] + np.finfo(float).eps)
                        )
                        
                        # Higher moment statistics
                        returns_centered = returns - returns.rolling(window=window).mean()
                        
                        # Skewness
                        skew = (
                            (returns_centered ** 3).rolling(window=window).mean() /
                            ((returns_centered ** 2).rolling(window=window).mean() ** 1.5 + np.finfo(float).eps)
                        )
                        df_with_features[f'skew_{window}'] = skew
                        
                        # Kurtosis
                        kurt = (
                            (returns_centered ** 4).rolling(window=window).mean() /
                            ((returns_centered ** 2).rolling(window=window).mean() ** 2 + np.finfo(float).eps)
                        ) - 3  # Excess kurtosis
                        df_with_features[f'kurt_{window}'] = kurt
                        
                        added_features.extend([
                            f'up_vol_{window}', f'down_vol_{window}', f'vol_ratio_{window}',
                            f'skew_{window}', f'kurt_{window}'
                        ])
                
                elif feature == "price_reversals":
                    # Price reversal indicators
                    
                    # Initialize columns
                    df_with_features['price_trend'] = np.sign(df_with_features[close_col].diff())
                    df_with_features['reversal'] = 0
                    
                    # Identify reversals (where price trend changes sign)
                    df_with_features.loc[df_with_features['price_trend'] != df_with_features['price_trend'].shift(), 'reversal'] = 1
                    
                    # Calculate various reversal metrics
                    for window in params.get("windows", [20]):
                        # Reversal frequency
                        df_with_features[f'reversal_freq_{window}'] = df_with_features['reversal'].rolling(window=window).mean()
                        
                        # Consecutive moves in same direction
                        consecutive_moves = df_with_features['price_trend'].groupby(
                            (df_with_features['price_trend'] != df_with_features['price_trend'].shift()).cumsum()
                        ).cumcount() + 1
                        
                        df_with_features[f'consec_moves_{window}'] = consecutive_moves.rolling(window=window).mean()
                        
                        added_features.extend([
                            'reversal', f'reversal_freq_{window}', f'consec_moves_{window}'
                        ])
                
                # Add more market microstructure features as needed
                
                else:
                    logger.warning(f"Unknown market microstructure feature: {feature}")
            
            except Exception as e:
                logger.error(f"Error calculating market microstructure feature {feature}: {e}")
        
        # Log the number of features added
        logger.info(f"Added {len(added_features)} market microstructure features")
        
        return df_with_features
    
    def add_on_chain_features(
        self,
        df: pd.DataFrame,
        on_chain_data: Optional[pd.DataFrame] = None,
        on_chain_columns: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Add on-chain blockchain metrics to the price data.
        
        Args:
            df: DataFrame with price data
            on_chain_data: Optional DataFrame with on-chain data
                         If None, looks for columns with prefix 'on_chain_' in df
            on_chain_columns: List of on-chain columns to include
                             If None, includes all on-chain columns
            
        Returns:
            DataFrame with added on-chain features
        """
        logger.info("Adding on-chain features")
        
        # Make a copy to avoid modifying the original DataFrame
        df_with_features = df.copy()
        
        # If on_chain_data is provided, join with price data
        if on_chain_data is not None:
            # Check if index is datetime
            if isinstance(on_chain_data.index, pd.DatetimeIndex) and isinstance(df.index, pd.DatetimeIndex):
                # Join the dataframes
                df_with_features = df.join(on_chain_data, how='left')
            else:
                logger.error("Both DataFrames must have DatetimeIndex to join")
                return df
        
        # Identify on-chain columns in the DataFrame
        on_chain_prefix = 'on_chain_'
        all_on_chain_cols = [col for col in df_with_features.columns if col.startswith(on_chain_prefix)]
        
        if not all_on_chain_cols:
            logger.warning("No on-chain columns found in the DataFrame")
            return df_with_features
        
        # Use specified columns or all on-chain columns
        on_chain_cols = on_chain_columns or all_on_chain_cols
        
        # Add basic on-chain derived features
        added_features = []
        
        try:
            # Create rate-of-change features for on-chain metrics
            for col in on_chain_cols:
                if col in df_with_features.columns:
                    # Daily change
                    df_with_features[f'{col}_change_1d'] = df_with_features[col].pct_change()
                    
                    # 7-day and 30-day change
                    df_with_features[f'{col}_change_7d'] = df_with_features[col].pct_change(periods=7)
                    df_with_features[f'{col}_change_30d'] = df_with_features[col].pct_change(periods=30)
                    
                    # Z-score (normalized value)
                    for window in [20, 50, 200]:
                        rolling_mean = df_with_features[col].rolling(window=window).mean()
                        rolling_std = df_with_features[col].rolling(window=window).std()
                        df_with_features[f'{col}_zscore_{window}'] = (
                            (df_with_features[col] - rolling_mean) / (rolling_std + np.finfo(float).eps)
                        )
                    
                    added_features.extend([
                        f'{col}_change_1d', f'{col}_change_7d', f'{col}_change_30d',
                        f'{col}_zscore_20', f'{col}_zscore_50', f'{col}_zscore_200'
                    ])
            
            # Specific on-chain metrics processing
            # Check for common on-chain metrics and add derived features
            
            # Active addresses features
            active_addr_col = next((col for col in on_chain_cols if 'active_address' in col), None)
            if active_addr_col and active_addr_col in df_with_features.columns:
                # Calculate active address momentum
                df_with_features['active_addr_momentum'] = df_with_features[active_addr_col].pct_change(periods=7) / df_with_features[active_addr_col].pct_change(periods=30)
                added_features.append('active_addr_momentum')
            
            # Transaction count features
            tx_count_col = next((col for col in on_chain_cols if 'transaction' in col and 'count' in col), None)
            if tx_count_col and tx_count_col in df_with_features.columns:
                # Transaction growth rate
                df_with_features['tx_growth_rate'] = df_with_features[tx_count_col].pct_change(periods=7)
                
                # Transaction count moving averages
                for window in [7, 30, 90]:
                    df_with_features[f'tx_count_sma_{window}'] = df_with_features[tx_count_col].rolling(window=window).mean()
                
                added_features.extend(['tx_growth_rate', 'tx_count_sma_7', 'tx_count_sma_30', 'tx_count_sma_90'])
            
            # Mining difficulty features
            difficulty_col = next((col for col in on_chain_cols if 'difficult' in col), None)
            if difficulty_col and difficulty_col in df_with_features.columns:
                # Difficulty growth rate
                df_with_features['difficulty_growth_rate'] = df_with_features[difficulty_col].pct_change(periods=14)
                added_features.append('difficulty_growth_rate')
            
            # Fee-related features
            fee_col = next((col for col in on_chain_cols if 'fee' in col), None)
            if fee_col and fee_col in df_with_features.columns:
                # Fee rate moving averages
                for window in [7, 30]:
                    df_with_features[f'fee_sma_{window}'] = df_with_features[fee_col].rolling(window=window).mean()
                
                # Fee volatility
                df_with_features['fee_volatility'] = df_with_features[fee_col].pct_change().rolling(window=30).std()
                
                added_features.extend(['fee_sma_7', 'fee_sma_30', 'fee_volatility'])
            
            # Network hash rate features
            hash_rate_col = next((col for col in on_chain_cols if 'hash' in col and 'rate' in col), None)
            if hash_rate_col and hash_rate_col in df_with_features.columns:
                # Hash rate growth
                df_with_features['hash_rate_growth'] = df_with_features[hash_rate_col].pct_change(periods=30)
                added_features.append('hash_rate_growth')
            
            # Exchange flow features
            inflow_col = next((col for col in on_chain_cols if 'exchange' in col and 'inflow' in col), None)
            outflow_col = next((col for col in on_chain_cols if 'exchange' in col and 'outflow' in col), None)
            
            if inflow_col and outflow_col and inflow_col in df_with_features.columns and outflow_col in df_with_features.columns:
                # Net flow (outflow - inflow)
                df_with_features['exchange_net_flow'] = df_with_features[outflow_col] - df_with_features[inflow_col]
                
                # Flow ratio (outflow / inflow)
                df_with_features['exchange_flow_ratio'] = df_with_features[outflow_col] / (df_with_features[inflow_col] + np.finfo(float).eps)
                
                added_features.extend(['exchange_net_flow', 'exchange_flow_ratio'])
            
            # UTXO age features
            utxo_col = next((col for col in on_chain_cols if 'utxo' in col), None)
            if utxo_col and utxo_col in df_with_features.columns:
                # UTXO change
                df_with_features['utxo_change'] = df_with_features[utxo_col].pct_change()
                added_features.append('utxo_change')
            
            # Supply-related features
            supply_col = next((col for col in on_chain_cols if 'supply' in col or 'circulation' in col), None)
            if supply_col and supply_col in df_with_features.columns:
                # Supply growth rate
                df_with_features['supply_growth_rate'] = df_with_features[supply_col].pct_change(periods=30) * 365  # Annualized
                added_features.append('supply_growth_rate')
            
            # Correlation between price and on-chain metrics
            price_col = self.target_column or 'close'
            for col in on_chain_cols:
                if col in df_with_features.columns:
                    # Calculate rolling correlation with price
                    for window in [30, 90]:
                        df_with_features[f'{col}_price_corr_{window}'] = (
                            df_with_features[price_col].rolling(window=window)
                                              .corr(df_with_features[col])
                        )
                        added_features.append(f'{col}_price_corr_{window}')
            
            logger.info(f"Added {len(added_features)} on-chain features")
            
        except Exception as e:
            logger.error(f"Error adding on-chain features: {e}")
        
        return df_with_features
    
    def add_sentiment_features(
        self,
        df: pd.DataFrame,
        sentiment_data: Optional[pd.DataFrame] = None,
        sentiment_columns: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Add sentiment metrics to the price data.
        
        Args:
            df: DataFrame with price data
            sentiment_data: Optional DataFrame with sentiment data
                          If None, looks for columns with prefix 'sentiment_' in df
            sentiment_columns: List of sentiment columns to include
                              If None, includes all sentiment columns
            
        Returns:
            DataFrame with added sentiment features
        """
        logger.info("Adding sentiment features")
        
        # Make a copy to avoid modifying the original DataFrame
        df_with_features = df.copy()
        
        # If sentiment_data is provided, join with price data
        if sentiment_data is not None:
            # Check if index is datetime
            if isinstance(sentiment_data.index, pd.DatetimeIndex) and isinstance(df.index, pd.DatetimeIndex):
                # Join the dataframes
                df_with_features = df.join(sentiment_data, how='left')
            else:
                logger.error("Both DataFrames must have DatetimeIndex to join")
                return df
        
        # Identify sentiment columns in the DataFrame
        sentiment_prefix = 'sentiment_'
        all_sentiment_cols = [col for col in df_with_features.columns if col.startswith(sentiment_prefix)]
        
        if not all_sentiment_cols:
            logger.warning("No sentiment columns found in the DataFrame")
            return df_with_features
        
        # Use specified columns or all sentiment columns
        sentiment_cols = sentiment_columns or all_sentiment_cols
        
        # Add derived sentiment features
        added_features = []
        
        try:
            # Process each sentiment column
            for col in sentiment_cols:
                if col in df_with_features.columns:
                    # Calculate moving averages
                    for window in [3, 7, 14, 30]:
                        df_with_features[f'{col}_sma_{window}'] = df_with_features[col].rolling(window=window).mean()
                        added_features.append(f'{col}_sma_{window}')
                    
                    # Calculate rate of change
                    for period in [1, 7, 14]:
                        df_with_features[f'{col}_change_{period}d'] = df_with_features[col].pct_change(periods=period)
                        added_features.append(f'{col}_change_{period}d')
                    
                    # Calculate sentiment momentum
                    df_with_features[f'{col}_momentum'] = df_with_features[col].diff(periods=3) / 3
                    added_features.append(f'{col}_momentum')
                    
                    # Calculate sentiment volatility
                    df_with_features[f'{col}_volatility'] = df_with_features[col].rolling(window=14).std()
                    added_features.append(f'{col}_volatility')
                    
                    # Calculate sentiment z-score
                    for window in [30, 90]:
                        rolling_mean = df_with_features[col].rolling(window=window).mean()
                        rolling_std = df_with_features[col].rolling(window=window).std()
                        df_with_features[f'{col}_zscore_{window}'] = (
                            (df_with_features[col] - rolling_mean) / (rolling_std + np.finfo(float).eps)
                        )
                        added_features.append(f'{col}_zscore_{window}')
            
            # Calculate sentiment divergence with price
            price_col = self.target_column or 'close'
            for col in sentiment_cols:
                if col in df_with_features.columns:
                    # Normalize both series
                    price_norm = (df_with_features[price_col] - df_with_features[price_col].rolling(window=30).mean()) / df_with_features[price_col].rolling(window=30).std()
                    sentiment_norm = (df_with_features[col] - df_with_features[col].rolling(window=30).mean()) / df_with_features[col].rolling(window=30).std()
                    
                    # Calculate divergence
                    df_with_features[f'{col}_divergence'] = price_norm - sentiment_norm
                    added_features.append(f'{col}_divergence')
                    
                    # Calculate correlation with price
                    for window in [14, 30, 60]:
                        df_with_features[f'{col}_price_corr_{window}'] = (
                            df_with_features[price_col].rolling(window=window)
                                              .corr(df_with_features[col])
                        )
                        added_features.append(f'{col}_price_corr_{window}')
            
            # Check for specific sentiment metrics and add custom features
            
            # Social volume features
            social_volume_col = next((col for col in sentiment_cols if 'volume' in col and col in df_with_features.columns), None)
            if social_volume_col:
                # Social volume ratio to price
                df_with_features['social_volume_price_ratio'] = df_with_features[social_volume_col] / df_with_features[price_col]
                
                # Normalized social volume
                df_with_features['social_volume_norm'] = df_with_features[social_volume_col] / df_with_features[social_volume_col].rolling(window=30).mean()
                
                added_features.extend(['social_volume_price_ratio', 'social_volume_norm'])
            
            # Sentiment score features
            sentiment_score_col = next((col for col in sentiment_cols if 'score' in col and col in df_with_features.columns), None)
            if sentiment_score_col:
                # Sentiment score extremes
                df_with_features['sentiment_extreme'] = np.where(
                    abs(df_with_features[f'{sentiment_score_col}_zscore_90']) > 2, 
                    np.sign(df_with_features[f'{sentiment_score_col}_zscore_90']), 
                    0
                )
                
                # Sentiment trend
                df_with_features['sentiment_trend'] = np.sign(df_with_features[f'{sentiment_score_col}_sma_3'] - df_with_features[f'{sentiment_score_col}_sma_14'])
                
                added_features.extend(['sentiment_extreme', 'sentiment_trend'])
            
            # Social dominance features
            dominance_col = next((col for col in sentiment_cols if 'dominance' in col and col in df_with_features.columns), None)
            if dominance_col:
                # Dominance change
                df_with_features['dominance_change'] = df_with_features[dominance_col].pct_change(periods=7)
                added_features.append('dominance_change')
            
            logger.info(f"Added {len(added_features)} sentiment features")
            
        except Exception as e:
            logger.error(f"Error adding sentiment features: {e}")
        
        return df_with_features
    
    def add_global_market_features(
        self,
        df: pd.DataFrame,
        global_market_data: Optional[pd.DataFrame] = None,
        market_columns: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Add global market metrics to the price data.
        
        Args:
            df: DataFrame with price data
            global_market_data: Optional DataFrame with global market data
                               If None, looks for columns with prefix 'market_' in df
            market_columns: List of market columns to include
                           If None, includes all market columns
            
        Returns:
            DataFrame with added global market features
        """
        logger.info("Adding global market features")
        
        # Make a copy to avoid modifying the original DataFrame
        df_with_features = df.copy()
        
        # If global_market_data is provided, join with price data
        if global_market_data is not None:
            # Check if index is datetime
            if isinstance(global_market_data.index, pd.DatetimeIndex) and isinstance(df.index, pd.DatetimeIndex):
                # Join the dataframes
                df_with_features = df.join(global_market_data, how='left')
            else:
                logger.error("Both DataFrames must have DatetimeIndex to join")
                return df
        
        # Identify market columns in the DataFrame
        market_prefix = 'market_'
        all_market_cols = [col for col in df_with_features.columns if col.startswith(market_prefix)]
        
        if not all_market_cols:
            logger.warning("No global market columns found in the DataFrame")
            return df_with_features
        
        # Use specified columns or all market columns
        market_cols = market_columns or all_market_cols
        
        # Add derived market features
        added_features = []
        
        try:
            # Process each market column
            for col in market_cols:
                if col in df_with_features.columns:
                    # Calculate returns
                    df_with_features[f'{col}_return_1d'] = df_with_features[col].pct_change()
                    
                    # Moving averages
                    for window in [7, 20, 50, 200]:
                        df_with_features[f'{col}_sma_{window}'] = df_with_features[col].rolling(window=window).mean()
                    
                    # Calculate momentum
                    df_with_features[f'{col}_momentum'] = df_with_features[col].pct_change(periods=20)
                    
                    # Volatility
                    df_with_features[f'{col}_volatility'] = df_with_features[col].pct_change().rolling(window=20).std() * np.sqrt(252)
                    
                    added_features.extend([
                        f'{col}_return_1d', f'{col}_sma_7', f'{col}_sma_20', 
                        f'{col}_sma_50', f'{col}_sma_200', f'{col}_momentum', f'{col}_volatility'
                    ])
            
            # Calculate correlations with price
            price_col = self.target_column or 'close'
            for col in market_cols:
                if col in df_with_features.columns:
                    for window in [20, 60, 120]:
                        df_with_features[f'{col}_price_corr_{window}'] = (
                            df_with_features[price_col].pct_change().rolling(window=window)
                                              .corr(df_with_features[col].pct_change())
                        )
                        added_features.append(f'{col}_price_corr_{window}')
            
            # Market regime indicators
            # Check for specific market metrics
            
            # VIX (volatility index)
            vix_col = next((col for col in market_cols if 'vix' in col.lower() and col in df_with_features.columns), None)
            if vix_col:
                # High volatility regime
                df_with_features['high_vol_regime'] = np.where(df_with_features[vix_col] > df_with_features[vix_col].rolling(window=252).quantile(0.8), 1, 0)
                
                # VIX trend
                df_with_features['vix_trend'] = np.sign(df_with_features[vix_col] - df_with_features[vix_col].rolling(window=20).mean())
                
                added_features.extend(['high_vol_regime', 'vix_trend'])
            
            # USD index
            usd_col = next((col for col in market_cols if 'usd' in col.lower() and col in df_with_features.columns), None)
            if usd_col:
                # USD strength indicator
                df_with_features['usd_strength'] = np.where(df_with_features[usd_col] > df_with_features[usd_col].rolling(window=50).mean(), 1, -1)
                added_features.append('usd_strength')
            
            # S&P 500 or other major index
            sp500_col = next((col for col in market_cols if ('sp500' in col.lower() or 'spx' in col.lower()) and col in df_with_features.columns), None)
            if sp500_col:
                # Risk-on/risk-off indicator
                df_with_features['risk_regime'] = np.where(
                    df_with_features[sp500_col] > df_with_features[sp500_col].rolling(window=200).mean(), 
                    'risk_on', 'risk_off'
                )
                
                # Convert to numeric
                df_with_features['risk_regime_num'] = np.where(df_with_features['risk_regime'] == 'risk_on', 1, -1)
                
                added_features.extend(['risk_regime', 'risk_regime_num'])
            
            # Interest rates
            rates_col = next((col for col in market_cols if ('treasury' in col.lower() or 'rate' in col.lower()) and col in df_with_features.columns), None)
            if rates_col:
                # Interest rate regime
                df_with_features['rate_regime'] = np.where(
                    df_with_features[rates_col] > df_with_features[rates_col].shift(20), 
                    'rising', 'falling'
                )
                
                # Convert to numeric
                df_with_features['rate_regime_num'] = np.where(df_with_features['rate_regime'] == 'rising', 1, -1)
                
                added_features.extend(['rate_regime', 'rate_regime_num'])
            
            # Commodity index
            commodity_col = next((col for col in market_cols if 'commodity' in col.lower() and col in df_with_features.columns), None)
            if commodity_col:
                # Commodity trend
                df_with_features['commodity_trend'] = np.sign(df_with_features[commodity_col] - df_with_features[commodity_col].rolling(window=50).mean())
                added_features.append('commodity_trend')
            
            logger.info(f"Added {len(added_features)} global market features")
            
        except Exception as e:
            logger.error(f"Error adding global market features: {e}")
        
        return df_with_features
    
    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handle missing values in the DataFrame.
        
        Args:
            df: DataFrame with potentially missing values
            
        Returns:
            DataFrame with imputed or dropped missing values
        """
        logger.info("Handling missing values")
        
        # Check for missing values
        missing_counts = df.isnull().sum()
        missing_pct = missing_counts / len(df) * 100
        
        # Log missing value stats
        missing_stats = pd.DataFrame({
            'missing_count': missing_counts,
            'missing_percent': missing_pct
        }).sort_values('missing_count', ascending=False)
        
        logger.info(f"Missing value statistics:\n{missing_stats[missing_stats['missing_count'] > 0]}")
        
        # Make a copy to avoid modifying the original DataFrame
        df_imputed = df.copy()
        
        # Get max allowed missing ratio from config or use default
        max_missing_ratio = self.config.get("max_missing_ratio", 0.3)
        
        # Get imputer type from config
        imputer_type = self.config.get("imputer", "mean")
        
        # Drop columns with too many missing values
        columns_to_drop = missing_stats[missing_stats['missing_percent'] > max_missing_ratio * 100].index.tolist()
        
        if columns_to_drop:
            logger.warning(f"Dropping columns with > {max_missing_ratio * 100:.1f}% missing values: {columns_to_drop}")
            df_imputed = df_imputed.drop(columns=columns_to_drop)
        
        # Apply different imputation methods for different column types
        numeric_cols = df_imputed.select_dtypes(include=['int', 'float']).columns
        categorical_cols = df_imputed.select_dtypes(include=['object', 'category']).columns
        
        # Impute numeric columns
        if len(numeric_cols) > 0:
            try:
                if imputer_type == "mean":
                    imputer = SimpleImputer(strategy='mean')
                    self.imputers['numeric'] = imputer
                    df_imputed[numeric_cols] = imputer.fit_transform(df_imputed[numeric_cols])
                
                elif imputer_type == "median":
                    imputer = SimpleImputer(strategy='median')
                    self.imputers['numeric'] = imputer
                    df_imputed[numeric_cols] = imputer.fit_transform(df_imputed[numeric_cols])
                
                elif imputer_type == "knn":
                    imputer = KNNImputer(n_neighbors=5)
                    self.imputers['numeric'] = imputer
                    df_imputed[numeric_cols] = imputer.fit_transform(df_imputed[numeric_cols])
                
                elif imputer_type == "forward":
                    df_imputed[numeric_cols] = df_imputed[numeric_cols].fillna(method='ffill')
                    
                    # For any remaining NaNs at the beginning, use backfill
                    df_imputed[numeric_cols] = df_imputed[numeric_cols].fillna(method='bfill')
                
                elif imputer_type == "linear":
                    # Linear interpolation
                    df_imputed[numeric_cols] = df_imputed[numeric_cols].interpolate(method='linear')
                    
                    # For any remaining NaNs at the edges, use forward/backward fill
                    df_imputed[numeric_cols] = df_imputed[numeric_cols].fillna(method='ffill').fillna(method='bfill')
                
                else:
                    logger.warning(f"Unknown imputer type: {imputer_type}, using mean imputation")
                    imputer = SimpleImputer(strategy='mean')
                    self.imputers['numeric'] = imputer
                    df_imputed[numeric_cols] = imputer.fit_transform(df_imputed[numeric_cols])
                    
            except Exception as e:
                logger.error(f"Error during numeric imputation: {e}")
                # Fallback to simpler imputation
                df_imputed[numeric_cols] = df_imputed[numeric_cols].fillna(df_imputed[numeric_cols].mean())
        
        # Impute categorical columns
        if len(categorical_cols) > 0:
            try:
                # Most frequent value imputation for categorical columns
                imputer = SimpleImputer(strategy='most                    # Tenkan-sen (Conversion Line)
                    high_values = df_with_indicators[high_col].rolling(window=conversion_period).max()
                    low_values = df_with_indicators[low_col].rolling(window=conversion_period).min()
                    df_with_indicators['ichimoku_conversion_line'] = (high_values + low_values) / 2
                    
                    # Kijun-sen (Base Line)
                    high_values = df_with_indicators[high_col].rolling(window=base_period).max()
                    low_values = df_with_indicators[low_col].rolling(window=base_period).min()
                    df_with_indicators['ichimoku_base_line'] = (high_values + low_values) / 2
                    
                    # Senkou Span A (Leading Span A)
                    df_with_indicators['ichimoku_senkou_span_a'] = (
                        (df_with_indicators['ichimoku_conversion_line'] + df_with_indicators['ichimoku_base_line']) / 2
                    ).shift(displacement)
                    
                    # Senkou Span B (Leading Span B)
                    high_values = df_with_indicators[high_col].rolling(window=lagging_span_period).max()
                    low_values = df_with_indicators[low_col].rolling(window=lagging_span_period).min()
                    df_with_indicators['ichimoku_senkou_span_b'] = (
                        (high_values + low_values) / 2
                    ).shift(displacement)
                    
                    # Chikou Span (Lagging Span)
                    df_with_indicators['ichimoku_chikou_span'] = df_with_indicators[close_col].shift(-displacement)
                    
                    added_indicators.extend([
                        'ichimoku_conversion_line', 'ichimoku_base_line',
                        'ichimoku_senkou_span_a', 'ichimoku_senkou_span_b', 'ichimoku_chikou_span'
                    ])
                
                elif indicator == "mfi":
                    # Money Flow Index
                    for window in params.get("windows", [14]):
                        # Typical price
                        typical_price = (df_with_indicators[high_col] + df_with_indicators[low_col] + df_with_indicators[close_col]) / 3
                        
                        # Money flow
                        money_flow = typical_price * df_with_indicators[volume_col]
                        
                        # Positive and negative money flow
                        positive_flow = np.where(typical_price > typical_price.shift(1), money_flow, 0)
                        negative_flow = np.where(typical_price < typical_price.shift(1), money_flow, 0)
                        
                        # Convert to pandas Series
                        positive_flow = pd.Series(positive_flow, index=df_with_indicators.index)
                        negative_flow = pd.Series(negative_flow, index=df_with_indicators.index)
                        
                        # Money flow ratio
                        positive_mf = positive_flow.rolling(window=window).sum()
                        negative_mf = negative_flow.rolling(window=window).sum()
                        
                        # Avoid division by zero
                        mf_ratio = positive_mf / negative_mf.replace(0, np.finfo(float).eps)
                        
                        # Money Flow Index
                        df_with_indicators[f'mfi_{window}'] = 100 - (100 / (1 + mf_ratio))
                        
                        added_indicators.append(f'mfi_{window}')
                
                elif indicator == "awesome_oscillator":
                    # Awesome Oscillator
                    median_price = (df_with_indicators[high_col] + df_with_indicators[low_col]) / 2
                    
                    # Calculate the simple moving averages
                    sma5 = median_price.rolling(window=5).mean()
                    sma34 = median_price.rolling(window=34).mean()
                    
                    # Calculate the Awesome Oscillator
                    df_with_indicators['awesome_oscillator'] = sma5 - sma34
                    
                    added_indicators.append('awesome_oscillator')
                
                elif indicator == "keltner_channel":
                    # Keltner Channel
                    for window in params.get("windows", [20]):
                        multiplier = params.get("multiplier", 2)
                        
                        # Calculate EMA of typical price
                        typical_price = (df_with_indicators[high_col] + df_with_indicators[low_col] + df_with_indicators[close_col]) / 3
                        ema = typical_price.ewm(span=window, adjust=False).mean()
                        
                        # Calculate ATR
                        tr1 = df_with_indicators[high_col] - df_with_indicators[low_col]
                        tr2 = abs(df_with_indicators[high_col] - df_with_indicators[close_col].shift())
                        tr3 = abs(df_with_indicators[low_col] - df_with_indicators[close_col].shift())
                        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
                        atr = tr.rolling(window=window).mean()
                        
                        # Calculate Keltner Channels
                        df_with_indicators[f'keltner_middle_{window}'] = ema
                        df_with_indicators[f'keltner_upper_{window}'] = ema + (multiplier * atr)
                        df_with_indicators[f'keltner_lower_{window}'] = ema - (multiplier * atr)
                        
                        added_indicators.extend([
                            f'keltner_middle_{window}', f'keltner_upper_{window}', f'keltner_lower_{window}'
                        ])
                
                elif indicator == "roc":
                    # Rate of Change
                    for window in params.get("windows", [10, 20, 50]):
                        df_with_indicators[f'roc_{window}'] = (
                            (df_with_indicators[close_col] - df_with_indicators[close_col].shift(window)) /
                            df_with_indicators[close_col].shift(window) * 100
                        )
                        
                        added_indicators.append(f'roc_{window}')
                
                elif indicator == "cci":
                    # Commodity Channel Index
                    for window in params.get("windows", [20]):
                        # Calculate typical price
                        typical_price = (df_with_indicators[high_col] + df_with_indicators[low_col] + df_with_indicators[close_col]) / 3
                        
                        # Calculate moving average of typical price
                        ma_tp = typical_price.rolling(window=window).mean()
                        
                        # Calculate mean deviation
                        mean_dev = abs(typical_price - ma_tp).rolling(window=window).mean()
                        
                        # Avoid division by zero
                        mean_dev = mean_dev.replace(0, np.finfo(float).eps)
                        
                        # Calculate CCI
                        df_with_indicators[f'cci_{window}'] = (typical_price - ma_tp) / (0.015 * mean_dev)
                        
                        added_indicators.append(f'cci_{window}')
                
                elif indicator == "donchian_channel":
                    # Donchian Channel
                    for window in params.get("windows", [20]):
                        df_with_indicators[f'donchian_high_{window}'] = df_with_indicators[high_col].rolling(window=window).max()
                        df_with_indicators[f'donchian_low_{window}'] = df_with_indicators[low_col].rolling(window=window).min()
                        df_with_indicators[f'donchian_mid_{window}'] = (
                            df_with_indicators[f'donchian_high_{window}'] + df_with_indicators[f'donchian_low_{window}']
                        ) / 2
                        
                        added_indicators.extend([
                            f'donchian_high_{window}', f'donchian_low_{window}', f'donchian_mid_{window}'
                        ])
                
                elif indicator == "psar":
                    # Parabolic SAR
                    af_start = params.get("af_start", 0.02)
                    af_increment = params.get("af_increment", 0.02)
                    af_max = params.get("af_max", 0.2)
                    
                    # Initialize columns
                    df_with_indicators['psar'] = np.nan
                    df_with_indicators['psar_up'] = np.nan
                    df_with_indicators['psar_down'] = np.nan
                    
                    # Simplistic PSAR implementation
                    # For more accurate results, consider using a library like ta-lib
                    trend_up = df_with_indicators[close_col].iloc[1] > df_with_indicators[close_col].iloc[0]
                    af = af_start
                    
                    if trend_up:
                        psar = df_with_indicators[low_col].iloc[0]
                        ep = df_with_indicators[high_col].iloc[1]
                    else:
                        psar = df_with_indicators[high_col].iloc[0]
                        ep = df_with_indicators[low_col].iloc[1]
                    
                    # Fill the first values
                    df_with_indicators.loc[df_with_indicators.index[1], 'psar'] = psar
                    if trend_up:
                        df_with_indicators.loc[df_with_indicators.index[1], 'psar_up'] = psar
                    else:
                        df_with_indicators.loc[df_with_indicators.index[1], 'psar_down'] = psar
                    
                    # Compute PSAR for the rest of the data
                    for i in range(2, len(df_with_indicators)):
                        # Update PSAR
                        psar = psar + af * (ep - psar)
                        
                        # Ensure PSAR doesn't go beyond the prior extremes
                        if trend_up:
                            psar = min(psar, df_with_indicators[low_col].iloc[i-1], df_with_indicators[low_col].iloc[i-2])
                        else:
                            psar = max(psar, df_with_indicators[high_col].iloc[i-1], df_with_indicators[high_col].iloc[i-2])
                        
                        # Check if trend changes
                        if (trend_up and df_with_indicators[low_col].iloc[i] < psar) or (not trend_up and df_with_indicators[high_col].iloc[i] > psar):
                            # Trend changes
                            trend_up = not trend_up
                            psar = ep
                            af = af_start
                            
                            if trend_up:
                                ep = df_with_indicators[high_col].iloc[i]
                            else:
                                ep = df_with_indicators[low_col].iloc[i]
                        else:
                            # Trend continues
                            if trend_up:
                                if df_with_indicators[high_col].iloc[i] > ep:
                                    ep = df_with_indicators[high_col].iloc[i]
                                    af = min(af + af_increment, af_max)
                            else:
                                if df_with_indicators[low_col].iloc[i] < ep:
                                    ep = df_with_indicators[low_col].iloc[i]
                                    af = min(af + af_increment, af_max)
                        
                        # Store results
                        df_with_indicators.loc[df_with_indicators.index[i], 'psar'] = psar
                        if trend_up:
                            df_with_indicators.loc[df_with_indicators.index[i], 'psar_up'] = psar
                        else:
                            df_with_indicators.loc[df_with_indicators.index[i], 'psar_down'] = psar
                    
                    added_indicators.extend(['psar', 'psar_up', 'psar_down'])
                
                elif indicator == "price_channels":
                    # Add price channel indicators (distance to high/low, normalized)
                    for window in params.get("windows", [20]):
                        # Rolling highs and lows
                        high_channel = df_with_indicators[high_col].rolling(window=window).max()
                        low_channel = df_with_indicators[low_col].rolling(window=window).min()
                        
                        # Distance to high and low channels
                        df_with_indicators[f'dist_to_high_{window}'] = (high_channel - df_with_indicators[close_col]) / df_with_indicators[close_col]
                        df_with_indicators[f'dist_to_low_{window}'] = (df_with_indicators[close_col] - low_channel) / df_with_indicators[close_col]
                        
                        # Position within channel (0 = at low, 1 = at high)
                        df_with_indicators[f'channel_pos_{window}'] = (df_with_indicators[close_col] - low_channel) / (high_channel - low_channel + np.finfo(float).eps)
                        
                        added_indicators.extend([f'dist_to_high_{window}', f'dist_to_low_{window}', f'channel_pos_{window}'])
                
                elif indicator == "relative_returns":
                    # Calculate relative returns against various timeframes
                    for window in params.get("windows", [1, 5, 10, 20, 60]):
                        df_with_indicators[f'return_{window}d'] = df_with_indicators[close_col].pct_change(periods=window)
                        added_indicators.append(f'return_{window}d')
                
                elif indicator == "z_score":
                    # Z-score (number of standard deviations from the mean)
                    for window in params.get("windows", [20, 50, 100]):
                        rolling_mean = df_with_indicators[close_col].rolling(window=window).mean()
                        rolling_std = df_with_indicators[close_col].rolling(window=window).std()
                        df_with_indicators[f'z_score_{window}'] = (df_with_indicators[close_col] - rolling_mean) / (rolling_std + np.finfo(float).eps)
                        added_indicators.append(f'z_score_{window}')
                
                elif indicator == "williams_r":
                    # Williams %R
                    for window in params.get("windows", [14]):
                        highest_high = df_with_indicators[high_col].rolling(window=window).max()
                        lowest_low = df_with_indicators[low_col].rolling(window=window).min()
                        df_with_indicators[f'williams_r_{window}'] = -100 * (highest_high - df_with_indicators[close_col]) / (highest_high - lowest_low + np.finfo(float).eps)
                        added_indicators.append(f'williams_r_{window}')
                
                elif indicator == "volume_indicators":
                    # Volume-based indicators
                    # Volume SMA
                    for window in params.get("windows", [20]):
                        df_with_indicators[f'volume_sma_{window}'] = df_with_indicators[volume_col].rolling(window=window).mean()
                        
                        # Volume ratio (current volume / volume SMA)
                        df_with_indicators[f'volume_ratio_{window}'] = df_with_indicators[volume_col] / df_with_indicators[f'volume_sma_{window}']
                        
                        # Volume oscillator (ratio of short-term to long-term volume)
                        if window > 10:  # Only calculate for longer windows
                            short_vol_sma = df_with_indicators[volume_col].rolling(window=5).mean()
                            long_vol_sma = df_with_indicators[f'volume_sma_{window}']
                            df_with_indicators[f'volume_osc_{window}'] = short_vol_sma / long_vol_sma - 1
                            
                            added_indicators.append(f'volume_osc_{window}')
                        
                        added_indicators.extend([f'volume_sma_{window}', f'volume_ratio_{window}'])
                    
                    # Volume standard deviation
                    for window in params.get("windows", [20]):
                        df_with_indicators[f'volume_std_{window}'] = df_with_indicators[volume_col].rolling(window=window).std()
                        added_indicators.append(f'volume_std_{window}')
                    
                    # Normalized volume (volume / average volume)
                    for window in params.get("windows", [20]):
                        df_with_indicators[f'norm_volume_{window}'] = df_with_indicators[volume_col] / df_with_indicators[volume_col].rolling(window=window).mean()
                        added_indicators.append(f'norm_volume_{window}')
                    
                elif indicator == "spread_indicators":
                    # Spread-based indicators (high-low, open-close)
                    
                    # High-Low range
                    df_with_indicators['hl_range'] = df_with_indicators[high_col] - df_with_indicators[low_col]
                    
                    # High-Low range relative to close
                    df_with_indicators['hl_range_pct'] = df_with_indicators['hl_range'] / df_with_indicators[close_col]
                    
                    # Open-Close range
                    open_col = self.config.get("ohlcv_columns", {}).get("open", "open")
                    df_with_indicators['oc_range'] = df_with_indicators[open_col] - df_with_indicators[close_col]
                    
                    # Open-Close range relative to close
                    df_with_indicators['oc_range_pct'] = df_with_indicators['oc_range'] / df_with_indicators[close_col]
                    
                    # Body size ratio (absolute Open-Close / High-Low)
                    df_with_indicators['body_ratio'] = abs(df_with_indicators['oc_range']) / (df_with_indicators['hl_range'] + np.finfo(float).eps)
                    
                    added_indicators.extend(['hl_range', 'hl_range_pct', 'oc_range', 'oc_range_pct', 'body_ratio'])
                
                elif indicator == "correlation_index":
                    # Correlation between price and volume
                    for window in params.get("windows", [20]):
                        price_returns = df_with_indicators[close_col].pct_change()
                        volume_changes = df_with_indicators[volume_col].pct_change()
                        
                        # Calculate rolling correlation
                        df_with_indicators[f'price_vol_corr_{window}'] = (
                            price_returns.rolling(window=window)
                                        .corr(volume_changes)
                        )
                        
                        added_indicators.append(f'price_vol_corr_{window}')
                
                # Add more indicators as needed
                
                else:
                    logger.warning(f"Unknown indicator: {indicator}")
            
            except Exception as e:
                logger.error(f"Error calculating indicator {indicator}: {e}")
        
        # Log the number of indicators added
        logger.info(f"Added {len(added_indicators)} technical indicators")
        
        return df_with_indicators
    
    def add_market_microstructure_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add market microstructure features to the price data.
        
        These features capture order flow dynamics, market efficiency,
        and liquidity characteristics.
        
        Args:
            df: DataFrame with price data
            
        Returns:
            DataFrame with added market microstructure features
        """
        logger.info("Adding market microstructure features")
        
        # Get configuration
        micro_features = self.config.get("market_microstructure_features", {})
        
        # If no features specified, return original DataFrame
        if not micro_features:
            logger.info("No market microstructure features specified")
            return df
        
        # Make a copy to avoid modifying the original DataFrame
        df_with_features = df.copy()
        
        # Track the features added
        added_features = []
        
        # Get price columns
        open_col = self.config.get("ohlcv_columns", {}).get("open", "open")
        high_col = self.config.get("ohlcv_columns", {}).get("high", "high")
        low_col = self.config.get("ohlcv_columns", {}).get("low", "low")
        close_col = self.config.get("ohlcv_columns", {}).get("close", "close")
        volume_col = self.config.get("ohlcv_columns", {}).get("volume", "volume")
        
        # Calculate each feature
        for feature, params in micro_features.items():
            try:
                if feature == "volatility_measures":
                    # Various volatility measures
                    
                    # Daily log returns
                    df_with_features['log_return'] = np.log(df_with_features[close_col] / df_with_features[close_col].shift(1))
                    
                    # Realized volatility
                    for window in params.get("windows", [10, 20, 30]):
                        df_with_features[f'realized_vol_{window}'] = df_with_features['log_return'].rolling(window=window).std() * np.sqrt(252)
                        
                        # Parkinson volatility estimator (uses high-low range)
                        df_with_features[f'parkinson_vol_{window}'] = np.sqrt(
                            252 / (4 * np.log(2)) * 
                            (np.log(df_with_features[high_col] / df_with_features[low_col]) ** 2).rolling(window=window).mean()
                        )
                        
                        # Garman-Klass volatility estimator (uses OHLC)
                        log_hl = np.log(df_with_features[high_col] / df_with_features[low_col])
                        log_co = np.log(df_with_features[close_col] / df_with_features[open_col])
                        
                        df_with_features[f'garman_klass_vol_{window}'] = np.sqrt(
                            252 * 0.5 * (log_hl ** 2) - (2 * np.log(2) - 1) * (log_co ** 2)
                        ).rolling(window=window).mean()
                        
                        added_features.extend([
                            f'realized_vol_{window}', f'parkinson_vol_{window}', f'garman_klass_vol_{window}'
                        ])
                    
                    added_features.append('log_return')
                
                elif feature == "illiquidity_measures":
                    # Amihud Illiquidity measure
                    for window in params.get("windows", [20]):
                        # |Return| / Volume
                        amihud = abs(df_with_features[close_col].pct_change()) / (df_with_features[volume_col] + np.finfo(float).eps)
                        df_with_features[f'amihud_{window}'] = amihud.rolling(window=window).mean()
                        
                        added_features.append(f'amihud_{window}')
                    
                    # Roll spread estimator
                    for window in params.get("windows", [20]):
                        price_changes = df_with_features[close_col].diff()
                        roll_cov = price_changes.rolling(window=window).apply(lambda x: np.cov(x[:-1], x[1:])[0, 1])
                        df_with_features[f'roll_spread_{window}'] = 2 * np.sqrt(-roll_cov)
                        
                        added_features.append(f'roll_spread_{window}')
                
                elif feature == "price_impact":
                    # Kyle's lambda (price impact)
                    for window in params.get("windows", [20]):
                        returns = df_with_features[close_col].pct_change()
                        
                        # Calculate price impact (returns divided by volume)
                        df_with_features[f'kyle_lambda_{window}'] = (
                            returns.abs() / (df_with_features[volume_col] + np.finfo(float).eps)
                        ).rolling(window=window).mean()
                        
                        added_features.append(f'kyle_lambda_{window}')
                
                elif feature == "volume_weighted_volatility":
                    # Volume-weighted volatility
                    for window in params.get("windows", [20]):
                        vw_returns = df_with_features[close_col].pct_change() * df_with_features[volume_col]
                        vw_returns_mean = vw_returns.rolling(window=window).mean()
                        
                        df_with_features[f'vol_weighted_vol_{window}'] = np.sqrt(
                            ((vw_returns - vw_returns_mean) ** 2).rolling(window=window).mean()
                        )
                        
                        added_features.append(f'vol_weighted_vol_{window}')
                
                elif feature == "tick_rule":
                    # Tick rule (proxy for order flow)
                    df_with_features['price_change'] = df_with_features[close_col].diff()
                    df_with_features['tick_rule'] = np.sign(df_with_features['price_change'])
                    
                    # Replace 0s with previous non-zero value
                    df_with_features.loc[df_with_features['tick_rule'] == 0, 'tick_rule'] = np.nan
                    df_with_features['tick_rule'] = df_with_features['tick_rule'].fillna(method='ffill')
                    
                    for window in params.get("windows", [20]):
                        df_with_features[f'tick_rule_sum_{window}'] = df_with_features['tick_rule'].rolling(window=window).sum()
                        
                        # Normalize to range [-1, 1]
                        df_with_features[f'tick_rule_indicator_{window}'] = df_with_features[f'tick_rule_sum_{window}'] / window
                        
                        added_features.extend(['tick_rule', f'tick_rule_sum_{window}', f'tick_rule_indicator_{window}'])
                
                elif feature == "vpin":
                    # Volume-Synchronized Probability of Informed Trading
                    bucket_size = params.get("bucket_size", 50)  # Number of buckets per day
                    
                    # Calculate average daily volume
                    if isinstance(df_with_features.index, pd.DatetimeIndex):
                        avg_daily_volume = df_with_features.groupby(df_with_features.index.date)[volume_col].sum().mean()
                    else:
                        avg_daily_volume = df_with_features[volume_col].rolling(window=1440).mean().mean()  # Assuming minute data
                    
                    # Calculate bucket volume
                    bucket_volume = avg_daily_volume / bucket_size
                    
                    # Initialize arrays for calculation
                    num_rows = len(df_with_features)
                    vpin_values = np.zeros(num_rows)
                    
                    volume_so_far = 0
                    buy_volume_so_far = 0
                    sell_volume_so_far = 0
                    bucket_idx = 0
                    vpin_window = params.get("vpin_window", 50)
                    
                    # Simplified VPIN calculation
                    for i in range(1, num_rows):
                        # Classify volume as buy or sell using tick rule
                        price_change = df_with_features[close_col].iloc[i] - df_with_features[close_col].iloc[i-1]
                        volume = df_with_features[volume_col].iloc[i]
                        
                        if price_change > 0:
                            buy_volume = volume
                            sell_volume = 0
                        elif price_change < 0:
                            buy_volume = 0
                            sell_volume = volume
                        else:
                            # No price change, split volume equally
                            buy_volume = volume / 2
                            sell_volume = volume / 2
                        
                        buy_volume_so_far += buy_volume
                        sell_volume_so_far += sell_volume
                        volume_so_far += volume
                        
                        if volume_so_far >= bucket_volume:
                            # Bucket is filled, calculate VPIN
                            vpin_values[i] = abs(buy_volume_so_far - sell_volume_so_far) / volume_so_far
                            
                            # Reset for next bucket
                            volume_so_far = 0
                            buy_volume_so_far = 0
                            sell_volume_so_far = 0
                            bucket_idx += 1
                    
                    # Convert to pandas Series and calculate rolling average
                    vpin_series = pd.Series(vpin_values)
                    df_with_features['vpin'] = vpin_series.rolling(window=vpin_window, min_periods=1).mean()
                    
                    added_features.append('vpin')
                
                elif feature == "market_efficiency":
                    # Market Efficiency Coefficient (MEC)
                    for window in params.get("windows", [20]):
                        # Calculate variance ratio
                        returns = df_with_features[close_col].pct_change()
                        var_long = returns.rolling(window=window).var()
                        var_short = returns.rolling(window=1).var() * window
                        
                        df_with_features[f'mec_{window}'] = var_long / (var_short + np.finfo(float).eps)
                        
                        added_features.append(f'mec_{window}')
                
                elif feature == "advanced_volatility":
                    # Advanced volatility metrics
                    
                    # Calculate returns
                    returns = df_with_features[close_col].pct_change()
                    
                    for window in params.get("windows", [20]):
                        # Upside/downside volatility
                        up_returns = returns.copy()
                        up_returns[up_returns < 0] = 0
                        
                        down_returns = returns.copy()
                        down_returns[down_returns > 0] = 0
                        
                        df_with_features[f'up_vol_{window}'] = up_returns.rolling(window=window).std() * np.sqrt(252)
                        df_with_features[f'down_vol_{window}'] = down_returns.abs().rolling(window=window).std() * np.sqrt(252)
                        
                        # Volatility ratio (up/down)
                        df_with_features[f'vol_ratio_{window}'] = (
                            df_"""
Ariadne Preprocessor - Data preprocessing for the Ariadne trading model

This module provides comprehensive data preprocessing capabilities for the Ariadne
trading model, including:

- Time series data normalization and standardization
- Technical indicator generation
- Market microstructure feature extraction
- On-chain data integration
- Sequence creation for deep learning models
- Data augmentation techniques specific to financial time series
- Anomaly detection and handling
- Missing data imputation
- Feature selection and dimensionality reduction

Author: Minos-AI Team
Date: December 15, 2024
License: Proprietary
"""

import os
import json
import logging
import warnings
from typing import Dict, List, Tuple, Union, Optional, Any, Callable
from pathlib import Path
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, QuantileTransformer
from sklearn.decomposition import PCA
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
from sklearn.ensemble import IsolationForest
from tslearn.preprocessing import TimeSeriesScalerMinMax, TimeSeriesScalerMeanVariance
import statsmodels.api as sm
from scipy import stats, signal
import joblib

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("ariadne_preprocessor.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Suppress specific warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning, message="invalid value encountered in")
pd.options.mode.chained_assignment = None


class AriadnePreprocessor:
    """
    Comprehensive data preprocessor for the Ariadne trading model.
    
    This class provides a complete pipeline for processing financial time series data
    for deep learning models, with special focus on cryptocurrency market data and
    on-chain metrics.
    
    Features:
    - Multi-timeframe processing
    - Technical indicator calculation
    - On-chain data integration
    - Market microstructure features
    - Sentiment data integration
    - Sequence creation and padding
    - Data augmentation for financial time series
    - Feature normalization and standardization
    - Cross-validation strategies for time series
    """
    
    # Class constants
    PREPROCESSOR_VERSION = "1.0.0"
    SUPPORTED_SCALERS = ["standard", "minmax", "robust", "quantile", "timeseries"]
    SUPPORTED_IMPUTERS = ["mean", "median", "knn", "forward", "linear"]
    MINIMUM_DATA_POINTS = 100
    
    def __init__(
        self,
        config_path: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        output_dir: Optional[str] = None,
        verbose: bool = True
    ):
        """
        Initialize the Ariadne preprocessor.
        
        Args:
            config_path: Path to JSON configuration file
            config: Configuration dictionary (overrides config_path if provided)
            output_dir: Directory to save processed data and scalers
            verbose: Whether to print verbose output
            
        Raises:
            FileNotFoundError: If config_path is provided but file doesn't exist
            ValueError: If neither config_path nor config is provided or config is invalid
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
        
        # Set output directory
        if output_dir is None:
            self.output_dir = Path(self.config.get("output_dir", "./data"))
        else:
            self.output_dir = Path(output_dir)
        
        # Create output directory if it doesn't exist
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Validate configuration
        self._validate_config()
        
        # Initialize preprocessor attributes
        self.feature_scalers = {}
        self.label_scalers = {}
        self.imputers = {}
        self.pca_transformers = {}
        self.feature_selectors = {}
        self.outlier_detectors = {}
        
        # Extract key configuration parameters
        self.sequence_length = self.config.get("sequence_length", 60)
        self.prediction_horizon = self.config.get("prediction_horizon", 5)
        self.feature_scaler_type = self.config.get("feature_scaler", "standard")
        self.label_scaler_type = self.config.get("label_scaler", "standard")
        self.train_val_test_split = self.config.get("train_val_test_split", [0.7, 0.15, 0.15])
        self.random_seed = self.config.get("random_seed", 42)
        self.features_to_include = self.config.get("features_to_include", [])
        self.target_column = self.config.get("target_column", "close")
        self.datetime_column = self.config.get("datetime_column", "timestamp")
        
        # Set random seeds for reproducibility
        np.random.seed(self.random_seed)
        
        if self.verbose:
            logger.info(f"Ariadne preprocessor initialized (Version: {self.PREPROCESSOR_VERSION})")
            logger.info(f"Output directory: {self.output_dir}")
    
    def _validate_config(self) -> None:
        """
        Validate the preprocessor configuration.
        
        Checks for required parameters, parameter types, and valid values.
        
        Raises:
            ValueError: If required configuration parameters are missing or invalid
        """
        # Required fields
        required_fields = ["sequence_length", "prediction_horizon"]
        for field in required_fields:
            if field not in self.config:
                logger.error(f"Missing required configuration parameter: {field}")
                raise ValueError(f"Missing required configuration parameter: {field}")
        
        # Validate scaler types
        feature_scaler = self.config.get("feature_scaler", "standard")
        if feature_scaler not in self.SUPPORTED_SCALERS:
            logger.error(f"Unsupported feature scaler: {feature_scaler}. "
                        f"Supported scalers are: {self.SUPPORTED_SCALERS}")
            raise ValueError(f"Unsupported feature scaler: {feature_scaler}. "
                           f"Supported scalers are: {self.SUPPORTED_SCALERS}")
        
        label_scaler = self.config.get("label_scaler", "standard")
        if label_scaler not in self.SUPPORTED_SCALERS:
            logger.error(f"Unsupported label scaler: {label_scaler}. "
                        f"Supported scalers are: {self.SUPPORTED_SCALERS}")
            raise ValueError(f"Unsupported label scaler: {label_scaler}. "
                           f"Supported scalers are: {self.SUPPORTED_SCALERS}")
        
        # Validate imputer types
        imputer = self.config.get("imputer", "mean")
        if imputer not in self.SUPPORTED_IMPUTERS:
            logger.error(f"Unsupported imputer: {imputer}. "
                        f"Supported imputers are: {self.SUPPORTED_IMPUTERS}")
            raise ValueError(f"Unsupported imputer: {imputer}. "
                           f"Supported imputers are: {self.SUPPORTED_IMPUTERS}")
        
        # Validate numeric parameters
        numeric_params = {
            "sequence_length": (int, (1, None)),
            "prediction_horizon": (int, (1, None)),
            "max_missing_ratio": (float, (0, 1)),
            "anomaly_contamination": (float, (0, 0.5)),
            "pca_variance": (float, (0, 1))
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
        
        # Validate train/val/test split
        train_val_test_split = self.config.get("train_val_test_split", [0.7, 0.15, 0.15])
        if not isinstance(train_val_test_split, list) or len(train_val_test_split) != 3:
            logger.error("train_val_test_split should be a list of 3 floats")
            raise ValueError("train_val_test_split should be a list of 3 floats")
        
        if abs(sum(train_val_test_split) - 1.0) > 1e-6:
            logger.error(f"train_val_test_split values should sum to 1.0, got {sum(train_val_test_split)}")
            raise ValueError(f"train_val_test_split values should sum to 1.0, got {sum(train_val_test_split)}")
        
        # Validate technical indicators
        tech_indicators = self.config.get("technical_indicators", {})
        for indicator, params in tech_indicators.items():
            if not isinstance(params, dict):
                logger.error(f"Parameters for indicator {indicator} should be a dictionary")
                raise ValueError(f"Parameters for indicator {indicator} should be a dictionary")
        
        logger.info("Configuration validation completed successfully")
    
    def load_data(
        self,
        data_path: str,
        additional_data_paths: Optional[Dict[str, str]] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Load and prepare raw financial data for preprocessing.
        
        Args:
            data_path: Path to the main price data file (CSV format)
            additional_data_paths: Dictionary mapping data type to file path for additional data
                                  (e.g., {"on_chain": "on_chain_data.csv", "sentiment": "sentiment_data.csv"})
            start_date: Optional start date for filtering data (format: "YYYY-MM-DD")
            end_date: Optional end date for filtering data (format: "YYYY-MM-DD")
            
        Returns:
            DataFrame with combined and prepared raw data
            
        Raises:
            FileNotFoundError: If any data file is not found
            ValueError: If data format is invalid or insufficient data points
        """
        try:
            # Load main price data
            logger.info(f"Loading price data from {data_path}")
            
            if data_path.endswith('.csv'):
                price_data = pd.read_csv(data_path)
            elif data_path.endswith('.parquet'):
                price_data = pd.read_parquet(data_path)
            elif data_path.endswith('.h5'):
                price_data = pd.read_hdf(data_path)
            else:
                logger.error(f"Unsupported file format for {data_path}")
                raise ValueError(f"Unsupported file format for {data_path}")
            
            # Validate price data
            self._validate_price_data(price_data)
            
            # Convert datetime column
            datetime_column = self.datetime_column
            if datetime_column in price_data.columns:
                if price_data[datetime_column].dtype == 'object':
                    price_data[datetime_column] = pd.to_datetime(price_data[datetime_column])
                price_data.set_index(datetime_column, inplace=True)
            
            # Filter by date if specified
            if start_date is not None:
                start_date = pd.to_datetime(start_date)
                price_data = price_data[price_data.index >= start_date]
                
            if end_date is not None:
                end_date = pd.to_datetime(end_date)
                price_data = price_data[price_data.index <= end_date]
            
            # Check if we have enough data points after filtering
            if len(price_data) < self.MINIMUM_DATA_POINTS:
                logger.error(f"Insufficient data points after date filtering: {len(price_data)} < {self.MINIMUM_DATA_POINTS}")
                raise ValueError(f"Insufficient data points after date filtering: {len(price_data)} < {self.MINIMUM_DATA_POINTS}")
            
            # Load and merge additional data if provided
            if additional_data_paths:
                for data_type, file_path in additional_data_paths.items():
                    logger.info(f"Loading {data_type} data from {file_path}")
                    
                    if not os.path.exists(file_path):
                        logger.error(f"File not found: {file_path}")
                        raise FileNotFoundError(f"File not found: {file_path}")
                    
                    if file_path.endswith('.csv'):
                        additional_data = pd.read_csv(file_path)
                    elif file_path.endswith('.parquet'):
                        additional_data = pd.read_parquet(file_path)
                    elif file_path.endswith('.h5'):
                        additional_data = pd.read_hdf(file_path)
                    else:
                        logger.error(f"Unsupported file format for {file_path}")
                        raise ValueError(f"Unsupported file format for {file_path}")
                    
                    # Convert datetime column if present
                    if datetime_column in additional_data.columns:
                        if additional_data[datetime_column].dtype == 'object':
                            additional_data[datetime_column] = pd.to_datetime(additional_data[datetime_column])
                        additional_data.set_index(datetime_column, inplace=True)
                    
                    # Add prefix to column names to avoid conflicts
                    additional_data = additional_data.add_prefix(f"{data_type}_")
                    
                    # Join with price data
                    price_data = price_data.join(additional_data, how='left')
            
            # Sort by datetime
            price_data.sort_index(inplace=True)
            
            logger.info(f"Data loaded successfully with {len(price_data)} rows and {len(price_data.columns)} columns")
            
            return price_data
            
        except Exception as e:
            logger.error(f"Failed to load data: {e}")
            raise
    
    def _validate_price_data(self, df: pd.DataFrame) -> None:
        """
        Validate price data format and content.
        
        Args:
            df: DataFrame with price data
            
        Raises:
            ValueError: If data format is invalid or required columns are missing
        """
        # Check if DataFrame is empty
        if df.empty:
            logger.error("Price data is empty")
            raise ValueError("Price data is empty")
        
        # Check minimum number of data points
        if len(df) < self.MINIMUM_DATA_POINTS:
            logger.error(f"Insufficient data points: {len(df)} < {self.MINIMUM_DATA_POINTS}")
            raise ValueError(f"Insufficient data points: {len(df)} < {self.MINIMUM_DATA_POINTS}")
        
        # Check for required columns (OHLCV)
        required_columns = self.config.get("required_columns", ["open", "high", "low", "close", "volume"])
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            logger.error(f"Missing required columns: {missing_columns}")
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        # Check target column
        if self.target_column not in df.columns:
            logger.error(f"Target column '{self.target_column}' not found in data")
            raise ValueError(f"Target column '{self.target_column}' not found in data")
        
        # Check datetime column if not already set as index
        if self.datetime_column not in df.columns and not isinstance(df.index, pd.DatetimeIndex):
            logger.error(f"Datetime column '{self.datetime_column}' not found in data")
            raise ValueError(f"Datetime column '{self.datetime_column}' not found in data")
    
    def add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add technical indicators to the price data.
        
        Calculates a wide range of technical indicators based on configuration.
        
        Args:
            df: DataFrame with price data
            
        Returns:
            DataFrame with added technical indicators
        """
        logger.info("Adding technical indicators")
        
        # Get indicator configuration
        tech_indicators = self.config.get("technical_indicators", {})
        
        # If no indicators specified, return original DataFrame
        if not tech_indicators:
            logger.info("No technical indicators specified")
            return df
        
        # Make a copy to avoid modifying the original DataFrame
        df_with_indicators = df.copy()
        
        # Track the indicators added
        added_indicators = []
        
        # Get price columns for indicators
        open_col = self.config.get("ohlcv_columns", {}).get("open", "open")
        high_col = self.config.get("ohlcv_columns", {}).get("high", "high")
        low_col = self.config.get("ohlcv_columns", {}).get("low", "low")
        close_col = self.config.get("ohlcv_columns", {}).get("close", "close")
        volume_col = self.config.get("ohlcv_columns", {}).get("volume", "volume")
        
        # Calculate each indicator
        for indicator, params in tech_indicators.items():
            try:
                if indicator == "sma":
                    # Simple Moving Average
                    for window in params.get("windows", [10, 20, 50, 100, 200]):
                        df_with_indicators[f'sma_{window}'] = df_with_indicators[close_col].rolling(window=window).mean()
                        added_indicators.append(f'sma_{window}')
                
                elif indicator == "ema":
                    # Exponential Moving Average
                    for window in params.get("windows", [10, 20, 50, 100, 200]):
                        df_with_indicators[f'ema_{window}'] = df_with_indicators[close_col].ewm(span=window, adjust=False).mean()
                        added_indicators.append(f'ema_{window}')
                
                elif indicator == "rsi":
                    # Relative Strength Index
                    for window in params.get("windows", [14]):
                        delta = df_with_indicators[close_col].diff()
                        gain = delta.where(delta > 0, 0)
                        loss = -delta.where(delta < 0, 0)
                        
                        avg_gain = gain.rolling(window=window).mean()
                        avg_loss = loss.rolling(window=window).mean()
                        
                        rs = avg_gain / avg_loss.replace(0, np.finfo(float).eps)  # Avoid division by zero
                        df_with_indicators[f'rsi_{window}'] = 100 - (100 / (1 + rs))
                        added_indicators.append(f'rsi_{window}')
                
                elif indicator == "macd":
                    # Moving Average Convergence Divergence
                    fast = params.get("fast", 12)
                    slow = params.get("slow", 26)
                    signal = params.get("signal", 9)
                    
                    ema_fast = df_with_indicators[close_col].ewm(span=fast, adjust=False).mean()
                    ema_slow = df_with_indicators[close_col].ewm(span=slow, adjust=False).mean()
                    
                    df_with_indicators['macd_line'] = ema_fast - ema_slow
                    df_with_indicators['macd_signal'] = df_with_indicators['macd_line'].ewm(span=signal, adjust=False).mean()
                    df_with_indicators['macd_histogram'] = df_with_indicators['macd_line'] - df_with_indicators['macd_signal']
                    
                    added_indicators.extend(['macd_line', 'macd_signal', 'macd_histogram'])
                
                elif indicator == "bollinger_bands":
                    # Bollinger Bands
                    window = params.get("window", 20)
                    num_std = params.get("num_std", 2)
                    
                    df_with_indicators[f'bb_middle_{window}'] = df_with_indicators[close_col].rolling(window=window).mean()
                    stddev = df_with_indicators[close_col].rolling(window=window).std()
                    
                    df_with_indicators[f'bb_upper_{window}'] = df_with_indicators[f'bb_middle_{window}'] + (stddev * num_std)
                    df_with_indicators[f'bb_lower_{window}'] = df_with_indicators[f'bb_middle_{window}'] - (stddev * num_std)
                    df_with_indicators[f'bb_width_{window}'] = (df_with_indicators[f'bb_upper_{window}'] - df_with_indicators[f'bb_lower_{window}']) / df_with_indicators[f'bb_middle_{window}']
                    df_with_indicators[f'bb_pct_{window}'] = (df_with_indicators[close_col] - df_with_indicators[f'bb_lower_{window}']) / (df_with_indicators[f'bb_upper_{window}'] - df_with_indicators[f'bb_lower_{window}'])
                    
                    added_indicators.extend([f'bb_middle_{window}', f'bb_upper_{window}', f'bb_lower_{window}', f'bb_width_{window}', f'bb_pct_{window}'])
                
                elif indicator == "atr":
                    # Average True Range
                    for window in params.get("windows", [14]):
                        tr1 = df_with_indicators[high_col] - df_with_indicators[low_col]
                        tr2 = abs(df_with_indicators[high_col] - df_with_indicators[close_col].shift())
                        tr3 = abs(df_with_indicators[low_col] - df_with_indicators[close_col].shift())
                        
                        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
                        df_with_indicators[f'atr_{window}'] = tr.rolling(window=window).mean()
                        
                        # Normalized ATR (ATR / Close)
                        df_with_indicators[f'natr_{window}'] = df_with_indicators[f'atr_{window}'] / df_with_indicators[close_col] * 100
                        
                        added_indicators.extend([f'atr_{window}', f'natr_{window}'])
                
                elif indicator == "stochastic":
                    # Stochastic Oscillator
                    k_period = params.get("k_period", 14)
                    d_period = params.get("d_period", 3)
                    slowing = params.get("slowing", 3)
                    
                    # Calculate %K
                    low_min = df_with_indicators[low_col].rolling(window=k_period).min()
                    high_max = df_with_indicators[high_col].rolling(window=k_period).max()
                    
                    # Handle division by zero
                    denom = high_max - low_min
                    denom = denom.replace(0, np.finfo(float).eps)
                    
                    df_with_indicators['stoch_%k'] = 100 * ((df_with_indicators[close_col] - low_min) / denom)
                    
                    # Calculate %D
                    df_with_indicators['stoch_%d'] = df_with_indicators['stoch_%k'].rolling(window=d_period).mean()
                    
                    added_indicators.extend(['stoch_%k', 'stoch_%d'])
                
                elif indicator == "adx":
                    # Average Directional Index
                    window = params.get("window", 14)
                    
                    # Calculate directional movement
                    plus_dm = df_with_indicators[high_col].diff()
                    minus_dm = df_with_indicators[low_col].diff(-1).abs()
                    
                    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
                    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)
                    
                    # Calculate true range
                    tr1 = df_with_indicators[high_col] - df_with_indicators[low_col]
                    tr2 = abs(df_with_indicators[high_col] - df_with_indicators[close_col].shift())
                    tr3 = abs(df_with_indicators[low_col] - df_with_indicators[close_col].shift())
                    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
                    
                    # Smooth with EMA
                    plus_di = 100 * plus_dm.ewm(span=window, adjust=False).mean() / tr.ewm(span=window, adjust=False).mean()
                    minus_di = 100 * minus_dm.ewm(span=window, adjust=False).mean() / tr.ewm(span=window, adjust=False).mean()
                    
                    # Calculate DX and ADX
                    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di).replace(0, np.finfo(float).eps)
                    adx = dx.ewm(span=window, adjust=False).mean()
                    
                    df_with_indicators[f'adx_{window}'] = adx
                    df_with_indicators[f'plus_di_{window}'] = plus_di
                    df_with_indicators[f'minus_di_{window}'] = minus_di
                    
                    added_indicators.extend([f'adx_{window}', f'plus_di_{window}', f'minus_di_{window}'])
                
                elif indicator == "obv":
                    # On-Balance Volume
                    df_with_indicators['obv'] = np.nan
                    df_with_indicators.loc[0, 'obv'] = df_with_indicators.loc[0, volume_col]
                    
                    for i in range(1, len(df_with_indicators)):
                        if df_with_indicators.iloc[i][close_col] > df_with_indicators.iloc[i-1][close_col]:
                            df_with_indicators.iloc[i, df_with_indicators.columns.get_loc('obv')] = (
                                df_with_indicators.iloc[i-1]['obv'] + df_with_indicators.iloc[i][volume_col]
                            )
                        elif df_with_indicators.iloc[i][close_col] < df_with_indicators.iloc[i-1][close_col]:
                            df_with_indicators.iloc[i, df_with_indicators.columns.get_loc('obv')] = (
                                df_with_indicators.iloc[i-1]['obv'] - df_with_indicators.iloc[i][volume_col]
                            )
                        else:
                            df_with_indicators.iloc[i, df_with_indicators.columns.get_loc('obv')] = df_with_indicators.iloc[i-1]['obv']
                    
                    added_indicators.append('obv')
                
                elif indicator == "vwap":
                    # Volume Weighted Average Price
                    # Typically calculated intraday, reset each day
                    if isinstance(df_with_indicators.index, pd.DatetimeIndex):
                        day_groups = df_with_indicators.groupby(df_with_indicators.index.date)
                        df_with_indicators['vwap'] = np.nan
                        
                        for day, group in day_groups:
                            typical_price = (group[high_col] + group[low_col] + group[close_col]) / 3
                            df_with_indicators.loc[group.index, 'vwap'] = (
                                (typical_price * group[volume_col]).cumsum() / group[volume_col].cumsum()
                            )
                        
                        added_indicators.append('vwap')
                    else:
                        logger.warning("VWAP calculation requires DatetimeIndex")
                
                elif indicator == "volatility":
                    # Historical Volatility
                    for window in params.get("windows", [10, 20, 30]):
                        # Log returns
                        returns = np.log(df_with_indicators[close_col] / df_with_indicators[close_col].shift(1))
                        
                        # Rolling standard deviation of returns
                        df_with_indicators[f'volatility_{window}'] = returns.rolling(window=window).std() * np.sqrt(252)  # Annualized
                        added_indicators.append(f'volatility_{window}')
                
                elif indicator == "ichimoku":
                    # Ichimoku Cloud
                    conversion_period = params.get("conversion_period", 9)
                    base_period = params.get("base_period", 26)
                    lagging_span_period = params.get("lagging_span_period", 52)
                    displacement = params.get("displacement", 26)
                    
                    # Tenkan-sen (Conversion Line)