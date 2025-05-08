train_time = time.time() - start_time
                
                # Evaluate on validation set
                val_metrics = model.evaluate(X_val, y_val, verbose=0)
                val_results = dict(zip(model.metrics_names, val_metrics))
                
                # Log metrics
                for name, value in history.history.items():
                    # Log final epoch metrics
                    mlflow.log_metric(name, value[-1])
                
                # Log training time
                mlflow.log_metric("training_time", train_time)
                
                # Log model
                mlflow.keras.log_model(model, "model")
                
                # Save model summary as artifact
                with open("model_summary.txt", "w") as f:
                    model.summary(print_fn=lambda x: f.write(x + '\n'))
                mlflow.log_artifact("model_summary.txt")
                
                # Log learning curves
                self._log_learning_curves(history.history)
            
            # Save model
            self.model = model
            model.save(os.path.join(self.output_dir, "model.h5"))
            
            # Record results
            results = {
                "training_time": train_time,
                "history": history.history,
                "val_metrics": val_results,
                "hyperparameters": hyperparams
            }
            
            logger.info(f"Training complete. Validation loss: {val_results['loss']:.4f}")
            
            return results
            
        except Exception as e:
            logger.error(f"Error during training: {str(e)}", exc_info=True)
            raise
    
    def _get_training_callbacks(self, hyperparams: Dict[str, Any]) -> List[tf.keras.callbacks.Callback]:
        """
        Get callbacks for model training.
        
        Args:
            hyperparams: Hyperparameters dictionary
            
        Returns:
            List of Keras callbacks
        """
        callbacks = []
        
        # Early stopping
        if hyperparams.get("early_stopping", True):
            patience = hyperparams.get("early_stopping_patience", 10)
            callbacks.append(EarlyStopping(
                monitor='val_loss',
                patience=patience,
                restore_best_weights=True,
                verbose=1
            ))
        
        # Learning rate reduction
        if hyperparams.get("reduce_lr_on_plateau", True):
            lr_patience = hyperparams.get("reduce_lr_patience", 5)
            lr_factor = hyperparams.get("reduce_lr_factor", 0.5)
            callbacks.append(ReduceLROnPlateau(
                monitor='val_loss',
                factor=lr_factor,
                patience=lr_patience,
                min_lr=1e-6,
                verbose=1
            ))
        
        # Model checkpoint
        model_path = os.path.join(self.output_dir, "checkpoints")
        os.makedirs(model_path, exist_ok=True)
        
        callbacks.append(ModelCheckpoint(
            filepath=os.path.join(model_path, "model_epoch_{epoch:02d}_val_loss_{val_loss:.4f}.h5"),
            save_best_only=True,
            monitor='val_loss',
            verbose=1
        ))
        
        return callbacks
    
    def _log_learning_curves(self, history: Dict[str, List[float]]) -> None:
        """
        Create and log learning curves as MLflow artifacts.
        
        Args:
            history: Training history dictionary
        """
        try:
            import matplotlib.pyplot as plt
            
            # Loss curves
            plt.figure(figsize=(10, 6))
            plt.plot(history['loss'], label='Training Loss')
            plt.plot(history['val_loss'], label='Validation Loss')
            plt.title('Loss During Training')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            
            loss_fig_path = os.path.join(self.output_dir, "loss_curves.png")
            plt.savefig(loss_fig_path)
            mlflow.log_artifact(loss_fig_path)
            
            # Metrics curves
            metrics = [m for m in history.keys() if m not in ['loss', 'val_loss']]
            
            for metric in metrics:
                val_metric = f'val_{metric}'
                if val_metric in history:
                    plt.figure(figsize=(10, 6))
                    plt.plot(history[metric], label=f'Training {metric}')
                    plt.plot(history[val_metric], label=f'Validation {metric}')
                    plt.title(f'{metric.upper()} During Training')
                    plt.xlabel('Epoch')
                    plt.ylabel(metric)
                    plt.legend()
                    plt.grid(True)
                    plt.tight_layout()
                    
                    metric_fig_path = os.path.join(self.output_dir, f"{metric}_curves.png")
                    plt.savefig(metric_fig_path)
                    mlflow.log_artifact(metric_fig_path)
                    
            plt.close('all')
            
        except Exception as e:
            logger.warning(f"Error creating learning curves: {str(e)}")
    
    def evaluate(self, dataset_dict: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """
        Evaluate trained model on test data.
        
        Args:
            dataset_dict: Dictionary containing datasets
            
        Returns:
            Dictionary of evaluation results
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet")
            
        try:
            # Extract test dataset
            X_test = dataset_dict["X_test"]
            y_test = dataset_dict["y_test"]
            
            # Get original test data for calculating metrics
            test_data_orig = None
            if self.data_processor:
                test_data_orig = self.data_processor.get_original_test_data()
            
            # Make predictions
            y_pred = self.model.predict(X_test)
            
            # Inverse transform predictions and actual values (if scaler is available)
            if self.data_processor and hasattr(self.data_processor, 'inverse_transform_y'):
                y_test_orig = self.data_processor.inverse_transform_y(y_test)
                y_pred_orig = self.data_processor.inverse_transform_y(y_pred)
            else:
                y_test_orig = y_test
                y_pred_orig = y_pred
            
            # Calculate standard regression metrics
            regression_metrics = calculate_regression_metrics(y_test_orig, y_pred_orig)
            
            # Calculate directional metrics if we have previous values
            directional_metrics = {}
            if test_data_orig is not None and 'prev_close' in test_data_orig.columns:
                prev_values = test_data_orig['prev_close'].values
                directional_metrics = calculate_directional_metrics(
                    y_test_orig, y_pred_orig, prev_values
                )
            
            # Calculate trading metrics if we have previous values
            trading_metrics = {}
            if test_data_orig is not None and 'prev_close' in test_data_orig.columns:
                prev_values = test_data_orig['prev_close'].values
                trading_metrics = calculate_trading_metrics(
                    y_test_orig, y_pred_orig, prev_values
                )
            
            # Combine all metrics
            all_metrics = {
                **regression_metrics,
                **directional_metrics,
                **trading_metrics
            }
            
            # Log metrics to MLflow
            with mlflow.start_run(experiment_id=self.experiment_id, run_name=f"evaluate_{int(time.time())}"):
                for name, value in all_metrics.items():
                    mlflow.log_metric(name, value)
                
                # Create evaluation plots
                self._create_evaluation_plots(y_test_orig, y_pred_orig, test_data_orig)
            
            # Log metrics to registry
            self.metrics_registry.log_metrics(
                all_metrics,
                model_name=self.model_name,
                model_version=self.model_version,
                dataset_name=self.config["dataset"],
                environment="development",
                tags=self.config.get("tags", [])
            )
            
            logger.info(f"Evaluation complete. Test RMSE: {regression_metrics['rmse']:.4f}")
            
            return {
                "regression_metrics": regression_metrics,
                "directional_metrics": directional_metrics,
                "trading_metrics": trading_metrics,
                "all_metrics": all_metrics
            }
            
        except Exception as e:
            logger.error(f"Error during evaluation: {str(e)}", exc_info=True)
            raise
    
    def _create_evaluation_plots(self, y_true: np.ndarray, y_pred: np.ndarray, test_data_orig: pd.DataFrame = None) -> None:
        """
        Create evaluation plots and log them as artifacts.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            test_data_orig: Original test data (optional)
        """
        try:
            import matplotlib.pyplot as plt
            
            # Create output directory for plots
            plots_dir = os.path.join(self.output_dir, "evaluation_plots")
            os.makedirs(plots_dir, exist_ok=True)
            
            # Predictions vs Actual plot
            plt.figure(figsize=(12, 6))
            plt.plot(y_true, label='Actual')
            plt.plot(y_pred, label='Predicted')
            plt.title('Predictions vs Actual Values')
            plt.xlabel('Time Step')
            plt.ylabel('Value')
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            
            pred_vs_actual_path = os.path.join(plots_dir, "predictions_vs_actual.png")
            plt.savefig(pred_vs_actual_path)
            mlflow.log_artifact(pred_vs_actual_path)
            
            # Scatter plot of Predicted vs Actual
            plt.figure(figsize=(8, 8))
            plt.scatter(y_true, y_pred, alpha=0.5)
            plt.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], 'r--')
            plt.title('Predicted vs Actual Values')
            plt.xlabel('Actual')
            plt.ylabel('Predicted')
            plt.grid(True)
            plt.tight_layout()
            
            scatter_path = os.path.join(plots_dir, "scatter_plot.png")
            plt.savefig(scatter_path)
            mlflow.log_artifact(scatter_path)
            
            # Error distribution plot
            errors = y_pred - y_true
            plt.figure(figsize=(10, 6))
            plt.hist(errors, bins=50, alpha=0.7)
            plt.title('Error Distribution')
            plt.xlabel('Prediction Error')
            plt.ylabel('Frequency')
            plt.grid(True)
            plt.tight_layout()
            
            error_dist_path = os.path.join(plots_dir, "error_distribution.png")
            plt.savefig(error_dist_path)
            mlflow.log_artifact(error_dist_path)
            
            # If we have original test data, create additional plots
            if test_data_orig is not None and 'timestamp' in test_data_orig.columns:
                # Time series plot with dates
                timestamps = test_data_orig['timestamp'].values
                
                plt.figure(figsize=(15, 7))
                plt.plot(timestamps, y_true, label='Actual')
                plt.plot(timestamps, y_pred, label='Predicted')
                plt.title('Predictions vs Actual Values Over Time')
                plt.xlabel('Date')
                plt.ylabel('Value')
                plt.legend()
                plt.grid(True)
                plt.xticks(rotation=45)
                plt.tight_layout()
                
                time_series_path = os.path.join(plots_dir, "time_series_plot.png")
                plt.savefig(time_series_path)
                mlflow.log_artifact(time_series_path)
            
            plt.close('all')
            
        except Exception as e:
            logger.warning(f"Error creating evaluation plots: {str(e)}")
    
    def hyperparameter_tuning(self, dataset_dict: Dict[str, np.ndarray], n_trials: int = 20) -> Dict[str, Any]:
        """
        Perform hyperparameter tuning using Optuna.
        
        Args:
            dataset_dict: Dictionary containing datasets
            n_trials: Number of hyperparameter optimization trials
            
        Returns:
            Dictionary with best hyperparameters and results
        """
        try:
            # Extract datasets
            X_train = dataset_dict["X_train"]
            y_train = dataset_dict["y_train"]
            X_val = dataset_dict["X_val"]
            y_val = dataset_dict["y_val"]
            
            # Set up hyperparameter search space based on model type
            model_type = self.config.get("model_type", "lstm")
            
            # Create Optuna study
            study_name = f"{self.model_name}_hparam_tuning_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            study = optuna.create_study(
                study_name=study_name,
                direction="minimize",
                sampler=TPESampler(seed=42),
                pruner=optuna.pruners.MedianPruner()
            )
            
            # Define objective function
            def objective(trial):
                # Generate hyperparameters based on model type
                hyperparams = self._generate_hyperparams(trial, model_type)
                
                # Create model with hyperparameters
                model = self.create_model(X_train.shape, hyperparams)
                
                # Train with early stopping
                early_stopping = EarlyStopping(
                    monitor="val_loss",
                    patience=hyperparams.get("early_stopping_patience", 10),
                    restore_best_weights=True
                )
                
                pruning_callback = TFKerasPruningCallback(trial, "val_loss")
                
                history = model.fit(
                    X_train, y_train,
                    validation_data=(X_val, y_val),
                    epochs=hyperparams.get("epochs", 30),
                    batch_size=hyperparams.get("batch_size", 32),
                    callbacks=[early_stopping, pruning_callback],
                    verbose=0
                )
                
                # Return final validation loss
                return history.history["val_loss"][-1]
            
            # Run optimization
            logger.info(f"Starting hyperparameter optimization with {n_trials} trials")
            study.optimize(objective, n_trials=n_trials)
            
            # Get best hyperparameters
            best_params = study.best_params
            best_value = study.best_value
            
            # Log results to MLflow
            with mlflow.start_run(experiment_id=self.experiment_id, run_name=f"hparam_tuning_{int(time.time())}"):
                mlflow.log_params(best_params)
                mlflow.log_metric("best_val_loss", best_value)
                
                # Log optimization history
                trials_df = study.trials_dataframe()
                trials_df.to_csv(os.path.join(self.output_dir, "hparam_trials.csv"), index=False)
                mlflow.log_artifact(os.path.join(self.output_dir, "hparam_trials.csv"))
                
                # Create and log optimization plots
                self._log_optuna_plots(study)
            
            logger.info(f"Hyperparameter optimization complete. Best val_loss: {best_value:.4f}")
            
            # Save best hyperparameters
            self.best_params = best_params
            
            # Return results
            return {
                "best_params": best_params,
                "best_value": best_value,
                "study": study
            }
            
        except Exception as e:
            logger.error(f"Error during hyperparameter tuning: {str(e)}", exc_info=True)
            raise
    
    def _generate_hyperparams(self, trial: optuna.Trial, model_type: str) -> Dict[str, Any]:
        """
        Generate hyperparameters for trial based on model type.
        
        Args:
            trial: Optuna trial object
            model_type: Type of model
            
        Returns:
            Hyperparameters dictionary
        """
        if model_type.lower() == "lstm":
            return {
                # Architecture hyperparameters
                "units": [
                    trial.suggest_int("lstm_units_1", 32, 256),
                    trial.suggest_int("lstm_units_2", 16, 128)
                ],
                "dropout": trial.suggest_float("dropout", 0.0, 0.5),
                "recurrent_dropout": trial.suggest_float("recurrent_dropout", 0.0, 0.5),
                "dense_units": trial.suggest_int("dense_units", 8, 64),
                "activation": trial.suggest_categorical("activation", ["relu", "tanh"]),
                
                # Training hyperparameters
                "learning_rate": trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True),
                "batch_size": trial.suggest_categorical("batch_size", [16, 32, 64, 128]),
                "epochs": 100,  # Will use early stopping
                "early_stopping_patience": trial.suggest_int("early_stopping_patience", 5, 20),
                
                # Loss function
                "loss": trial.suggest_categorical("loss", ["mse", "mae", "huber_loss"])
            }
        elif model_type.lower() == "gru":
            return {
                # Architecture hyperparameters
                "units": [
                    trial.suggest_int("gru_units_1", 32, 256),
                    trial.suggest_int("gru_units_2", 16, 128)
                ],
                "dropout": trial.suggest_float("dropout", 0.0, 0.5),
                "recurrent_dropout": trial.suggest_float("recurrent_dropout", 0.0, 0.5),
                "dense_units": trial.suggest_int("dense_units", 8, 64),
                "activation": trial.suggest_categorical("activation", ["relu", "tanh"]),
                
                # Training hyperparameters
                "learning_rate": trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True),
                "batch_size": trial.suggest_categorical("batch_size", [16, 32, 64, 128]),
                "epochs": 100,  # Will use early stopping
                "early_stopping_patience": trial.suggest_int("early_stopping_patience", 5, 20),
                
                # Loss function
                "loss": trial.suggest_categorical("loss", ["mse", "mae", "huber_loss"])
            }
        elif model_type.lower() == "cnn_lstm":
            return {
                # CNN hyperparameters
                "conv_filters": [
                    trial.suggest_int("conv_filters_1", 32, 128),
                    trial.suggest_int("conv_filters_2", 64, 256)
                ],
                "conv_kernel_size": trial.suggest_int("conv_kernel_size", 2, 5),
                
                # LSTM hyperparameters
                "lstm_units": [
                    trial.suggest_int("lstm_units_1", 32, 256),
                    trial.suggest_int("lstm_units_2", 16, 128)
                ],
                "dropout": trial.suggest_float("dropout", 0.0, 0.5),
                "dense_units": trial.suggest_int("dense_units", 8, 64),
                "activation": trial.suggest_categorical("activation", ["relu", "tanh"]),
                
                # Training hyperparameters
                "learning_rate": trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True),
                "batch_size": trial.suggest_categorical("batch_size", [16, 32, 64, 128]),
                "epochs": 100,  # Will use early stopping
                "early_stopping_patience": trial.suggest_int("early_stopping_patience", 5, 20),
                
                # Loss function
                "loss": trial.suggest_categorical("loss", ["mse", "mae", "huber_loss"])
            }
        elif model_type.lower() == "transformer":
            return {
                # Transformer hyperparameters
                "head_size": trial.suggest_int("head_size", 16, 64),
                "num_heads": trial.suggest_int("num_heads", 2, 8),
                "ff_dim": trial.suggest_int("ff_dim", 32, 256),
                "num_transformer_blocks": trial.suggest_int("num_transformer_blocks", 1, 4),
                "mlp_units": [
                    trial.suggest_int("mlp_units_1", 32, 256),
                    trial.suggest_int("mlp_units_2", 16, 128)
                ],
                "dropout": trial.suggest_float("dropout", 0.0, 0.5),
                
                # Training hyperparameters
                "learning_rate": trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True),
                "batch_size": trial.suggest_categorical("batch_size", [16, 32, 64, 128]),
                "epochs": 100,  # Will use early stopping
                "early_stopping_patience": trial.suggest_int("early_stopping_patience", 5, 20),
                
                # Loss function
                "loss": trial.suggest_categorical("loss", ["mse", "mae", "huber_loss"])
            }
        else:
            raise ValueError(f"Unsupported model type for hyperparameter tuning: {model_type}")
    
    def _log_optuna_plots(self, study: optuna.Study) -> None:
        """
        Create and log Optuna visualization plots.
        
        Args:
            study: Optuna study object
        """
        try:
            from optuna.visualization import plot_optimization_history, plot_param_importances, plot_contour
            import matplotlib.pyplot as plt
            
            # Create plots directory
            plots_dir = os.path.join(self.output_dir, "optuna_plots")
            os.makedirs(plots_dir, exist_ok=True)
            
            # Optimization history plot
            fig = plot_optimization_history(study)
            history_path = os.path.join(plots_dir, "optimization_history.png")
            fig.write_image(history_path)
            mlflow.log_artifact(history_path)
            
            # Parameter importance
            fig = plot_param_importances(study)
            importance_path = os.path.join(plots_dir, "param_importance.png")
            fig.write_image(importance_path)
            mlflow.log_artifact(importance_path)
            
            # If there are at least two parameters, create contour plots
            params = list(study.best_params.keys())
            if len(params) >= 2:
                for i in range(min(3, len(params) - 1)):  # Create up to 3 contour plots
                    param1 = params[i]
                    param2 = params[i + 1]
                    
                    fig = plot_contour(study, params=[param1, param2])
                    contour_path = os.path.join(plots_dir, f"contour_{param1}_{param2}.png")
                    fig.write_image(contour_path)
                    mlflow.log_artifact(contour_path)
            
        except Exception as e:
            logger.warning(f"Error creating Optuna plots: {str(e)}")
    
    def save_model_package(self) -> str:
        """
        Save model and metadata for deployment.
        
        Returns:
            Path to saved model package
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet")
            
        try:
            # Create package directory
            package_dir = os.path.join(self.output_dir, "package")
            os.makedirs(package_dir, exist_ok=True)
            
            # Save model
            self.model.save(os.path.join(package_dir, "model.h5"))
            
            # Save data processor
            if self.data_processor:
                with open(os.path.join(package_dir, "data_processor.pkl"), 'wb') as f:
                    pickle.dump(self.data_processor, f)
            
            # Save feature transformer
            if self.feature_transformer:
                with open(os.path.join(package_dir, "feature_transformer.pkl"), 'wb') as f:
                    pickle.dump(self.feature_transformer, f)
            
            # Save configuration and metadata
            config = {
                "model_name": self.model_name,
                "model_version": self.model_version,
                "model_type": self.config.get("model_type", "lstm"),
                "hyperparameters": self.best_params or self.config.get("hyperparameters", {}),
                "features": self.config.get("feature_columns", []),
                "target": self.config.get("target_column", "close"),
                "lookback_window": self.config.get("lookback_window", 24),
                "forecast_horizon": self.config.get("forecast_horizon", 1),
                "scaling_method": self.config.get("scaling_method", "standard"),
                "created_at": datetime.now().isoformat(),
                "created_by": self.config.get("author", "Minos-AI")
            }
            
            with open(os.path.join(package_dir, "config.json"), 'w') as f:
                json.dump(config, f, indent=2)
            
            # Create archive
            archive_path = os.path.join(self.output_dir, f"{self.model_name}_v{self.model_version}.zip")
            
            import shutil
            shutil.make_archive(
                os.path.splitext(archive_path)[0],
                'zip',
                package_dir
            )
            
            logger.info(f"Model package saved to {archive_path}")
            
            return archive_path
            
        except Exception as e:
            logger.error(f"Error saving model package: {str(e)}", exc_info=True)
            raise
    
    async def run_pipeline(self) -> Dict[str, Any]:
        """
        Run complete training pipeline from data loading to model package.
        
        Returns:
            Dictionary with pipeline results
        """
        try:
            # 1. Load data
            logger.info("Step 1: Loading data")
            data, metadata = await self.load_data()
            
            # 2. Preprocess data
            logger.info("Step 2: Preprocessing data")
            dataset_dict, preprocessing_info = self.preprocess_data(data)
            
            # 3. Hyperparameter tuning (if enabled)
            if self.config.get("hyperparameter_tuning", False):
                logger.info("Step 3: Hyperparameter tuning")
                tuning_results = self.hyperparameter_tuning(
                    dataset_dict,
                    n_trials=self.config.get("tuning_trials", 20)
                )
                best_params = tuning_results["best_params"]
            else:
                best_params = self.config.get("hyperparameters", {})
            
            # 4. Train model
            logger.info("Step 4: Training model")
            training_results = self.train(dataset_dict, best_params)
            
            # 5. Evaluate model
            logger.info("Step 5: Evaluating model")
            evaluation_results = self.evaluate(dataset_dict)
            
            # 6. Save model package
            logger.info("Step 6: Saving model package")
            package_path = self.save_model_package()
            
            # Return pipeline results
            return {
                "model_name": self.model_name,
                "model_version": self.model_version,
                "metadata": metadata,
                "preprocessing_info": preprocessing_info,
                "training_results": training_results,
                "evaluation_results": evaluation_results,
                "package_path": package_path
            }
            
        except Exception as e:
            logger.error(f"Error running pipeline: {str(e)}", exc_info=True)
            raise


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configuration dictionary
    """
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    except Exception as e:
        logger.error(f"Error loading configuration: {str(e)}")
        raise


async def main():
    """Main entry point for the training script."""
    parser = argparse.ArgumentParser(description="Train ML model for Minos-AI DeFi strategy platform")
    
    # Model and dataset parameters
    parser.add_argument("--model", required=True, help="Model name")
    parser.add_argument("--dataset", required=True, help="Dataset name (format: <asset>_<interval>)")
    parser.add_argument("--config", default=DEFAULT_CONFIG_PATH, help="Path to configuration file")
    
    # Training parameters
    parser.add_argument("--lookback", type=int, help="Lookback window size")
    parser.add_argument("--horizon", type=int, help="Forecast horizon")
    parser.add_argument("--features", help="Comma-separated list of features")
    parser.add_argument("--days", type=int, help="Number of days to use for training")
    
    # Hyperparameter tuning
    parser.add_argument("--tune", action="store_true", help="Perform hyperparameter tuning")
    parser.add_argument("--trials", type=int, default=20, help="Number of hyperparameter tuning trials")
    
    # Parse arguments
    args = parser.parse_args()
    
    try:
        # Load configuration
        config = load_config(args.config)
        
        # Override with command line arguments
        config["model_name"] = args.model
        config["dataset"] = args.dataset
        
        if args.lookback:
            config["lookback_window"] = args.lookback
            
        if args.horizon:
            config["forecast_horizon"] = args.horizon
            
        if args.features:
            config["feature_columns"] = args.features.split(",")
            
        if args.days:
            config["days_lookback"] = args.days
            
        if args.tune:
            config["hyperparameter_tuning"] = True
            config["tuning_trials"] = args.trials
        
        # Initialize and run pipeline
        pipeline = TrainingPipeline(config)
        results = await pipeline.run_pipeline()
        
        logger.info(f"Training pipeline completed successfully for {args.model}")
        logger.info(f"Model package saved to: {results['package_path']}")
        
        # Print key metrics
        if 'evaluation_results' in results and 'regression_metrics' in results['evaluation_results']:
            metrics = results['evaluation_results']['regression_metrics']
            logger.info(f"Test RMSE: {metrics.get('rmse', 'N/A')}")
            logger.info(f"Test MAE: {metrics.get('mae', 'N/A')}")
            logger.info(f"Test R²: {metrics.get('r2', 'N/A')}")
            
        return results
        
    except Exception as e:
        logger.error(f"Error in training pipeline: {str(e)}", exc_info=True)
        raise


if __name__ == "__main__":
    try:
        # Set up GPU memory growth to avoid OOM errors
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logger.info(f"Found {len(gpus)} GPU(s), enabled memory growth")
        
        # Run async main function
        import asyncio
        results = asyncio.run(main())
        
        # Exit with success
        sys.exit(0)
        
    except Exception as e:
        logger.error(f"Unhandled exception: {str(e)}", exc_info=True)
        sys.exit(1)
"""
Training Pipeline for Minos-AI DeFi Strategy Platform

This module implements the model training pipeline for the Minos-AI platform.
It coordinates the end-to-end process from data loading to model deployment:
- Loads and preprocesses data from various sources
- Configures and initializes models
- Executes training with hyperparameter optimization
- Evaluates model performance on validation and test data
- Tracks experiments and logs metrics
- Exports trained models for deployment
- Handles distributed training on multi-GPU/multi-node setups

Integration Points:
- Uses data_loader.py for accessing financial data
- Uses metrics.py for evaluating model performance
- Integrates with MLflow for experiment tracking
- Uses Optuna for hyperparameter optimization
- Outputs models compatible with the prediction service

Usage:
    python -m training.train --model price_prediction \\
        --dataset sol_usdc_1h \\
        --features price,volume,liquidity \\
        --horizon 24 \\
        --lookback 48

Author: Minos-AI Team
"""

import os
import json
import logging
import argparse
import yaml
import time
import hashlib
import pickle
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple, Any, Callable

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import mlflow
import optuna
from optuna.samplers import TPESampler
from optuna.integration import TFKerasPruningCallback, PyTorchLightningPruningCallback
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from tqdm import tqdm

# Add parent directory to path
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import project modules
from utils.data_loader import create_data_loader, DataSource, SamplingFrequency, DataFormat
from utils.metrics import (
    calculate_regression_metrics, 
    calculate_directional_metrics,
    calculate_trading_metrics,
    evaluate_model,
    generate_metrics_report,
    MetricsRegistry
)
from training.dataset import (
    prepare_time_series_dataset,
    create_sliding_window_dataset,
    SequenceDataset,
    TimeSeriesTransformer,
    FeatureEngineering
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("train.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Constants
DEFAULT_CONFIG_PATH = "config/training_config.yaml"
MODELS_DIR = "models"
DATA_CACHE_DIR = "data/cache"
EXPERIMENT_DIR = "experiments"
MAX_CACHE_AGE_DAYS = 7


class TrainingPipeline:
    """
    Main training pipeline class that orchestrates the entire training process.
    
    This class provides a standardized workflow for training models with
    configuration management, experiment tracking, and evaluation.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize training pipeline with configuration.
        
        Args:
            config: Configuration dictionary with training parameters
        """
        self.config = config
        self.model_name = config.get("model_name", "unknown_model")
        self.model_version = config.get("model_version", datetime.now().strftime("%Y%m%d_%H%M%S"))
        self.experiment_name = config.get("experiment_name", f"{self.model_name}_{self.model_version}")
        
        # Initialize directories
        self.output_dir = os.path.join(MODELS_DIR, self.model_name, self.model_version)
        self.cache_dir = os.path.join(DATA_CACHE_DIR, self.model_name)
        
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Initialize components
        self._init_mlflow()
        self._init_metrics_registry()
        
        # For storing the trained model
        self.model = None
        self.best_params = None
        self.data_processor = None
        self.feature_transformer = None
        
        logger.info(f"Initialized training pipeline for {self.model_name} (version: {self.model_version})")
        
    def _init_mlflow(self):
        """Initialize MLflow experiment tracking."""
        mlflow_uri = self.config.get("mlflow", {}).get("tracking_uri")
        if mlflow_uri:
            mlflow.set_tracking_uri(mlflow_uri)
            
        # Set or create experiment
        experiment = mlflow.get_experiment_by_name(self.experiment_name)
        if experiment:
            self.experiment_id = experiment.experiment_id
        else:
            self.experiment_id = mlflow.create_experiment(
                self.experiment_name,
                artifact_location=os.path.join(EXPERIMENT_DIR, self.experiment_name)
            )
            
        logger.info(f"MLflow experiment: {self.experiment_name} (ID: {self.experiment_id})")
        
    def _init_metrics_registry(self):
        """Initialize metrics registry for tracking performance."""
        metrics_path = os.path.join(MODELS_DIR, "metrics_registry.json")
        self.metrics_registry = MetricsRegistry(storage_path=metrics_path)
    
    async def load_data(self) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Load data for training using data_loader.
        
        Returns:
            Tuple of (dataframe, metadata)
        """
        try:
            cache_file = os.path.join(
                self.cache_dir, 
                f"{self.config['dataset']}_{self.config.get('start_date', 'all')}.pkl"
            )
            
            # Check if cached data exists and is recent enough
            if os.path.exists(cache_file):
                cache_age = datetime.now() - datetime.fromtimestamp(os.path.getmtime(cache_file))
                if cache_age.days < MAX_CACHE_AGE_DAYS:
                    logger.info(f"Loading data from cache: {cache_file}")
                    with open(cache_file, 'rb') as f:
                        return pickle.load(f)
            
            # Data parameters
            dataset = self.config["dataset"]  # e.g., "sol_usdc_1h"
            parts = dataset.split("_")
            
            if len(parts) < 2:
                raise ValueError(f"Invalid dataset format: {dataset}. Expected format: <asset>_<interval>")
                
            asset = parts[0] + "/" + parts[1]  # e.g., "sol/usdc"
            interval = SamplingFrequency(parts[2])  # e.g., "1h"
            
            # Time range parameters
            days_lookback = self.config.get("days_lookback", 365)
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days_lookback)
            
            if "start_date" in self.config:
                start_date = datetime.fromisoformat(self.config["start_date"])
                
            if "end_date" in self.config:
                end_date = datetime.fromisoformat(self.config["end_date"])
                
            logger.info(f"Loading data for {asset} from {start_date} to {end_date} with interval {interval.value}")
            
            # Create data loader
            data_loader = await create_data_loader(
                cache_size_mb=self.config.get("cache_size_mb", 4096),
                api_keys_file=self.config.get("api_keys_file"),
                data_warehouse_url=self.config.get("data_warehouse_url")
            )
            
            # Get price data with indicators
            price_data = await data_loader.get_price_data(
                asset=asset,
                start_time=start_date,
                end_time=end_date,
                interval=interval,
                include_indicators=True,
                source=DataSource[self.config.get("data_source", "DATA_WAREHOUSE")]
            )
            
            # Additional data sources based on configuration
            data_sources = self.config.get("data_sources", [])
            
            # On-chain metrics
            if "on_chain" in data_sources:
                token = asset.split("/")[0]  # e.g., "sol" from "sol/usdc"
                onchain_data = await data_loader.get_on_chain_metrics(
                    asset=token,
                    start_time=start_date,
                    end_time=end_date,
                    interval=interval
                )
                
                # Merge with price data
                onchain_data.set_index("timestamp", inplace=True)
                price_data.set_index("timestamp", inplace=True)
                price_data = price_data.join(onchain_data, how="left")
                price_data.reset_index(inplace=True)
                
            # Sentiment data
            if "sentiment" in data_sources:
                token = asset.split("/")[0]  # e.g., "sol" from "sol/usdc"
                sentiment_data = await data_loader.get_sentiment_data(
                    asset=token,
                    start_time=start_date,
                    end_time=end_date,
                    interval=interval
                )
                
                # Merge with price data
                sentiment_data.set_index("timestamp", inplace=True)
                if "timestamp" not in price_data.columns:
                    price_data.set_index("timestamp", inplace=True)
                    price_data = price_data.join(sentiment_data, how="left")
                    price_data.reset_index(inplace=True)
                else:
                    price_data.set_index("timestamp", inplace=True)
                    price_data = price_data.join(sentiment_data, how="left")
                    price_data.reset_index(inplace=True)
                    
            # Market comparison data
            if "market_comparison" in data_sources and "comparison_assets" in self.config:
                comparison_assets = self.config["comparison_assets"]
                metric = self.config.get("comparison_metric", "price")
                
                comparison_data = await data_loader.get_multiple_assets_data(
                    assets=comparison_assets,
                    metric=metric,
                    start_time=start_date,
                    end_time=end_date,
                    interval=interval
                )
                
                # Merge with price data
                comparison_data.set_index("timestamp", inplace=True)
                if "timestamp" not in price_data.columns:
                    price_data.set_index("timestamp", inplace=True)
                    price_data = price_data.join(comparison_data, how="left")
                    price_data.reset_index(inplace=True)
                else:
                    price_data.set_index("timestamp", inplace=True)
                    price_data = price_data.join(comparison_data, how="left")
                    price_data.reset_index(inplace=True)
            
            # Clean up and finalize dataset
            
            # Fill missing values with forward fill then backward fill
            price_data = price_data.ffill().bfill()
            
            # Remove any remaining rows with NaN values
            price_data.dropna(inplace=True)
            
            # Create metadata
            metadata = {
                "asset": asset,
                "interval": interval.value,
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat(),
                "data_sources": data_sources,
                "columns": list(price_data.columns),
                "rows": len(price_data)
            }
            
            # Cache the data
            with open(cache_file, 'wb') as f:
                pickle.dump((price_data, metadata), f)
            
            logger.info(f"Loaded {len(price_data)} data points with {len(price_data.columns)} features")
            
            # Close data loader
            await data_loader.close()
            
            return price_data, metadata
            
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}", exc_info=True)
            raise
    
    def preprocess_data(self, data: pd.DataFrame) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        """
        Preprocess data for model training.
        
        Args:
            data: Raw data from load_data
            
        Returns:
            Tuple of (dataset_dict, preprocessing_info)
        """
        try:
            # Extract configuration
            target_column = self.config.get("target_column", "close")
            feature_columns = self.config.get("feature_columns")
            lookback_window = self.config.get("lookback_window", 24)
            forecast_horizon = self.config.get("forecast_horizon", 1)
            test_size = self.config.get("test_size", 0.2)
            val_size = self.config.get("val_size", 0.15)
            scaling_method = self.config.get("scaling_method", "standard")
            
            # Set default feature columns if not provided
            if not feature_columns:
                # Use basic OHLCV + some indicators if available
                feature_columns = ["open", "high", "low", "close", "volume"]
                for col in ["rsi_14", "sma_7", "sma_25", "macd", "bb_upper", "bb_lower"]:
                    if col in data.columns:
                        feature_columns.append(col)
                        
            logger.info(f"Preprocessing data with {len(feature_columns)} features: {feature_columns}")
            
            # Create feature engineering transformer
            self.feature_transformer = FeatureEngineering(
                feature_columns=feature_columns,
                target_column=target_column,
                config=self.config.get("feature_engineering", {})
            )
            
            # Apply feature engineering
            data = self.feature_transformer.transform(data)
            
            # Prepare time series dataset
            dataset_creator = TimeSeriesTransformer(
                lookback_window=lookback_window,
                forecast_horizon=forecast_horizon,
                scaling_method=scaling_method
            )
            
            # Split into train/val/test sets
            train_data, test_data = train_test_split(data, test_size=test_size, shuffle=False)
            train_data, val_data = train_test_split(train_data, test_size=val_size/(1-test_size), shuffle=False)
            
            # Store the data processor for later use
            self.data_processor = dataset_creator
            
            # Create training datasets
            X_train, y_train, X_val, y_val, X_test, y_test = dataset_creator.create_train_val_test_datasets(
                train_data, val_data, test_data
            )
            
            # Prepare dataset dictionary
            dataset_dict = {
                "X_train": X_train,
                "y_train": y_train,
                "X_val": X_val,
                "y_val": y_val,
                "X_test": X_test,
                "y_test": y_test
            }
            
            # Create preprocessing info for metadata
            preprocessing_info = {
                "target_column": target_column,
                "feature_columns": feature_columns,
                "lookback_window": lookback_window,
                "forecast_horizon": forecast_horizon,
                "train_samples": len(X_train),
                "val_samples": len(X_val),
                "test_samples": len(X_test),
                "scaling_method": scaling_method,
                "feature_engineering": self.feature_transformer.get_config()
            }
            
            logger.info(f"Preprocessing complete: {preprocessing_info}")
            
            return dataset_dict, preprocessing_info
            
        except Exception as e:
            logger.error(f"Error preprocessing data: {str(e)}", exc_info=True)
            raise
    
    def create_model(self, input_shape: Tuple[int, ...], hyperparams: Dict[str, Any] = None) -> Any:
        """
        Create model based on configuration.
        
        Args:
            input_shape: Shape of input tensor
            hyperparams: Optional hyperparameters dictionary
            
        Returns:
            Model instance
        """
        # Get model type from config
        model_type = self.config.get("model_type", "lstm")
        
        # If hyperparams not provided, use config defaults
        if hyperparams is None:
            hyperparams = self.config.get("hyperparameters", {})
            
        # Create model based on type
        if model_type.lower() == "lstm":
            return self._create_lstm_model(input_shape, hyperparams)
        elif model_type.lower() == "gru":
            return self._create_gru_model(input_shape, hyperparams)
        elif model_type.lower() == "cnn_lstm":
            return self._create_cnn_lstm_model(input_shape, hyperparams)
        elif model_type.lower() == "transformer":
            return self._create_transformer_model(input_shape, hyperparams)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
    
    def _create_lstm_model(self, input_shape: Tuple[int, ...], hyperparams: Dict[str, Any]) -> tf.keras.Model:
        """
        Create LSTM model with specified hyperparameters.
        
        Args:
            input_shape: Shape of input tensor
            hyperparams: Hyperparameters dictionary
            
        Returns:
            Configured TensorFlow Keras model
        """
        # Extract hyperparameters with defaults
        units = hyperparams.get("units", [64, 32])
        dropout = hyperparams.get("dropout", 0.2)
        recurrent_dropout = hyperparams.get("recurrent_dropout", 0.0)
        learning_rate = hyperparams.get("learning_rate", 0.001)
        activation = hyperparams.get("activation", "relu")
        
        # Build model
        model = tf.keras.Sequential()
        
        # Add LSTM layers
        for i, n_units in enumerate(units):
            return_sequences = i < len(units) - 1  # Return sequences for all except last layer
            
            if i == 0:
                model.add(tf.keras.layers.LSTM(
                    units=n_units,
                    activation=activation,
                    return_sequences=return_sequences,
                    dropout=dropout,
                    recurrent_dropout=recurrent_dropout,
                    input_shape=input_shape[1:]  # Skip batch dimension
                ))
            else:
                model.add(tf.keras.layers.LSTM(
                    units=n_units,
                    activation=activation,
                    return_sequences=return_sequences,
                    dropout=dropout,
                    recurrent_dropout=recurrent_dropout
                ))
        
        # Add dense output layer(s)
        model.add(tf.keras.layers.Dense(units=hyperparams.get("dense_units", 16), activation=activation))
        model.add(tf.keras.layers.Dense(units=1))  # Output layer
        
        # Compile model
        optimizer = Adam(learning_rate=learning_rate)
        model.compile(
            optimizer=optimizer,
            loss=hyperparams.get("loss", "mse"),
            metrics=["mae", "mse"]
        )
        
        return model
    
    def _create_gru_model(self, input_shape: Tuple[int, ...], hyperparams: Dict[str, Any]) -> tf.keras.Model:
        """
        Create GRU model with specified hyperparameters.
        
        Args:
            input_shape: Shape of input tensor
            hyperparams: Hyperparameters dictionary
            
        Returns:
            Configured TensorFlow Keras model
        """
        # Extract hyperparameters with defaults
        units = hyperparams.get("units", [64, 32])
        dropout = hyperparams.get("dropout", 0.2)
        recurrent_dropout = hyperparams.get("recurrent_dropout", 0.0)
        learning_rate = hyperparams.get("learning_rate", 0.001)
        activation = hyperparams.get("activation", "relu")
        
        # Build model
        model = tf.keras.Sequential()
        
        # Add GRU layers
        for i, n_units in enumerate(units):
            return_sequences = i < len(units) - 1  # Return sequences for all except last layer
            
            if i == 0:
                model.add(tf.keras.layers.GRU(
                    units=n_units,
                    activation=activation,
                    return_sequences=return_sequences,
                    dropout=dropout,
                    recurrent_dropout=recurrent_dropout,
                    input_shape=input_shape[1:]  # Skip batch dimension
                ))
            else:
                model.add(tf.keras.layers.GRU(
                    units=n_units,
                    activation=activation,
                    return_sequences=return_sequences,
                    dropout=dropout,
                    recurrent_dropout=recurrent_dropout
                ))
        
        # Add dense output layer(s)
        model.add(tf.keras.layers.Dense(units=hyperparams.get("dense_units", 16), activation=activation))
        model.add(tf.keras.layers.Dense(units=1))  # Output layer
        
        # Compile model
        optimizer = Adam(learning_rate=learning_rate)
        model.compile(
            optimizer=optimizer,
            loss=hyperparams.get("loss", "mse"),
            metrics=["mae", "mse"]
        )
        
        return model
    
    def _create_cnn_lstm_model(self, input_shape: Tuple[int, ...], hyperparams: Dict[str, Any]) -> tf.keras.Model:
        """
        Create CNN-LSTM hybrid model with specified hyperparameters.
        
        Args:
            input_shape: Shape of input tensor
            hyperparams: Hyperparameters dictionary
            
        Returns:
            Configured TensorFlow Keras model
        """
        # Extract hyperparameters with defaults
        conv_filters = hyperparams.get("conv_filters", [64, 128])
        conv_kernel_size = hyperparams.get("conv_kernel_size", 3)
        lstm_units = hyperparams.get("lstm_units", [64, 32])
        dropout = hyperparams.get("dropout", 0.2)
        learning_rate = hyperparams.get("learning_rate", 0.001)
        activation = hyperparams.get("activation", "relu")
        
        # Build model
        model = tf.keras.Sequential()
        
        # Add convolutional layers for feature extraction
        for i, filters in enumerate(conv_filters):
            if i == 0:
                model.add(tf.keras.layers.Conv1D(
                    filters=filters,
                    kernel_size=conv_kernel_size,
                    activation=activation,
                    padding='same',
                    input_shape=input_shape[1:]  # Skip batch dimension
                ))
            else:
                model.add(tf.keras.layers.Conv1D(
                    filters=filters,
                    kernel_size=conv_kernel_size,
                    activation=activation,
                    padding='same'
                ))
                
            # Add pooling after each conv layer
            model.add(tf.keras.layers.MaxPooling1D(pool_size=2))
        
        # Add LSTM layers
        for i, n_units in enumerate(lstm_units):
            return_sequences = i < len(lstm_units) - 1  # Return sequences for all except last layer
            model.add(tf.keras.layers.LSTM(
                units=n_units,
                return_sequences=return_sequences,
                dropout=dropout
            ))
        
        # Add dense output layer(s)
        model.add(tf.keras.layers.Dense(units=hyperparams.get("dense_units", 16), activation=activation))
        model.add(tf.keras.layers.Dense(units=1))  # Output layer
        
        # Compile model
        optimizer = Adam(learning_rate=learning_rate)
        model.compile(
            optimizer=optimizer,
            loss=hyperparams.get("loss", "mse"),
            metrics=["mae", "mse"]
        )
        
        return model
    
    def _create_transformer_model(self, input_shape: Tuple[int, ...], hyperparams: Dict[str, Any]) -> tf.keras.Model:
        """
        Create Transformer model with specified hyperparameters.
        
        Args:
            input_shape: Shape of input tensor
            hyperparams: Hyperparameters dictionary
            
        Returns:
            Configured TensorFlow Keras model
        """
        # Extract hyperparameters with defaults
        head_size = hyperparams.get("head_size", 256)
        num_heads = hyperparams.get("num_heads", 4)
        ff_dim = hyperparams.get("ff_dim", 256)
        num_transformer_blocks = hyperparams.get("num_transformer_blocks", 2)
        mlp_units = hyperparams.get("mlp_units", [128, 64])
        dropout = hyperparams.get("dropout", 0.2)
        learning_rate = hyperparams.get("learning_rate", 0.001)
        
        # Define input
        inputs = tf.keras.layers.Input(shape=input_shape[1:])  # Skip batch dimension
        
        # Create positional encoding
        positions = tf.range(start=0, limit=input_shape[1], delta=1)
        pos_encoding = self._positional_encoding(positions, input_shape[2])
        
        # Add positional encoding
        x = inputs + pos_encoding
        
        # Transformer blocks
        for _ in range(num_transformer_blocks):
            # Multi-head attention
            attention_output = tf.keras.layers.MultiHeadAttention(
                key_dim=head_size, num_heads=num_heads, dropout=dropout
            )(x, x)
            
            # Skip connection and layer normalization
            x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(attention_output + x)
            
            # Feed-forward network
            ffn_output = tf.keras.Sequential([
                tf.keras.layers.Dense(ff_dim, activation="relu"),
                tf.keras.layers.Dense(input_shape[2])
            ])(x)
            
            # Skip connection and layer normalization
            x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(ffn_output + x)
        
        # Global average pooling
        x = tf.keras.layers.GlobalAveragePooling1D()(x)
        
        # MLP head
        for dim in mlp_units:
            x = tf.keras.layers.Dense(dim, activation="relu")(x)
            x = tf.keras.layers.Dropout(dropout)(x)
            
        # Output layer
        outputs = tf.keras.layers.Dense(1)(x)
        
        # Build model
        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        
        # Compile model
        optimizer = Adam(learning_rate=learning_rate)
        model.compile(
            optimizer=optimizer,
            loss=hyperparams.get("loss", "mse"),
            metrics=["mae", "mse"]
        )
        
        return model
    
    def _positional_encoding(self, positions, d_model):
        """
        Create positional encoding for transformer models.
        
        Args:
            positions: Position indices
            d_model: Dimension of the model
            
        Returns:
            Positional encoding tensor
        """
        # Calculate positional encoding using sine and cosine functions
        angle_rads = self._get_angles(
            positions=tf.cast(positions, dtype=tf.float32)[:, tf.newaxis],
            i=tf.cast(tf.range(d_model), dtype=tf.float32)[tf.newaxis, :],
            d_model=d_model
        )
        
        # Apply sin to even indices
        sines = tf.math.sin(angle_rads[:, 0::2])
        
        # Apply cos to odd indices
        cosines = tf.math.cos(angle_rads[:, 1::2])
        
        # Combine sine and cosine values
        pos_encoding = tf.concat([sines, cosines], axis=-1)
        pos_encoding = pos_encoding[tf.newaxis, ...]
        
        return tf.cast(pos_encoding, dtype=tf.float32)
    
    def _get_angles(self, positions, i, d_model):
        """
        Helper function for positional encoding.
        
        Args:
            positions: Position indices
            i: Dimension indices
            d_model: Dimension of the model
            
        Returns:
            Angle radian values
        """
        angle_rates = 1 / tf.pow(10000, (2 * (i // 2)) / tf.cast(d_model, dtype=tf.float32))
        return positions * angle_rates
    
    def train(self, dataset_dict: Dict[str, np.ndarray], hyperparams: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Train model with the provided dataset.
        
        Args:
            dataset_dict: Dictionary containing training, validation, and test datasets
            hyperparams: Optional hyperparameters dictionary
            
        Returns:
            Dictionary of training results
        """
        try:
            # Extract datasets
            X_train = dataset_dict["X_train"]
            y_train = dataset_dict["y_train"]
            X_val = dataset_dict["X_val"]
            y_val = dataset_dict["y_val"]
            
            # If hyperparams not provided, use config defaults
            if hyperparams is None:
                hyperparams = self.config.get("hyperparameters", {})
                
            # Create model
            model = self.create_model(X_train.shape, hyperparams)
            
            # Set up callbacks
            callbacks = self._get_training_callbacks(hyperparams)
            
            # Start MLflow run
            with mlflow.start_run(experiment_id=self.experiment_id, run_name=f"train_{int(time.time())}"):
                # Log hyperparameters
                mlflow.log_params(hyperparams)
                
                # Train model
                start_time = time.time()
                history = model.fit(
                    X_train, y_train,
                    validation_data=(X_val, y_val),
                    epochs=hyperparams.get("epochs", 100),
                    batch_size=hyperparams.get("batch_size", 32),
                    callbacks=callbacks,
                    verbose=1
                )
                train_time = time.time