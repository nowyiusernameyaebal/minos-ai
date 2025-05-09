signals: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """
        Evaluate model performance against actual values.
        
        Args:
            actual_values: Actual target values
            predictions: Model predictions
            signals: Trading signals (if available)
            
        Returns:
            Dictionary of performance metrics
        """
        logger.info("Evaluating model performance")
        
        try:
            # Initialize metrics dictionary
            metrics = {}
            
            # Check if signal type is compatible with input
            if self.signal_type == "binary" or self.signal_type == "multi_class":
                # Classification metrics
                from sklearn.metrics import (
                    accuracy_score, precision_score, recall_score, f1_score,
                    roc_auc_score, confusion_matrix, classification_report
                )
                
                # Convert predictions to class labels if needed
                if len(predictions.shape) > 1 and predictions.shape[1] > 1:
                    # Multi-class probabilities, convert to class labels
                    pred_labels = np.argmax(predictions, axis=1)
                else:
                    # Binary probabilities, convert to 0/1
                    pred_labels = (predictions > 0.5).astype(int)
                
                # Calculate classification metrics
                metrics["accuracy"] = float(accuracy_score(actual_values, pred_labels))
                
                try:
                    # These might fail depending on the classes present
                    metrics["precision"] = float(precision_score(actual_values, pred_labels, average='weighted'))
                    metrics["recall"] = float(recall_score(actual_values, pred_labels, average='weighted'))
                    metrics["f1"] = float(f1_score(actual_values, pred_labels, average='weighted'))
                except Exception as e:
                    logger.warning(f"Error calculating precision/recall/f1: {e}")
                
                try:
                    # ROC AUC might fail for multi-class or if only one class is present
                    if self.signal_type == "binary":
                        metrics["roc_auc"] = float(roc_auc_score(actual_values, predictions))
                    else:
                        # One-vs-rest ROC AUC for multi-class
                        metrics["roc_auc"] = float(roc_auc_score(
                            actual_values, 
                            predictions, 
                            multi_class='ovr', 
                            average='weighted'
                        ))
                except Exception as e:
                    logger.warning(f"Error calculating ROC AUC: {e}")
                
                # Confusion matrix
                cm = confusion_matrix(actual_values, pred_labels)
                metrics["confusion_matrix"] = cm.tolist()
                
                # Classification report
                report = classification_report(actual_values, pred_labels, output_dict=True)
                metrics["classification_report"] = report
            
            elif self.signal_type == "regression":
                # Regression metrics
                from sklearn.metrics import (
                    mean_squared_error, mean_absolute_error, r2_score,
                    mean_absolute_percentage_error
                )
                
                # Calculate regression metrics
                metrics["mse"] = float(mean_squared_error(actual_values, predictions))
                metrics["rmse"] = float(np.sqrt(metrics["mse"]))
                metrics["mae"] = float(mean_absolute_error(actual_values, predictions))
                
                try:
                    metrics["mape"] = float(mean_absolute_percentage_error(actual_values, predictions))
                except Exception as e:
                    logger.warning(f"Error calculating MAPE: {e}")
                
                metrics["r2"] = float(r2_score(actual_values, predictions))
            
            # Calculate trading performance metrics if signals provided
            if signals is not None:
                metrics.update(self._calculate_trading_metrics(actual_values, signals))
            
            # Store metrics
            self.performance_metrics = {**self.performance_metrics, **metrics}
            
            logger.info(f"Performance evaluation complete: {metrics}")
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error evaluating performance: {e}")
            return {"error": str(e)}
    
    def _calculate_trading_metrics(
        self,
        price_changes: np.ndarray,
        signals: np.ndarray
    ) -> Dict[str, float]:
        """
        Calculate trading performance metrics.
        
        Args:
            price_changes: Actual price changes
            signals: Trading signals (-1, 0, 1)
            
        Returns:
            Dictionary of trading metrics
        """
        try:
            metrics = {}
            
            # Convert signals to expected returns
            # Signal 1 (buy) expects positive returns
            # Signal -1 (sell) expects negative returns
            # Signal 0 (hold) expects small or zero returns
            
            # Calculate signal accuracy
            correct_signals = (
                (signals == 1) & (price_changes > 0) |  # Correct buy
                (signals == -1) & (price_changes < 0) |  # Correct sell
                (signals == 0) & (np.abs(price_changes) < 0.01)  # Correct hold (small change)
            )
            
            metrics["signal_accuracy"] = float(np.mean(correct_signals))
            
            # Calculate profitability
            # Positive return for (buy & price up) or (sell & price down)
            # Negative return for (buy & price down) or (sell & price up)
            # Zero return for hold
            
            # For buys, return is the price change
            buy_returns = np.where(signals == 1, price_changes, 0)
            
            # For sells, return is the negative price change
            sell_returns = np.where(signals == -1, -price_changes, 0)
            
            # Total returns
            total_returns = buy_returns + sell_returns
            
            # Profitability metrics
            metrics["total_return"] = float(np.sum(total_returns))
            metrics["mean_return_per_trade"] = float(np.mean(total_returns[signals != 0]))
            metrics["win_rate"] = float(np.mean(total_returns > 0))
            
            # Calculate Sharpe ratio (annualized)
            # Assuming daily returns
            daily_returns = total_returns
            metrics["sharpe_ratio"] = float(
                np.mean(daily_returns) / (np.std(daily_returns) + 1e-10) * np.sqrt(252)
            )
            
            # Maximum drawdown
            cumulative_returns = np.cumsum(daily_returns)
            max_return = np.maximum.accumulate(cumulative_returns)
            drawdown = max_return - cumulative_returns
            metrics["max_drawdown"] = float(np.max(drawdown))
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating trading metrics: {e}")
            return {"trading_metrics_error": str(e)}
    
    def verify_signal_integrity(
        self,
        signal: Dict[str, Any],
        signature: str,
        public_key: str
    ) -> bool:
        """
        Verify the integrity of a trading signal.
        
        Args:
            signal: Trading signal data
            signature: Digital signature
            public_key: Public key to verify signature
            
        Returns:
            True if the signature is valid, False otherwise
        """
        try:
            # Convert signal to canonical string
            signal_str = json.dumps(signal, sort_keys=True)
            
            # Decode signature from base64
            import base64
            signature_bytes = base64.b64decode(signature)
            
            # Verify signature
            import cryptography.hazmat.primitives.asymmetric.padding as padding
            from cryptography.hazmat.primitives import hashes
            from cryptography.hazmat.primitives.asymmetric import padding, rsa
            from cryptography.hazmat.backends import default_backend
            
            # Load public key
            from cryptography.hazmat.primitives.serialization import load_pem_public_key
            public_key_obj = load_pem_public_key(
                public_key.encode(),
                backend=default_backend()
            )
            
            # Verify signature
            public_key_obj.verify(
                signature_bytes,
                signal_str.encode(),
                padding.PSS(
                    mgf=padding.MGF1(hashes.SHA256()),
                    salt_length=padding.PSS.MAX_LENGTH
                ),
                hashes.SHA256()
            )
            
            # If no exception raised, signature is valid
            return True
            
        except Exception as e:
            logger.error(f"Error verifying signal integrity: {e}")
            return False
    
    def export_artifacts(self, export_dir: str) -> Dict[str, str]:
        """
        Export all model artifacts for deployment.
        
        Args:
            export_dir: Directory to export artifacts to
            
        Returns:
            Dictionary mapping artifact names to paths
        """
        logger.info(f"Exporting model artifacts to {export_dir}")
        
        # Create export directory if it doesn't exist
        os.makedirs(export_dir, exist_ok=True)
        
        artifacts = {}
        
        try:
            # Export model
            if self.model_loaded:
                model_path = os.path.join(export_dir, "ariadne_model")
                self.model.save(model_path)
                artifacts["model"] = model_path + ".h5"
                
                # Export model config
                model_config_path = os.path.join(export_dir, "model_config.json")
                with open(model_config_path, "w") as f:
                    json.dump(self.model.config, f, indent=2)
                artifacts["model_config"] = model_config_path
            
            # Export preprocessor
            if self.preprocessor_loaded:
                # Export preprocessor config
                preprocessor_config_path = os.path.join(export_dir, "preprocessor_config.json")
                with open(preprocessor_config_path, "w") as f:
                    json.dump(self.preprocessor.config, f, indent=2)
                artifacts["preprocessor_config"] = preprocessor_config_path
                
                # Export feature scalers
                if hasattr(self.preprocessor, "feature_scalers"):
                    for name, scaler in self.preprocessor.feature_scalers.items():
                        scaler_path = os.path.join(export_dir, f"{name}_scaler.joblib")
                        joblib.dump(scaler, scaler_path)
                        artifacts[f"{name}_scaler"] = scaler_path
                
                # Export label scalers
                if hasattr(self.preprocessor, "label_scalers"):
                    for name, scaler in self.preprocessor.label_scalers.items():
                        scaler_path = os.path.join(export_dir, f"{name}_label_scaler.joblib")
                        joblib.dump(scaler, scaler_path)
                        artifacts[f"{name}_label_scaler"] = scaler_path
            
            # Export inference config
            inference_config_path = os.path.join(export_dir, "inference_config.json")
            with open(inference_config_path, "w") as f:
                json.dump(self.config, f, indent=2)
            artifacts["inference_config"] = inference_config_path
            
            # Export metadata
            metadata = {
                "version": self.INFERENCE_VERSION,
                "model_type": self.model.__class__.__name__ if self.model_loaded else "None",
                "preprocessor_type": self.preprocessor.__class__.__name__ if self.preprocessor_loaded else "None",
                "signal_type": self.signal_type,
                "prediction_mode": self.prediction_mode,
                "deployment_mode": self.deployment_mode,
                "export_timestamp": datetime.now().isoformat(),
                "model_hash": self._calculate_model_hash() if self.model_loaded else "None"
            }
            
            metadata_path = os.path.join(export_dir, "metadata.json")
            with open(metadata_path, "w") as f:
                json.dump(metadata, f, indent=2)
            artifacts["metadata"] = metadata_path
            
            logger.info(f"Exported {len(artifacts)} artifacts to {export_dir}")
            
            return artifacts
            
        except Exception as e:
            logger.error(f"Error exporting artifacts: {e}")
            raise
    
    def simulate_trading(
        self,
        price_data: pd.DataFrame,
        initial_capital: float = 10000.0,
        position_size: float = 0.1,
        fee_rate: float = 0.001,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Simulate trading using the model's signals.
        
        Args:
            price_data: DataFrame with price data
            initial_capital: Initial trading capital
            position_size: Size of each position as fraction of capital
            fee_rate: Trading fee rate
            stop_loss: Optional stop loss percentage
            take_profit: Optional take profit percentage
            
        Returns:
            Dictionary with simulation results
        """
        logger.info("Starting trading simulation")
        
        try:
            # Check if price data has required columns
            required_columns = ["open", "high", "low", "close"]
            missing_columns = [col for col in required_columns if col not in price_data.columns]
            
            if missing_columns:
                logger.error(f"Missing required columns in price data: {missing_columns}")
                raise ValueError(f"Missing required columns in price data: {missing_columns}")
            
            # Generate predictions for each time step
            predictions = []
            signals = []
            
            # Use sliding window approach
            sequence_length = self.preprocessor.sequence_length
            
            for i in range(sequence_length, len(price_data)):
                # Extract window
                window = price_data.iloc[i-sequence_length:i]
                
                # Generate prediction
                result = self.predict(window, include_confidence=True, generate_signals=True)
                
                # Store prediction and signal
                predictions.append(result["predictions"][-1])
                signals.append(result["signals"][-1])
            
            # Add predictions and signals to price data
            result_df = price_data.iloc[sequence_length:].copy()
            result_df["prediction"] = predictions
            result_df["signal"] = signals
            
            # Initialize simulation variables
            capital = initial_capital
            position = 0
            entry_price = 0
            trades = []
            
            # Simulation loop
            for i in range(len(result_df)):
                current_row = result_df.iloc[i]
                current_price = current_row["close"]
                current_signal = current_row["signal"]
                
                # Check stop loss and take profit
                if position != 0:
                    price_change = (current_price - entry_price) / entry_price
                    
                    # Stop loss check (for long positions)
                    if stop_loss is not None and position > 0 and price_change < -stop_loss:
                        # Execute stop loss
                        trade_value = position * current_price
                        fee = trade_value * fee_rate
                        capital += trade_value - fee
                        
                        # Record trade
                        trades.append({
                            "type": "stop_loss",
                            "side": "sell",
                            "price": current_price,
                            "quantity": position,
                            "value": trade_value,
                            "fee": fee,
                            "profit_loss": (current_price - entry_price) * position - fee,
                            "timestamp": current_row.name
                        })
                        
                        position = 0
                    
                    # Stop loss check (for short positions)
                    elif stop_loss is not None and position < 0 and price_change > stop_loss:
                        # Execute stop loss
                        trade_value = -position * current_price
                        fee = trade_value * fee_rate
                        capital += trade_value - fee
                        
                        # Record trade
                        trades.append({
                            "type": "stop_loss",
                            "side": "buy",
                            "price": current_price,
                            "quantity": -position,
                            "value": trade_value,
                            "fee": fee,
                            "profit_loss": (entry_price - current_price) * (-position) - fee,
                            "timestamp": current_row.name
                        })
                        
                        position = 0
                    
                    # Take profit check (for long positions)
                    elif take_profit is not None and position > 0 and price_change > take_profit:
                        # Execute take profit
                        trade_value = position * current_price
                        fee = trade_value * fee_rate
                        capital += trade_value - fee
                        
                        # Record trade
                        trades.append({
                            "type": "take_profit",
                            "side": "sell",
                            "price": current_price,
                            "quantity": position,
                            "value": trade_value,
                            "fee": fee,
                            "profit_loss": (current_price - entry_price) * position - fee,
                            "timestamp": current_row.name
                        })
                        
                        position = 0
                    
                    # Take profit check (for short positions)
                    elif take_profit is not None and position < 0 and price_change < -take_profit:
                        # Execute take profit
                        trade_value = -position * current_price
                        fee = trade_value * fee_rate
                        capital += trade_value - fee
                        
                        # Record trade
                        trades.append({
                            "type": "take_profit",
                            "side": "buy",
                            "price": current_price,
                            "quantity": -position,
                            "value": trade_value,
                            "fee": fee,
                            "profit_loss": (entry_price - current_price) * (-position) - fee,
                            "timestamp": current_row.name
                        })
                        
                        position = 0
                
                # Process trading signals
                if current_signal == 1 and position <= 0:  # Buy signal
                    # Close any existing short position
                    if position < 0:
                        trade_value = -position * current_price
                        fee = trade_value * fee_rate
                        capital += trade_value - fee
                        
                        # Record trade
                        trades.append({
                            "type": "signal",
                            "side": "buy",
                            "price": current_price,
                            "quantity": -position,
                            "value": trade_value,
                            "fee": fee,
                            "profit_loss": (entry_price - current_price) * (-position) - fee,
                            "timestamp": current_row.name
                        })
                    
                    # Open new long position
                    position_value = capital * position_size
                    quantity = position_value / current_price
                    fee = position_value * fee_rate
                    
                    if capital >= position_value + fee:
                        capital -= position_value + fee
                        position = quantity
                        entry_price = current_price
                        
                        # Record trade
                        trades.append({
                            "type": "signal",
                            "side": "buy",
                            "price": current_price,
                            "quantity": quantity,
                            "value": position_value,
                            "fee": fee,
                            "profit_loss": -fee,
                            "timestamp": current_row.name
                        })
                
                elif current_signal == -1 and position >= 0:  # Sell signal
                    # Close any existing long position
                    if position > 0:
                        trade_value = position * current_price
                        fee = trade_value * fee_rate
                        capital += trade_value - fee
                        
                        # Record trade
                        trades.append({
                            "type": "signal",
                            "side": "sell",
                            "price": current_price,
                            "quantity": position,
                            "value": trade_value,
                            "fee": fee,
                            "profit_loss": (current_price - entry_price) * position - fee,
                            "timestamp": current_row.name
                        })
                    
                    # Open new short position
                    position_value = capital * position_size
                    quantity = position_value / current_price
                    fee = position_value * fee_rate
                    
                    if capital >= position_value + fee:
                        capital -= position_value + fee
                        position = -quantity
                        entry_price = current_price
                        
                        # Record trade
                        trades.append({
                            "type": "signal",
                            "side": "sell",
                            "price": current_price,
                            "quantity": quantity,
                            "value": position_value,
                            "fee": fee,
                            "profit_loss": -fee,
                            "timestamp": current_row.name
                        })
            
            # Close final position
            if position != 0:
                final_price = result_df.iloc[-1]["close"]
                
                if position > 0:  # Long position
                    trade_value = position * final_price
                    fee = trade_value * fee_rate
                    capital += trade_value - fee
                    
                    # Record trade
                    trades.append({
                        "type": "final",
                        "side": "sell",
                        "price": final_price,
                        "quantity": position,
                        "value": trade_value,
                        "fee": fee,
                        "profit_loss": (final_price - entry_price) * position - fee,
                        "timestamp": result_df.index[-1]
                    })
                
                else:  # Short position
                    trade_value = -position * final_price
                    fee = trade_value * fee_rate
                    capital += trade_value - fee
                    
                    # Record trade
                    trades.append({
                        "type": "final",
                        "side": "buy",
                        "price": final_price,
                        "quantity": -position,
                        "value": trade_value,
                        "fee": fee,
                        "profit_loss": (entry_price - final_price) * (-position) - fee,
                        "timestamp": result_df.index[-1]
                    })
            
            # Calculate performance metrics
            total_profit_loss = sum(trade["profit_loss"] for trade in trades)
            roi = total_profit_loss / initial_capital
            
            # Calculate returns by day
            daily_returns = []
            current_capital = initial_capital
            
            for date, group in result_df.groupby(result_df.index.date):
                day_trades = [t for t in trades if t["timestamp"].date() == date]
                day_profit_loss = sum(t["profit_loss"] for t in day_trades)
                day_return = day_profit_loss / current_capital if current_capital > 0 else 0
                
                daily_returns.append({
                    "date": date,
                    "profit_loss": day_profit_loss,
                    "return": day_return
                })
                
                current_capital += day_profit_loss
            
            # Calculate metrics
            if len(daily_returns) > 0:
                returns = np.array([r["return"] for r in daily_returns])
                sharpe_ratio = np.mean(returns) / (np.std(returns) + 1e-10) * np.sqrt(252)
                max_drawdown = self._calculate_max_drawdown(returns)
                win_rate = sum(1 for t in trades if t["profit_loss"] > 0) / len(trades) if trades else 0
            else:
                sharpe_ratio = 0
                max_drawdown = 0
                win_rate = 0
            
            # Prepare results
            results = {
                "initial_capital": initial_capital,
                "final_capital": capital,
                "total_profit_loss": total_profit_loss,
                "roi": roi,
                "sharpe_ratio": sharpe_ratio,
                "max_drawdown": max_drawdown,
                "total_trades": len(trades),
                "win_rate": win_rate,
                "trades": trades,
                "daily_returns": daily_returns
            }
            
            logger.info(f"Simulation complete: ROI={roi:.2%}, Sharpe={sharpe_ratio:.2f}, Trades={len(trades)}")
            
            return results
            
        except Exception as e:
            logger.error(f"Error in trading simulation: {e}")
            raise
    
    def _calculate_max_drawdown(self, returns: np.ndarray) -> float:
        """
        Calculate maximum drawdown from returns.
        
        Args:
            returns: Array of period returns
            
        Returns:
            Maximum drawdown
        """
        # Calculate cumulative returns
        cum_returns = np.cumprod(1 + returns) - 1
        
        # Calculate running maximum
        running_max = np.maximum.accumulate(cum_returns)
        
        # Calculate drawdown
        drawdown = (running_max - cum_returns) / (1 + running_max)
        
        # Return maximum drawdown
        return float(np.max(drawdown))
    
    def __str__(self) -> str:
        """
        Return string representation of the inference engine.
        
        Returns:
            String description
        """
        status = []
        status.append(f"AriadneInference (Version: {self.INFERENCE_VERSION})")
        status.append(f"Model Loaded: {self.model_loaded}")
        status.append(f"Preprocessor Loaded: {self.preprocessor_loaded}")
        status.append(f"Prediction Mode: {self.prediction_mode}")
        status.append(f"Signal Type: {self.signal_type}")
        status.append(f"Deployment Mode: {self.deployment_mode}")
        
        if self.model_loaded:
            status.append(f"Model Type: {self.model.__class__.__name__}")
        
        if self.prediction_history:
            status.append(f"Prediction History: {len(self.prediction_history)} entries")
        
        return "\n".join(status)
    
    def __repr__(self) -> str:
        """
        Return string representation of the inference engine.
        
        Returns:
            String description
        """
        return self.__str__()


if __name__ == "__main__":
    # Example usage as script
    import argparse
    
    parser = argparse.ArgumentParser(description='Ariadne Inference CLI')
    parser.add_argument('--config', type=str, help='Path to configuration file')
    parser.add_argument('--model-dir', type=str, help='Path to model directory')
    parser.add_argument('--preprocessor-dir', type=str, help='Path to preprocessor directory')
    parser.add_argument('--mode', type=str, choices=['predict', 'server', 'docker', 'blockchain'], 
                        help='Operation mode')
    parser.add_argument('--data', type=str, help='Path to input data file')
    parser.add_argument('--output', type=str, help='Path to output file')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='API host')
    parser.add_argument('--port', type=int, default=5000, help='API port')
    
    args = parser.parse_args()
    
    if args.config:
        # Initialize inference engine
        inference = AriadneInference(
            config_path=args.config,
            model_dir=args.model_dir,
            preprocessor_dir=args.preprocessor_dir
        )
        
        if args.mode == 'predict' and args.data:
            # Load input data
            if args.data.endswith('.csv'):
                data = pd.read_csv(args.data)
            elif args.data.endswith('.parquet'):
                data = pd.read_parquet(args.data)
            else:
                print(f"Unsupported file format: {args.data}")
                exit(1)
            
            # Make predictions
            results = inference.predict(data, include_confidence=True, generate_signals=True)
            
            # Convert NumPy arrays to lists for JSON serialization
            for key, value in results.items():
                if isinstance(value, np.ndarray):
                    results[key] = value.tolist()
            
            # Save results
            if args.output:
                with open(args.output, 'w') as f:
                    json.dump(results, f, indent=2)
                print(f"Results saved to {args.output}")
            else:
                print(json.dumps(results, indent=2))
        
        elif args.mode == 'server':
            # Start REST API server
            inference.deploy_rest_api(
                host=args.host,
                port=args.port
            )
        
        elif args.mode == 'docker':
            # Deploy Docker container
            container_id = inference.deploy_docker_container(
                port=args.port
            )
            print(f"Docker container deployed: {container_id}")
        
        elif args.mode == 'blockchain':
            # Deploy to blockchain
            if not args.output:
                print("Error: --output required for blockchain deployment (wallet key file)")
                exit(1)
            
            tx_signature = inference.deploy_to_blockchain(
                wallet_path=args.output
            )
            print(f"Deployed to blockchain: {tx_signature}")
        
        else:
            print("Error: Invalid mode or missing required arguments")
            parser.print_help()
    else:
        print("Error: Configuration file required")
        parser.print_help()
            # Binary classification: 1 (buy), 0 (hold), -1 (sell)
            signals = np.zeros(len(predictions))
            
            # Apply buy threshold
            buy_mask = predictions > 0.5
            
            # Apply sell threshold
            sell_mask = predictions < 0.5
            
            # Set signals
            signals[buy_mask] = 1
            signals[sell_mask] = -1
            
            # Apply confidence filter if available
            if confidence is not None:
                # Only keep signals with sufficient confidence
                signals[confidence < threshold] = 0
        
        elif self.signal_type == "probability":
            # Probability output: map to buy/hold/sell based on probability ranges
            signals = np.zeros(len(predictions))
            
            # Apply buy threshold (high probability)
            buy_threshold = self.config.get("buy_threshold", 0.7)
            buy_mask = predictions > buy_threshold
            
            # Apply sell threshold (low probability)
            sell_threshold = self.config.get("sell_threshold", 0.3)
            sell_mask = predictions < sell_threshold
            
            # Set signals
            signals[buy_mask] = 1
            signals[sell_mask] = -1
            
            # Apply confidence filter if available
            if confidence is not None:
                signals[confidence < threshold] = 0
        
        elif self.signal_type == "multi_class":
            # Multi-class: map class predictions to signals
            # Assuming classes are ordered as [sell, hold, buy] or similar
            if len(predictions.shape) > 1 and predictions.shape[1] > 1:
                # Get class with highest probability
                signal_classes = np.argmax(predictions, axis=1)
                
                # Map classes to signals
                class_mapping = self.config.get("class_mapping", {})
                
                if class_mapping:
                    # Use provided mapping
                    signals = np.array([class_mapping.get(str(cls), 0) for cls in signal_classes])
                else:
                    # Default mapping: 0=sell, 1=hold, 2=buy
                    signals = np.where(signal_classes == 0, -1, 
                              np.where(signal_classes == 2, 1, 0))
                
                # Apply confidence filter if available
                if confidence is not None:
                    signals[confidence < threshold] = 0
            else:
                # Invalid shape for multi-class
                logger.error(f"Invalid shape for multi-class predictions: {predictions.shape}")
                signals = np.zeros(len(predictions))
        
        elif self.signal_type == "regression":
            # Regression: map continuous values to signals based on thresholds
            buy_threshold = self.config.get("buy_threshold", 0.02)  # e.g., 2% expected return
            sell_threshold = self.config.get("sell_threshold", -0.01)  # e.g., -1% expected return
            
            signals = np.zeros(len(predictions))
            signals[predictions > buy_threshold] = 1
            signals[predictions < sell_threshold] = -1
            
            # Apply confidence filter if available
            if confidence is not None:
                signals[confidence < threshold] = 0
        
        else:
            logger.warning(f"Unknown signal type: {self.signal_type}, using default signals")
            signals = np.zeros(len(predictions))
        
        return signals
    
    def _detect_data_drift(self, X: np.ndarray) -> float:
        """
        Detect data drift in input features.
        
        Args:
            X: Current batch of input data
            
        Returns:
            Drift score (higher means more drift)
        """
        # Check if drift detection is enabled
        if not self.config.get("enable_drift_detection", False):
            return 0.0
        
        try:
            # Get reference data
            if self.reference_data is None:
                # Try to load reference data from preprocessor
                if hasattr(self.preprocessor, 'X_train'):
                    # Use sample of training data as reference
                    max_samples = self.config.get("max_reference_samples", 1000)
                    indices = np.random.choice(
                        len(self.preprocessor.X_train),
                        size=min(max_samples, len(self.preprocessor.X_train)),
                        replace=False
                    )
                    self.reference_data = self.preprocessor.X_train[indices]
                    logger.info(f"Loaded reference data for drift detection, shape: {self.reference_data.shape}")
                else:
                    logger.warning("Reference data not available for drift detection")
                    return 0.0
            
            # Calculate drift score based on distribution difference
            # For simplicity, use mean absolute difference between feature distributions
            
            # Flatten feature dimensions (keeping batch dimension)
            if len(X.shape) > 2:
                X_flat = X.reshape(X.shape[0], -1)
                ref_flat = self.reference_data.reshape(self.reference_data.shape[0], -1)
            else:
                X_flat = X
                ref_flat = self.reference_data
            
            # Calculate means and standard deviations
            X_mean = np.mean(X_flat, axis=0)
            ref_mean = np.mean(ref_flat, axis=0)
            
            X_std = np.std(X_flat, axis=0)
            ref_std = np.std(ref_flat, axis=0)
            
            # Calculate normalized mean difference
            mean_diff = np.abs(X_mean - ref_mean) / (ref_std + 1e-6)
            
            # Calculate distribution difference (Kullback-Leibler divergence approximation)
            dist_diff = np.mean(mean_diff)
            
            # Normalize to [0, 1] scale
            drift_score = np.tanh(dist_diff)
            
            # Update monitoring metrics
            if 'drift_score' in self.monitoring_metrics:
                self.monitoring_metrics['drift_score'].set(drift_score)
            
            # Log warning if drift exceeds threshold
            drift_threshold = self.config.get("drift_detection_threshold", 0.5)
            if drift_score > drift_threshold:
                logger.warning(f"Data drift detected: score {drift_score:.3f} exceeds threshold {drift_threshold}")
                
                # Add to performance metrics
                self.performance_metrics["drift_detected"] = True
                self.performance_metrics["drift_score"] = drift_score
                self.performance_metrics["drift_timestamp"] = datetime.now().isoformat()
            
            return drift_score
            
        except Exception as e:
            logger.error(f"Error in drift detection: {e}")
            return 0.0
    
    def start_batch_processing(
        self,
        data_provider: Callable,
        result_handler: Callable,
        batch_size: Optional[int] = None,
        max_batches: Optional[int] = None
    ) -> None:
        """
        Start batch processing of data.
        
        Args:
            data_provider: Function that provides batches of data
            result_handler: Function that handles prediction results
            batch_size: Batch size for processing
            max_batches: Maximum number of batches to process
        """
        logger.info("Starting batch processing")
        
        # Use provided batch size or default
        batch_size = batch_size or self.batch_size
        
        # Track batches processed
        batches_processed = 0
        
        try:
            while True:
                # Check if max batches reached
                if max_batches is not None and batches_processed >= max_batches:
                    logger.info(f"Maximum batches ({max_batches}) reached, stopping batch processing")
                    break
                
                # Get next batch from provider
                batch = data_provider(batch_size)
                
                # Break if no more data
                if batch is None or (isinstance(batch, (list, np.ndarray)) and len(batch) == 0):
                    logger.info("No more data, stopping batch processing")
                    break
                
                # Process batch
                predictions = self.predict(batch)
                
                # Handle results
                result_handler(predictions)
                
                # Increment counter
                batches_processed += 1
                
                # Log progress
                if batches_processed % 10 == 0:
                    logger.info(f"Processed {batches_processed} batches")
            
            logger.info(f"Batch processing completed, {batches_processed} batches processed")
            
        except Exception as e:
            logger.error(f"Error in batch processing: {e}")
            raise
    
    def start_streaming_prediction(
        self,
        stream_provider: Callable,
        result_handler: Callable,
        prediction_interval: float = 1.0,
        max_runtime: Optional[float] = None
    ) -> None:
        """
        Start streaming prediction mode.
        
        Args:
            stream_provider: Function that provides streaming data
            result_handler: Function that handles prediction results
            prediction_interval: Interval between predictions (seconds)
            max_runtime: Maximum runtime (seconds)
        """
        logger.info("Starting streaming prediction")
        
        # Track runtime
        start_time = time.time()
        
        # Start worker thread if not already running
        if not self.is_running:
            self._start_worker_thread()
        
        try:
            while True:
                # Check if max runtime reached
                if max_runtime is not None and (time.time() - start_time) >= max_runtime:
                    logger.info(f"Maximum runtime ({max_runtime}s) reached, stopping streaming prediction")
                    break
                
                # Get data from stream
                data = stream_provider()
                
                # Skip if no data
                if data is None:
                    time.sleep(0.1)
                    continue
                
                # Add to prediction queue
                self.prediction_queue.put(data)
                
                # Check for results
                while not self.result_queue.empty():
                    result = self.result_queue.get()
                    result_handler(result)
                
                # Wait for next prediction interval
                time.sleep(prediction_interval)
            
            logger.info("Streaming prediction completed")
            
        except Exception as e:
            logger.error(f"Error in streaming prediction: {e}")
            raise
        finally:
            # Stop worker thread
            self.is_running = False
            if self.worker_thread and self.worker_thread.is_alive():
                self.worker_thread.join(timeout=5.0)
    
    def _worker_thread_func(self) -> None:
        """
        Worker thread function for processing prediction queue.
        """
        logger.info("Prediction worker thread started")
        
        while self.is_running:
            try:
                # Get data from queue with timeout
                try:
                    data = self.prediction_queue.get(timeout=0.5)
                except Queue.Empty:
                    continue
                
                # Lock during prediction
                with self.prediction_lock:
                    # Make prediction
                    result = self.predict(data)
                
                # Put result in result queue
                self.result_queue.put(result)
                
                # Update queue size metric
                if 'queue_size' in self.monitoring_metrics:
                    self.monitoring_metrics['queue_size'].set(self.prediction_queue.qsize())
                
            except Exception as e:
                logger.error(f"Error in prediction worker thread: {e}")
        
        logger.info("Prediction worker thread stopped")
    
    def _start_worker_thread(self) -> None:
        """
        Start the worker thread for processing predictions.
        """
        if self.is_running:
            logger.warning("Worker thread already running")
            return
        
        self.is_running = True
        self.worker_thread = Thread(target=self._worker_thread_func)
        self.worker_thread.daemon = True
        self.worker_thread.start()
        
        logger.info("Started prediction worker thread")
    
    def deploy_rest_api(
        self,
        host: Optional[str] = None,
        port: Optional[int] = None,
        api_prefix: str = "/api/v1",
        enable_auth: bool = True
    ) -> None:
        """
        Deploy the model as a REST API server.
        
        Args:
            host: Hostname to bind the server to
            port: Port to bind the server to
            api_prefix: URL prefix for API endpoints
            enable_auth: Enable authentication
        """
        # Check if flask is available
        if not optional_modules['flask']:
            logger.error("Flask not available, cannot deploy REST API")
            raise ImportError("Flask not available, cannot deploy REST API")
        
        # Use provided host/port or from config
        host = host or self.config.get("api_host", "0.0.0.0")
        port = port or self.config.get("api_port", 5000)
        
        # Create Flask app
        app = Flask("AriadneModelService")
        
        # API key for authentication
        api_key = self.config.get("api_key", None)
        if api_key is None and enable_auth:
            # Generate random API key
            api_key = uuid.uuid4().hex
            logger.info(f"Generated random API key: {api_key}")
        
        # Authentication decorator
        def require_auth(f):
            @wraps(f)
            def decorated(*args, **kwargs):
                if not enable_auth:
                    return f(*args, **kwargs)
                
                auth_header = request.headers.get('Authorization')
                if auth_header is None or not auth_header.startswith('Bearer '):
                    return jsonify({"error": "Missing or invalid Authorization header"}), 401
                
                token = auth_header.split(' ')[1]
                if token != api_key:
                    return jsonify({"error": "Invalid API key"}), 401
                
                return f(*args, **kwargs)
            return decorated
        
        # Define API routes
        @app.route(f"{api_prefix}/health", methods=['GET'])
        def health_check():
            return jsonify({
                "status": "ok",
                "model_loaded": self.model_loaded,
                "preprocessor_loaded": self.preprocessor_loaded,
                "version": self.INFERENCE_VERSION
            })
        
        @app.route(f"{api_prefix}/predict", methods=['POST'])
        @require_auth
        def predict_endpoint():
            # Get request data
            request_data = request.json
            
            if not request_data:
                return jsonify({"error": "No data provided"}), 400
            
            try:
                # Extract parameters
                data = request_data.get("data")
                include_confidence = request_data.get("include_confidence", True)
                generate_signals = request_data.get("generate_signals", False)
                
                if data is None:
                    return jsonify({"error": "No data field in request"}), 400
                
                # Convert data to NumPy array if needed
                if isinstance(data, list):
                    data = np.array(data)
                
                # Make prediction
                result = self.predict(data, include_confidence, generate_signals)
                
                # Convert NumPy arrays to lists for JSON serialization
                for key, value in result.items():
                    if isinstance(value, np.ndarray):
                        result[key] = value.tolist()
                
                return jsonify(result)
                
            except Exception as e:
                logger.error(f"Error in prediction endpoint: {e}")
                return jsonify({"error": str(e)}), 500
        
        @app.route(f"{api_prefix}/model_info", methods=['GET'])
        @require_auth
        def model_info_endpoint():
            if not self.model_loaded:
                return jsonify({"error": "Model not loaded"}), 500
            
            try:
                # Get model information
                info = {
                    "version": self.INFERENCE_VERSION,
                    "model_type": self.model.__class__.__name__,
                    "signal_type": self.signal_type,
                    "performance_metrics": self.performance_metrics,
                    "prediction_history_length": len(self.prediction_history)
                }
                
                return jsonify(info)
                
            except Exception as e:
                logger.error(f"Error in model info endpoint: {e}")
                return jsonify({"error": str(e)}), 500
        
        @app.route(f"{api_prefix}/metrics", methods=['GET'])
        @require_auth
        def metrics_endpoint():
            if optional_modules['prometheus_client']:
                # Generate Prometheus metrics
                return prom.generate_latest().decode('utf-8'), 200, {'Content-Type': 'text/plain'}
            else:
                return jsonify({"error": "Prometheus client not available"}), 500
        
        # Start server
        logger.info(f"Starting REST API server on {host}:{port}")
        
        # Store server instance
        self.server = app
        
        # Run in separate thread if in non-blocking mode
        if self.config.get("non_blocking_api", False):
            from threading import Thread
            api_thread = Thread(target=app.run, kwargs={
                'host': host,
                'port': port,
                'debug': False,
                'threaded': True
            })
            api_thread.daemon = True
            api_thread.start()
            logger.info("REST API server started in non-blocking mode")
        else:
            # Blocking mode
            app.run(host=host, port=port, debug=False, threaded=True)
    
    def deploy_docker_container(
        self,
        image_name: str = "ariadne-inference",
        port: int = 5000,
        gpu: bool = False
    ) -> str:
        """
        Deploy the model as a Docker container.
        
        Args:
            image_name: Name for the Docker image
            port: Port to expose
            gpu: Whether to use GPU
            
        Returns:
            Container ID
        """
        # Check if docker module is available
        if not optional_modules['docker']:
            logger.error("Docker module not available, cannot deploy container")
            raise ImportError("Docker module not available, cannot deploy container")
        
        try:
            # Initialize Docker client
            client = docker.from_env()
            
            # Export model and preprocessor
            model_archive_path = os.path.join(self.model_dir, "model_archive.zip")
            self._export_model_archive(model_archive_path)
            
            # Create Dockerfile
            dockerfile_content = self._generate_dockerfile(gpu)
            dockerfile_path = os.path.join(self.model_dir, "Dockerfile")
            
            with open(dockerfile_path, "w") as f:
                f.write(dockerfile_content)
            
            # Build Docker image
            logger.info(f"Building Docker image: {image_name}")
            
            image, logs = client.images.build(
                path=str(self.model_dir),
                tag=image_name,
                rm=True
            )
            
            # Log build output
            for log in logs:
                if 'stream' in log:
                    logger.info(log['stream'].strip())
            
            # Run container
            logger.info(f"Starting Docker container from image: {image_name}")
            
            container = client.containers.run(
                image_name,
                detach=True,
                ports={f"5000/tcp": port},
                environment={
                    "MODEL_DIR": "/app/models",
                    "API_PORT": "5000",
                    "API_HOST": "0.0.0.0"
                },
                restart_policy={"Name": "unless-stopped"}
            )
            
            logger.info(f"Docker container started: {container.id}")
            
            return container.id
            
        except Exception as e:
            logger.error(f"Failed to deploy Docker container: {e}")
            raise
    
    def _generate_dockerfile(self, gpu: bool = False) -> str:
        """
        Generate Dockerfile for containerization.
        
        Args:
            gpu: Whether to use GPU base image
            
        Returns:
            Dockerfile content
        """
        # Determine base image
        base_image = "tensorflow/tensorflow:2.12.0-gpu" if gpu else "tensorflow/tensorflow:2.12.0"
        
        # Create Dockerfile content
        dockerfile = f"""FROM {base_image}

# Install dependencies
RUN pip install numpy pandas scikit-learn flask gunicorn scipy joblib pyyaml

# Copy model files
WORKDIR /app
COPY model_archive.zip /app/model_archive.zip
RUN mkdir -p /app/models && unzip model_archive.zip -d /app/models

# Copy source code
COPY inference.py /app/
COPY model.py /app/
COPY preprocessor.py /app/

# Set environment variables
ENV MODEL_DIR=/app/models
ENV PYTHONUNBUFFERED=1

# Expose port
EXPOSE 5000

# Start API server
CMD ["python", "-c", "from inference import AriadneInference; model = AriadneInference(model_dir=\\"/app/models\\"); model.deploy_rest_api(host=\\"0.0.0.0\\", port=5000)"]
"""
        
        return dockerfile
    
    def _export_model_archive(self, archive_path: str) -> None:
        """
        Export model and preprocessor to an archive for containerization.
        
        Args:
            archive_path: Path to save the archive
        """
        import zipfile
        import tempfile
        
        logger.info(f"Exporting model archive to {archive_path}")
        
        # Create a temporary directory
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create directory structure
            os.makedirs(os.path.join(temp_dir, "model"), exist_ok=True)
            os.makedirs(os.path.join(temp_dir, "preprocessor"), exist_ok=True)
            
            # Save model files
            if self.model_loaded:
                model_path = os.path.join(temp_dir, "model", "ariadne_model")
                self.model.save(model_path)
                
                # Save config
                with open(os.path.join(temp_dir, "model", "config.json"), "w") as f:
                    json.dump(self.model.config, f, indent=2)
            
            # Save preprocessor files
            if self.preprocessor_loaded:
                # Save config
                with open(os.path.join(temp_dir, "preprocessor", "config.json"), "w") as f:
                    json.dump(self.preprocessor.config, f, indent=2)
                
                # Save scalers
                for scaler_name, scaler in self.preprocessor.feature_scalers.items():
                    joblib.dump(scaler, os.path.join(temp_dir, "preprocessor", f"{scaler_name}_scaler.joblib"))
                
                for scaler_name, scaler in self.preprocessor.label_scalers.items():
                    joblib.dump(scaler, os.path.join(temp_dir, "preprocessor", f"{scaler_name}_label_scaler.joblib"))
            
            # Create zip archive
            with zipfile.ZipFile(archive_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                for root, _, files in os.walk(temp_dir):
                    for file in files:
                        file_path = os.path.join(root, file)
                        arcname = os.path.relpath(file_path, temp_dir)
                        zipf.write(file_path, arcname)
        
        logger.info(f"Model archive exported to {archive_path}")
    
    def deploy_to_blockchain(
        self,
        wallet_path: str,
        model_hash: Optional[str] = None,
        blockchain_endpoint: Optional[str] = None
    ) -> str:
        """
        Deploy model metadata and verification to blockchain.
        
        Args:
            wallet_path: Path to wallet key file
            model_hash: Hash of the model (if None, computed automatically)
            blockchain_endpoint: Blockchain RPC endpoint
            
        Returns:
            Transaction signature
        """
        # Check if blockchain modules are available
        if not optional_modules['base58']:
            logger.error("base58 module not available, cannot deploy to blockchain")
            raise ImportError("base58 module not available, cannot deploy to blockchain")
        
        # Use provided endpoint or from config
        blockchain_endpoint = blockchain_endpoint or self.config.get("blockchain_endpoint")
        
        if not blockchain_endpoint:
            logger.error("No blockchain endpoint provided")
            raise ValueError("No blockchain endpoint provided")
        
        try:
            # Calculate model hash if not provided
            if model_hash is None:
                model_hash = self._calculate_model_hash()
            
            # Current timestamp
            timestamp = int(time.time())
            
            # Prepare model metadata
            metadata = {
                "model_name": "Ariadne",
                "version": self.INFERENCE_VERSION,
                "model_hash": model_hash,
                "timestamp": timestamp,
                "signal_type": self.signal_type,
                "deployment_id": str(uuid.uuid4())
            }
            
            # Hash metadata
            metadata_str = json.dumps(metadata, sort_keys=True)
            metadata_hash = hashlib.sha256(metadata_str.encode()).hexdigest()
            
            logger.info(f"Deploying model metadata to blockchain with hash: {metadata_hash}")
            
            # Simplified blockchain interaction for pseudocode
            # In a real implementation, this would use solana-web3.js or another library
            
            # Load wallet
            with open(wallet_path, 'r') as f:
                wallet_key = f.read().strip()
            
            # Create transaction
            # This is pseudocode; actual implementation would use Solana SDK
            transaction_data = {
                "metadata": metadata,
                "metadata_hash": metadata_hash,
                "wallet": wallet_key,
                "timestamp": timestamp
            }
            
            # Submit transaction to blockchain
            transaction_signature = "simulated_blockchain_tx_" + metadata_hash[:16]
            
            logger.info(f"Model metadata deployed to blockchain: {transaction_signature}")
            
            # Store deployment info
            self.performance_metrics["blockchain_deployment"] = {
                "timestamp": datetime.now().isoformat(),
                "model_hash": model_hash,
                "metadata_hash": metadata_hash,
                "transaction": transaction_signature
            }
            
            return transaction_signature
            
        except Exception as e:
            logger.error(f"Failed to deploy to blockchain: {e}")
            raise
    
    def _calculate_model_hash(self) -> str:
        """
        Calculate a unique hash of the model.
        
        Returns:
            Hash string
        """
        try:
            # Get model weights
            if not self.model_loaded:
                logger.error("Model not loaded, cannot calculate hash")
                return "model_not_loaded"
            
            # Get model weights as NumPy arrays
            weights = self.model.model.get_weights()
            
            # Concatenate weight arrays
            all_weights = np.concatenate([arr.flatten() for arr in weights if arr.size > 0])
            
            # Calculate hash of weights
            weight_hash = hashlib.sha256(all_weights.tobytes()).hexdigest()
            
            # Include configuration in hash
            config_str = json.dumps(self.model.config, sort_keys=True)
            config_hash = hashlib.sha256(config_str.encode()).hexdigest()
            
            # Combine hashes
            combined_hash = hashlib.sha256((weight_hash + config_hash).encode()).hexdigest()
            
            return combined_hash
            
        except Exception as e:
            logger.error(f"Failed to calculate model hash: {e}")
            return f"error_{uuid.uuid4().hex[:8]}"
    
    def publish_signals_to_exchange(
        self,
        signals: np.ndarray,
        exchange_api_endpoint: str,
        api_key: str,
        api_secret: str,
        symbol: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Publish trading signals to an exchange API.
        
        Args:
            signals: Array of trading signals
            exchange_api_endpoint: Exchange API endpoint URL
            api_key: Exchange API key
            api_secret: Exchange API secret
            symbol: Trading symbol
            metadata: Additional metadata to include
            
        Returns:
            Response from exchange API
        """
        if not optional_modules['requests']:
            logger.error("Requests module not available, cannot publish signals")
            raise ImportError("Requests module not available, cannot publish signals")
        
        logger.info(f"Publishing signals to exchange: {exchange_api_endpoint}")
        
        try:
            # Check for valid signals
            if signals is None or len(signals) == 0:
                logger.warning("No signals to publish")
                return {"status": "warning", "message": "No signals to publish"}
            
            # Latest signal (most recent)
            latest_signal = signals[-1]
            
            # Convert signal to action
            signal_map = {1: "buy", 0: "hold", -1: "sell"}
            action = signal_map.get(int(latest_signal), "hold")
            
            # Current time
            timestamp = int(time.time() * 1000)  # milliseconds
            
            # Prepare request payload
            payload = {
                "action": action,
                "symbol": symbol,
                "timestamp": timestamp,
                "model": "Ariadne",
                "version": self.INFERENCE_VERSION,
                "signal_type": self.signal_type
            }
            
            # Add metadata if provided
            if metadata:
                payload["metadata"] = metadata
            
            # Create signature
            signature_base = f"{api_key}{timestamp}{symbol}{action}"
            signature = hmac.new(
                api_secret.encode(),
                signature_base.encode(),
                hashlib.sha256
            ).hexdigest()
            
            # Prepare headers
            headers = {
                "X-API-Key": api_key,
                "X-API-Signature": signature,
                "Content-Type": "application/json"
            }
            
            # Send request to exchange API
            response = requests.post(
                exchange_api_endpoint,
                headers=headers,
                json=payload
            )
            
            # Check response
            response.raise_for_status()
            
            # Parse response
            response_data = response.json()
            
            logger.info(f"Signals published to exchange: {response_data.get('status', 'unknown')}")
            
            return response_data
            
        except Exception as e:
            logger.error(f"Failed to publish signals to exchange: {e}")
            raise
    
    def start_metrics_server(self, port: int = 8000) -> None:
        """
        Start a Prometheus metrics server.
        
        Args:
            port: Port to bind the server to
        """
        if not optional_modules['prometheus_client']:
            logger.error("Prometheus client not available, cannot start metrics server")
            raise ImportError("Prometheus client not available, cannot start metrics server")
        
        logger.info(f"Starting Prometheus metrics server on port {port}")
        
        try:
            # Start metrics server
            prom.start_http_server(port)
            
            logger.info(f"Prometheus metrics server started on port {port}")
            
        except Exception as e:
            logger.error(f"Failed to start metrics server: {e}")
            raise
    
    def evaluate_performance(
        self,
        actual_values: np.ndarray,
        predictions: np.ndarray,
        signals: Optional[np.ndarray] = None
    ) -> Dict"""
Ariadne Inference - Inference and deployment module for the Ariadne trading model

This module handles model inference, prediction, and deployment of the Ariadne
trading model. It provides functionality for:

- Loading trained models
- Real-time and batch inference
- Trading signal generation
- Performance monitoring and drift detection
- Model serving for production environments
- Integration with the blockchain and trading infrastructure
- Confidence scoring and risk management
- Multi-model ensemble inference

Author: Minos-AI Team
Date: December 15, 2024
License: Proprietary
"""

import os
import json
import time
import logging
import warnings
import importlib
import hmac
import hashlib
from typing import Dict, List, Tuple, Union, Optional, Any, Callable
from pathlib import Path
from datetime import datetime, timedelta
from threading import Thread, Lock
from queue import Queue
import socket
import uuid
from functools import wraps

import numpy as np
import pandas as pd
import tensorflow as tf

# Import local modules with error handling
try:
    from .model import AriadneModel
    from .preprocessor import AriadnePreprocessor
except ImportError:
    # Relative imports may fail when running as script
    from model import AriadneModel
    from preprocessor import AriadnePreprocessor

# Optional imports for additional functionality
optional_modules = {
    'requests': False,
    'flask': False,
    'fastapi': False,
    'uvicorn': False,
    'websockets': False,
    'joblib': False,
    'docker': False,
    'prometheus_client': False,
    'redis': False,
    'kafka': False,
    's3fs': False,
    'base58': False,  # For Solana blockchain integration
    'psutil': False
}

# Try importing optional modules
for module_name in optional_modules:
    try:
        importlib.import_module(module_name)
        optional_modules[module_name] = True
    except ImportError:
        optional_modules[module_name] = False

# Import optional modules only if available
if optional_modules['requests']:
    import requests

if optional_modules['flask']:
    from flask import Flask, request, jsonify

if optional_modules['fastapi']:
    import fastapi
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel

if optional_modules['docker']:
    import docker

if optional_modules['prometheus_client']:
    import prometheus_client as prom

if optional_modules['joblib']:
    import joblib

if optional_modules['redis']:
    import redis

if optional_modules['kafka']:
    from kafka import KafkaProducer, KafkaConsumer

if optional_modules['s3fs']:
    import s3fs

if optional_modules['base58']:
    import base58

if optional_modules['psutil']:
    import psutil

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("ariadne_inference.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Suppress specific warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, module='tensorflow')
tf.get_logger().setLevel('ERROR')


class AriadneInference:
    """
    Inference engine for the Ariadne trading model.
    
    Handles model loading, prediction, signal generation, and deployment.
    
    Features:
    - Real-time and batch prediction
    - Trading signal generation with confidence scores
    - Model performance monitoring
    - Drift detection
    - High-availability deployment options
    - Blockchain integration for signal verification
    - Multi-model ensemble for improved robustness
    """
    
    # Class constants
    INFERENCE_VERSION = "1.0.0"
    SUPPORTED_PREDICTION_MODES = ["real_time", "batch", "streaming"]
    SUPPORTED_SIGNAL_TYPES = ["binary", "probability", "multi_class", "regression"]
    SUPPORTED_DEPLOYMENT_MODES = ["local", "rest", "docker", "kubernetes", "serverless", "blockchain"]
    
    def __init__(
        self,
        config_path: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        model_dir: Optional[str] = None,
        preprocessor_dir: Optional[str] = None,
        verbose: bool = True
    ):
        """
        Initialize the Ariadne inference engine.
        
        Args:
            config_path: Path to JSON configuration file
            config: Configuration dictionary (overrides config_path if provided)
            model_dir: Directory containing the trained model
            preprocessor_dir: Directory containing the preprocessor
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
        
        # Set model directory
        if model_dir is None:
            self.model_dir = Path(self.config.get("model_dir", "./models"))
        else:
            self.model_dir = Path(model_dir)
        
        # Set preprocessor directory
        if preprocessor_dir is None:
            self.preprocessor_dir = Path(self.config.get("preprocessor_dir", "./data"))
        else:
            self.preprocessor_dir = Path(preprocessor_dir)
        
        # Validate configuration
        self._validate_config()
        
        # Initialize instance attributes
        self.model = None
        self.preprocessor = None
        self.ensemble_models = []
        self.model_loaded = False
        self.preprocessor_loaded = False
        self.prediction_mode = self.config.get("prediction_mode", "real_time")
        self.signal_type = self.config.get("signal_type", "binary")
        self.deployment_mode = self.config.get("deployment_mode", "local")
        self.confidence_threshold = self.config.get("confidence_threshold", 0.6)
        self.batch_size = self.config.get("batch_size", 32)
        self.prediction_history = []
        self.performance_metrics = {}
        self.prediction_queue = Queue()
        self.result_queue = Queue()
        self.prediction_lock = Lock()
        self.is_running = False
        self.worker_thread = None
        self.server = None
        self.container = None
        self.monitoring_metrics = self._initialize_metrics() if optional_modules['prometheus_client'] else {}
        self.reference_data = None
        
        # Set random seeds for reproducibility
        np.random.seed(self.config.get("random_seed", 42))
        tf.random.set_seed(self.config.get("random_seed", 42))
        
        if self.verbose:
            logger.info(f"Ariadne inference engine initialized (Version: {self.INFERENCE_VERSION})")
            logger.info(f"Model directory: {self.model_dir}")
            logger.info(f"Preprocessor directory: {self.preprocessor_dir}")
        
        # Load model and preprocessor if auto_load is enabled
        if self.config.get("auto_load", True):
            self.load_model_and_preprocessor()
    
    def _validate_config(self) -> None:
        """
        Validate the inference configuration.
        
        Checks for required parameters, parameter types, and valid values.
        
        Raises:
            ValueError: If required configuration parameters are missing or invalid
        """
        # Validate prediction mode
        prediction_mode = self.config.get("prediction_mode", "real_time")
        if prediction_mode not in self.SUPPORTED_PREDICTION_MODES:
            logger.error(f"Unsupported prediction mode: {prediction_mode}. "
                        f"Supported modes are: {self.SUPPORTED_PREDICTION_MODES}")
            raise ValueError(f"Unsupported prediction mode: {prediction_mode}. "
                           f"Supported modes are: {self.SUPPORTED_PREDICTION_MODES}")
        
        # Validate signal type
        signal_type = self.config.get("signal_type", "binary")
        if signal_type not in self.SUPPORTED_SIGNAL_TYPES:
            logger.error(f"Unsupported signal type: {signal_type}. "
                        f"Supported types are: {self.SUPPORTED_SIGNAL_TYPES}")
            raise ValueError(f"Unsupported signal type: {signal_type}. "
                           f"Supported types are: {self.SUPPORTED_SIGNAL_TYPES}")
        
        # Validate deployment mode
        deployment_mode = self.config.get("deployment_mode", "local")
        if deployment_mode not in self.SUPPORTED_DEPLOYMENT_MODES:
            logger.error(f"Unsupported deployment mode: {deployment_mode}. "
                        f"Supported modes are: {self.SUPPORTED_DEPLOYMENT_MODES}")
            raise ValueError(f"Unsupported deployment mode: {deployment_mode}. "
                           f"Supported modes are: {self.SUPPORTED_DEPLOYMENT_MODES}")
        
        # Validate numeric parameters
        numeric_params = {
            "confidence_threshold": (float, (0, 1)),
            "batch_size": (int, (1, None)),
            "max_request_size": (int, (1, None)),
            "prediction_timeout": (float, (0, None)),
            "num_parallel_threads": (int, (1, None)),
            "model_refresh_interval": (int, (0, None)),
            "drift_detection_threshold": (float, (0, None)),
            "max_prediction_history": (int, (0, None))
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
        
        # Validate REST API parameters
        if deployment_mode == "rest":
            if not all(k in self.config for k in ["api_host", "api_port"]):
                logger.error("Missing required API parameters for REST deployment")
                raise ValueError("Missing required API parameters for REST deployment")
        
        # Validate blockchain parameters
        if deployment_mode == "blockchain":
            if not all(k in self.config for k in ["blockchain_endpoint", "wallet_path"]):
                logger.error("Missing required blockchain parameters for blockchain deployment")
                raise ValueError("Missing required blockchain parameters for blockchain deployment")
        
        logger.info("Configuration validation completed successfully")
    
    def _initialize_metrics(self) -> Dict[str, Any]:
        """
        Initialize monitoring metrics for Prometheus.
        
        Returns:
            Dictionary of Prometheus metrics
        """
        metrics = {}
        
        # Only create metrics if prometheus_client is available
        if optional_modules['prometheus_client']:
            # Prediction latency histogram
            metrics['prediction_latency'] = prom.Histogram(
                'ariadne_prediction_latency_seconds',
                'Prediction latency in seconds',
                buckets=prom.exponential_buckets(0.001, 2, 10)
            )
            
            # Prediction count counter
            metrics['prediction_count'] = prom.Counter(
                'ariadne_prediction_count_total',
                'Total number of predictions made',
                ['signal_type', 'outcome']
            )
            
            # Model loading time gauge
            metrics['model_load_time'] = prom.Gauge(
                'ariadne_model_load_time_seconds',
                'Time taken to load the model'
            )
            
            # Prediction batch size histogram
            metrics['batch_size'] = prom.Histogram(
                'ariadne_batch_size',
                'Size of prediction batches',
                buckets=[1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
            )
            
            # Memory usage gauge
            metrics['memory_usage'] = prom.Gauge(
                'ariadne_memory_usage_bytes',
                'Memory usage in bytes'
            )
            
            # Prediction confidence histogram
            metrics['prediction_confidence'] = prom.Histogram(
                'ariadne_prediction_confidence',
                'Confidence of predictions',
                buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]
            )
            
            # Request queue size gauge
            metrics['queue_size'] = prom.Gauge(
                'ariadne_queue_size',
                'Number of requests in the queue'
            )
            
            # Drift detection gauge
            metrics['drift_score'] = prom.Gauge(
                'ariadne_drift_score',
                'Data drift score'
            )
            
            logger.info("Prometheus metrics initialized")
        
        return metrics
    
    def load_model_and_preprocessor(
        self,
        model_path: Optional[str] = None,
        preprocessor_path: Optional[str] = None
    ) -> bool:
        """
        Load the Ariadne model and preprocessor.
        
        Args:
            model_path: Optional specific path to the model directory
            preprocessor_path: Optional specific path to the preprocessor directory
            
        Returns:
            True if loading was successful, False otherwise
        """
        logger.info("Loading model and preprocessor")
        
        start_time = time.time()
        success = True
        
        try:
            # Load preprocessor first
            if not self.load_preprocessor(preprocessor_path):
                logger.error("Failed to load preprocessor")
                return False
            
            # Then load model
            if not self.load_model(model_path):
                logger.error("Failed to load model")
                return False
            
            # Load ensemble models if enabled
            if self.config.get("use_ensemble", False):
                ensemble_paths = self.config.get("ensemble_model_paths", [])
                
                if not ensemble_paths and model_path:
                    # Try to find ensemble models in the directory
                    base_path = Path(model_path)
                    ensemble_dir = base_path.parent / f"{base_path.name}_ensemble"
                    
                    if ensemble_dir.exists():
                        ensemble_paths = [str(p) for p in ensemble_dir.glob("ensemble_model_*.h5")]
                
                # Load each ensemble model
                for path in ensemble_paths:
                    try:
                        model_config = self.config.copy()
                        model = AriadneModel(config=model_config)
                        model.load(path, custom_objects=None)
                        self.ensemble_models.append(model)
                        logger.info(f"Loaded ensemble model from {path}")
                    except Exception as e:
                        logger.error(f"Failed to load ensemble model from {path}: {e}")
                        success = False
                
                logger.info(f"Loaded {len(self.ensemble_models)} ensemble models")
            
            # Record model loading time for monitoring
            load_time = time.time() - start_time
            
            if 'model_load_time' in self.monitoring_metrics:
                self.monitoring_metrics['model_load_time'].set(load_time)
            
            logger.info(f"Model and preprocessor loaded successfully in {load_time:.2f} seconds")
            
            return success
            
        except Exception as e:
            logger.error(f"Failed to load model and preprocessor: {e}")
            return False
    
    def load_model(self, model_path: Optional[str] = None) -> bool:
        """
        Load the Ariadne model.
        
        Args:
            model_path: Optional specific path to the model
            
        Returns:
            True if loading was successful, False otherwise
        """
        try:
            # Use provided path or find latest model
            if model_path is None:
                model_files = list(self.model_dir.glob("ariadne_model_*.h5"))
                
                if not model_files:
                    logger.error(f"No model files found in {self.model_dir}")
                    return False
                
                # Sort by creation time (latest first)
                model_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
                model_path = str(model_files[0])
            
            # Remove file extension for AriadneModel.load()
            if model_path.endswith('.h5'):
                model_path = model_path[:-3]
            
            # Initialize model
            model_config = self.config.copy()
            self.model = AriadneModel(config=model_config)
            
            # Load the model
            self.model.load(model_path, custom_objects=None)
            self.model_loaded = True
            
            logger.info(f"Model loaded from {model_path}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False
    
    def load_preprocessor(self, preprocessor_path: Optional[str] = None) -> bool:
        """
        Load the Ariadne preprocessor.
        
        Args:
            preprocessor_path: Optional specific path to the preprocessor directory
            
        Returns:
            True if loading was successful, False otherwise
        """
        try:
            # Use provided path or default path
            preprocessor_path = preprocessor_path or self.preprocessor_dir
            
            # Initialize preprocessor
            preprocessor_config = self.config.copy()
            self.preprocessor = AriadnePreprocessor(config=preprocessor_config, output_dir=preprocessor_path)
            
            # Load the preprocessor data
            preprocessor_data = self.preprocessor.load_processed_data()
            self.preprocessor_loaded = True
            
            logger.info(f"Preprocessor loaded from {preprocessor_path}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to load preprocessor: {e}")
            return False
    
    def predict(
        self,
        data: Union[pd.DataFrame, np.ndarray, List[float]],
        include_confidence: bool = True,
        generate_signals: bool = False
    ) -> Dict[str, Any]:
        """
        Make predictions using the Ariadne model.
        
        Args:
            data: Input data for prediction (DataFrame, NumPy array, or list)
            include_confidence: Whether to include confidence scores
            generate_signals: Whether to generate trading signals
            
        Returns:
            Dictionary containing predictions and metadata
            
        Raises:
            ValueError: If model or preprocessor is not loaded
            RuntimeError: If prediction fails
        """
        if not self.model_loaded:
            logger.error("Model not loaded. Call load_model_and_preprocessor() first")
            raise ValueError("Model not loaded. Call load_model_and_preprocessor() first")
        
        if not self.preprocessor_loaded:
            logger.error("Preprocessor not loaded. Call load_model_and_preprocessor() first")
            raise ValueError("Preprocessor not loaded. Call load_model_and_preprocessor() first")
        
        # Record start time for latency measurement
        start_time = time.time()
        
        try:
            # Process input data
            X = self._prepare_input_data(data)
            
            # Make prediction
            if self.config.get("use_ensemble", False) and self.ensemble_models:
                # Use ensemble prediction
                predictions, uncertainty = self._ensemble_predict(X)
            else:
                # Use single model prediction
                if include_confidence:
                    predictions, uncertainty = self.model.predict(X, batch_size=self.batch_size, return_uncertainty=True)
                else:
                    predictions = self.model.predict(X, batch_size=self.batch_size, return_uncertainty=False)
                    uncertainty = None
            
            # Calculate prediction latency
            latency = time.time() - start_time
            
            # Update monitoring metrics
            if 'prediction_latency' in self.monitoring_metrics:
                self.monitoring_metrics['prediction_latency'].observe(latency)
            
            if 'batch_size' in self.monitoring_metrics:
                self.monitoring_metrics['batch_size'].observe(len(X) if hasattr(X, "__len__") else 1)
            
            # Record memory usage
            if 'memory_usage' in self.monitoring_metrics and optional_modules['psutil']:
                # Get memory usage in bytes
                process = psutil.Process(os.getpid())
                memory_info = process.memory_info()
                self.monitoring_metrics['memory_usage'].set(memory_info.rss)
            
            # Build result dictionary
            result = {
                "predictions": predictions,
                "timestamp": datetime.now().isoformat(),
                "latency": latency
            }
            
            # Add confidence scores if requested
            if include_confidence:
                confidence_scores = self._calculate_confidence_scores(predictions, uncertainty)
                result["confidence"] = confidence_scores
                
                # Update monitoring metrics
                if 'prediction_confidence' in self.monitoring_metrics:
                    for score in confidence_scores:
                        self.monitoring_metrics['prediction_confidence'].observe(score)
            
            # Generate trading signals if requested
            if generate_signals:
                signals = self._generate_trading_signals(predictions, confidence_scores if include_confidence else None)
                result["signals"] = signals
                
                # Update monitoring metrics
                if 'prediction_count' in self.monitoring_metrics:
                    for signal in signals:
                        self.monitoring_metrics['prediction_count'].labels(
                            signal_type=self.signal_type,
                            outcome=signal
                        ).inc()
            
            # Store prediction in history
            max_history = self.config.get("max_prediction_history", 1000)
            if max_history > 0:
                self.prediction_history.append(result)
                
                # Trim history if needed
                if len(self.prediction_history) > max_history:
                    self.prediction_history = self.prediction_history[-max_history:]
            
            # Perform drift detection
            self._detect_data_drift(X)
            
            return result
            
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            raise RuntimeError(f"Prediction failed: {e}")
    
    def _prepare_input_data(self, data: Union[pd.DataFrame, np.ndarray, List[float]]) -> np.ndarray:
        """
        Prepare input data for prediction.
        
        Args:
            data: Input data in various formats
            
        Returns:
            NumPy array ready for model input
            
        Raises:
            ValueError: If data format is invalid
        """
        try:
            # Handle different input types
            if isinstance(data, pd.DataFrame):
                # Process DataFrame using preprocessor
                df_norm = self.preprocessor.normalize_features(data, is_train=False)
                
                # Create sequences
                X, _ = self.preprocessor.create_sequences(
                    df_norm,
                    sequence_length=self.preprocessor.sequence_length,
                    prediction_horizon=1,  # For inference, we only need one step ahead
                    step_size=1,
                    target_column=self.preprocessor.target_column
                )
                
                return X
                
            elif isinstance(data, np.ndarray):
                # Handle NumPy array based on shape
                if len(data.shape) == 1:
                    # Single feature vector, reshape for model
                    return data.reshape(1, -1, 1)
                    
                elif len(data.shape) == 2:
                    # Batch of feature vectors or sequence, check dimensions
                    if data.shape[0] == 1 or data.shape[1] == self.preprocessor.sequence_length:
                        # Already in correct format or single sequence
                        return data.reshape(1, self.preprocessor.sequence_length, -1)
                    else:
                        # Batch of feature vectors, reshape for model
                        return data.reshape(-1, 1, data.shape[1])
                
                elif len(data.shape) == 3:
                    # Already in (batch_size, sequence_length, features) format
                    return data
                
                else:
                    logger.error(f"Invalid input shape: {data.shape}")
                    raise ValueError(f"Invalid input shape: {data.shape}")
            
            elif isinstance(data, list):
                # Convert list to NumPy array
                arr = np.array(data)
                
                # Reshape based on model requirements
                if len(arr.shape) == 1:
                    # Single feature vector
                    return arr.reshape(1, -1, 1)
                else:
                    # Already multidimensional
                    return self._prepare_input_data(arr)
            
            else:
                logger.error(f"Unsupported input type: {type(data)}")
                raise ValueError(f"Unsupported input type: {type(data)}")
                
        except Exception as e:
            logger.error(f"Failed to prepare input data: {e}")
            raise
    
    def _ensemble_predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate predictions using ensemble of models.
        
        Args:
            X: Input data
            
        Returns:
            Tuple of (predictions, uncertainty)
        """
        # Collect predictions from all ensemble models
        ensemble_preds = []
        
        for model in self.ensemble_models:
            preds = model.predict(X, batch_size=self.batch_size)
            ensemble_preds.append(preds)
        
        # If no ensemble models, use main model
        if not ensemble_preds:
            return self.model.predict(X, batch_size=self.batch_size, return_uncertainty=True)
        
        # Calculate ensemble mean and standard deviation
        ensemble_preds = np.array(ensemble_preds)
        mean_preds = np.mean(ensemble_preds, axis=0)
        std_preds = np.std(ensemble_preds, axis=0)
        
        return mean_preds, std_preds
    
    def _calculate_confidence_scores(
        self,
        predictions: np.ndarray,
        uncertainty: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Calculate confidence scores for predictions.
        
        Args:
            predictions: Model predictions
            uncertainty: Uncertainty estimates (if available)
            
        Returns:
            Array of confidence scores
        """
        # Different calculation based on signal type
        if self.signal_type == "binary":
            # For binary classification, confidence is distance from decision boundary
            confidence = np.abs(predictions - 0.5) * 2
            
            # Apply uncertainty adjustment if available
            if uncertainty is not None:
                # Normalize uncertainty to [0, 1]
                norm_uncertainty = np.clip(uncertainty / uncertainty.max(), 0, 1)
                
                # Adjust confidence by uncertainty (higher uncertainty -> lower confidence)
                confidence = confidence * (1 - norm_uncertainty)
        
        elif self.signal_type == "probability":
            # For probability outputs, confidence is the maximum probability
            confidence = np.max(predictions, axis=1) if len(predictions.shape) > 1 else predictions
            
            # Apply uncertainty adjustment if available
            if uncertainty is not None:
                # Normalize uncertainty
                norm_uncertainty = np.clip(uncertainty / uncertainty.max(), 0, 1)
                
                # Adjust confidence
                confidence = confidence * (1 - norm_uncertainty)
        
        elif self.signal_type == "multi_class":
            # For multi-class, confidence is the gap between top two probabilities
            if len(predictions.shape) > 1 and predictions.shape[1] > 1:
                # Sort probabilities in descending order
                sorted_preds = np.sort(predictions, axis=1)[:, ::-1]
                
                # Confidence is the gap between top two classes
                confidence = sorted_preds[:, 0] - sorted_preds[:, 1]
            else:
                # Only one class, use raw probability
                confidence = predictions
            
            # Apply uncertainty adjustment if available
            if uncertainty is not None:
                # Normalize uncertainty
                norm_uncertainty = np.clip(uncertainty / uncertainty.max(), 0, 1)
                
                # Adjust confidence
                confidence = confidence * (1 - norm_uncertainty)
        
        elif self.signal_type == "regression":
            # For regression, confidence is inverse of normalized uncertainty
            if uncertainty is not None:
                # Normalize uncertainty to [0, 1]
                norm_uncertainty = np.clip(uncertainty / uncertainty.max(), 0, 1)
                
                # Confidence is inverse of uncertainty
                confidence = 1 - norm_uncertainty
            else:
                # Without uncertainty, use fixed confidence
                confidence = np.ones_like(predictions) * 0.8
        
        else:
            logger.warning(f"Unknown signal type: {self.signal_type}, using default confidence")
            confidence = np.ones_like(predictions) * 0.5
        
        # Ensure confidence is in range [0, 1]
        confidence = np.clip(confidence, 0, 1)
        
        return confidence
    
    def _generate_trading_signals(
        self,
        predictions: np.ndarray,
        confidence: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Generate trading signals from model predictions.
        
        Args:
            predictions: Model predictions
            confidence: Confidence scores
            
        Returns:
            Array of trading signals
        """
        # Get confidence threshold
        threshold = self.confidence_threshold
        
        # Different generation based on signal type
        if self.signal_type == "binary":
            # Binary classification: 1 (buy), 0 (hold), -1 (sell)
            signals = np.zeros