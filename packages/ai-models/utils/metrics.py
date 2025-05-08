crps = y_std * (1/np.sqrt(np.pi) - 2*pdf - z_scores*(2*cdf - 1))
    metrics['mean_crps'] = np.mean(crps)
    
    # Interval scores for different quantiles
    alpha_values = [0.05, 0.1, 0.2]
    interval_scores = []
    
    for alpha in alpha_values:
        # Calculate prediction intervals
        z_alpha = np.percentile(abs(z_scores), (1 - alpha) * 100)
        lower = y_pred - z_alpha * y_std
        upper = y_pred + z_alpha * y_std
        
        # Calculate interval score
        interval_score = (upper - lower) + (2/alpha) * np.maximum(0, lower - y_true) + (2/alpha) * np.maximum(0, y_true - upper)
        interval_scores.append(np.mean(interval_score))
    
    metrics['mean_interval_score'] = np.mean(interval_scores)
    
    return metrics


def calculate_robustness_metrics(model, X: np.ndarray, y: np.ndarray, 
                               noise_level: float = 0.01,
                               n_perturbations: int = 10) -> Dict[str, float]:
    """
    Calculate robustness metrics by testing model sensitivity to input perturbations.
    
    Args:
        model: Model object with a predict method
        X: Features array
        y: True values array
        noise_level: Standard deviation of Gaussian noise as a fraction of input range
        n_perturbations: Number of perturbations to test
        
    Returns:
        Dictionary containing calculated metrics
    """
    if not hasattr(model, 'predict'):
        raise ValueError("Model must have a predict method")
    
    metrics = {}
    
    # Original predictions
    y_pred_original = model.predict(X)
    
    # Calculate input range for proportional noise
    input_range = np.max(X) - np.min(X)
    noise_std = noise_level * input_range
    
    # Initialize arrays for perturbed predictions
    all_perturbed_preds = []
    
    # Generate perturbed predictions
    for i in range(n_perturbations):
        # Add Gaussian noise to inputs
        X_noisy = X + np.random.normal(0, noise_std, size=X.shape)
        
        # Get predictions on noisy inputs
        y_pred_noisy = model.predict(X_noisy)
        all_perturbed_preds.append(y_pred_noisy)
    
    # Stack all predictions
    all_perturbed_preds = np.stack(all_perturbed_preds)
    
    # Calculate metrics
    
    # Mean prediction variance due to noise
    pred_variance = np.var(all_perturbed_preds, axis=0)
    metrics['mean_pred_variance'] = np.mean(pred_variance)
    
    # Coefficient of variation of predictions
    pred_mean = np.mean(all_perturbed_preds, axis=0)
    pred_std = np.std(all_perturbed_preds, axis=0)
    cv = pred_std / np.abs(pred_mean)
    cv[np.abs(pred_mean) < 1e-10] = 0  # Avoid division by zero
    metrics['mean_pred_cv'] = np.mean(cv)
    
    # Maximum prediction difference due to noise
    pred_max_diff = np.max(all_perturbed_preds, axis=0) - np.min(all_perturbed_preds, axis=0)
    metrics['mean_max_diff'] = np.mean(pred_max_diff)
    metrics['relative_max_diff'] = np.mean(pred_max_diff / np.abs(y_pred_original))
    
    # Stability score (1 - average normalized variance)
    normalized_variance = pred_variance / (np.abs(y_pred_original) ** 2 + 1e-10)
    metrics['stability_score'] = 1 - np.mean(normalized_variance)
    
    # Direction consistency (how often perturbed predictions have same direction as original)
    direction_original = np.sign(y_pred_original)
    direction_consistency = []
    
    for i in range(n_perturbations):
        direction_match = np.sign(all_perturbed_preds[i]) == direction_original
        direction_consistency.append(np.mean(direction_match))
    
    metrics['direction_consistency'] = np.mean(direction_consistency)
    
    # Performance robustness (how much does performance degrade with noise)
    original_rmse = np.sqrt(mean_squared_error(y, y_pred_original))
    perturbed_rmses = []
    
    for i in range(n_perturbations):
        perturbed_rmse = np.sqrt(mean_squared_error(y, all_perturbed_preds[i]))
        perturbed_rmses.append(perturbed_rmse)
    
    avg_perturbed_rmse = np.mean(perturbed_rmses)
    metrics['rmse_degradation'] = (avg_perturbed_rmse - original_rmse) / original_rmse
    
    return metrics


# ==========================================
# Specialized Metrics for Time Series Models
# ==========================================

def calculate_time_series_metrics(y_true: np.ndarray, y_pred: np.ndarray, 
                                 horizon: int = 1) -> Dict[str, float]:
    """
    Calculate specialized metrics for time series predictions.
    
    Args:
        y_true: Array of true values
        y_pred: Array of predicted values
        horizon: Prediction horizon
        
    Returns:
        Dictionary containing calculated metrics
    """
    if len(y_true) != len(y_pred):
        raise ValueError(f"Length mismatch: y_true has {len(y_true)} elements, y_pred has {len(y_pred)}")
    
    metrics = {}
    
    # Standard Metrics
    metrics['rmse'] = np.sqrt(mean_squared_error(y_true, y_pred))
    metrics['mae'] = mean_absolute_error(y_true, y_pred)
    
    # Mean Absolute Scaled Error (MASE)
    # MASE = MAE / MAE of naive forecast
    # For naive forecast, we use y_t = y_{t-horizon}
    if len(y_true) > horizon:
        naive_pred = y_true[:-horizon]  # y_{t-horizon}
        naive_true = y_true[horizon:]   # y_t
        
        # Calculate MAE of naive forecast
        naive_mae = mean_absolute_error(naive_true, naive_pred)
        
        # Calculate MASE
        if naive_mae > 0:
            metrics['mase'] = metrics['mae'] / naive_mae
        else:
            metrics['mase'] = float('inf')
    else:
        metrics['mase'] = float('nan')
    
    # Normalized RMSE
    metrics['nrmse'] = metrics['rmse'] / (np.max(y_true) - np.min(y_true)) if (np.max(y_true) - np.min(y_true)) > 0 else float('inf')
    
    # Theil's U statistic (U2)
    # U2 = sqrt(sum((y_pred(t+1) - y_true(t+1))/y_true(t))^2) / sqrt(sum((y_true(t+1) - y_true(t))/y_true(t))^2)
    if len(y_true) > 1:
        # Calculate percent changes
        actual_pct_change = (y_true[1:] - y_true[:-1]) / y_true[:-1]
        pred_pct_change = (y_pred[1:] - y_true[:-1]) / y_true[:-1]
        
        # Remove inf/nan values
        valid_indices = ~(np.isnan(actual_pct_change) | np.isnan(pred_pct_change) | 
                         np.isinf(actual_pct_change) | np.isinf(pred_pct_change))
        
        actual_pct_change = actual_pct_change[valid_indices]
        pred_pct_change = pred_pct_change[valid_indices]
        
        if len(actual_pct_change) > 0 and np.sum(actual_pct_change**2) > 0:
            metrics['theil_u2'] = np.sqrt(np.sum((pred_pct_change - actual_pct_change)**2)) / np.sqrt(np.sum(actual_pct_change**2))
        else:
            metrics['theil_u2'] = float('nan')
    else:
        metrics['theil_u2'] = float('nan')
    
    # Autocorrelation of residuals
    residuals = y_true - y_pred
    if len(residuals) > 1:
        # Calculate lag-1 autocorrelation
        residual_mean = np.mean(residuals)
        numerator = np.sum((residuals[1:] - residual_mean) * (residuals[:-1] - residual_mean))
        denominator = np.sum((residuals - residual_mean) ** 2)
        
        if denominator > 0:
            metrics['residual_autocorr'] = numerator / denominator
        else:
            metrics['residual_autocorr'] = 0
    else:
        metrics['residual_autocorr'] = float('nan')
    
    # Forecast bias
    metrics['bias'] = np.mean(y_pred - y_true)
    metrics['relative_bias'] = metrics['bias'] / np.mean(y_true) if np.mean(y_true) != 0 else float('inf')
    
    # Tracking signal (cumulative sum of errors / MAD)
    mad = np.mean(np.abs(residuals))
    if mad > 0:
        metrics['tracking_signal'] = np.sum(residuals) / (len(residuals) * mad)
    else:
        metrics['tracking_signal'] = float('inf')
    
    return metrics


# ==========================================
# Metrics Aggregation and Reporting
# ==========================================

def evaluate_model(model, X_test: np.ndarray, y_test: np.ndarray, 
                  prev_values: Optional[np.ndarray] = None,
                  task_type: str = 'regression',
                  model_name: str = 'model',
                  additional_metrics: Optional[Dict[str, Callable]] = None) -> Dict[str, Any]:
    """
    Comprehensive model evaluation that combines multiple metric types.
    
    Args:
        model: Model object with a predict method
        X_test: Test features
        y_test: Test target values
        prev_values: Previous values for directional and trading metrics
        task_type: Type of task ('regression', 'classification', 'trading', 'portfolio')
        model_name: Name of the model for reporting
        additional_metrics: Additional custom metrics to include
        
    Returns:
        Dictionary containing all calculated metrics
    """
    if not hasattr(model, 'predict'):
        raise ValueError("Model must have a predict method")
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Initialize results dictionary
    results = {
        'model_name': model_name,
        'task_type': task_type,
        'n_samples': len(y_test)
    }
    
    # Calculate basic regression metrics
    regression_metrics = calculate_regression_metrics(y_test, y_pred)
    results.update({f'regression_{k}': v for k, v in regression_metrics.items()})
    
    # Calculate directional metrics if prev_values provided
    if prev_values is not None:
        if len(prev_values) != len(y_test):
            raise ValueError(f"Length mismatch: prev_values has {len(prev_values)} elements, expected {len(y_test)}")
        
        directional_metrics = calculate_directional_metrics(y_test, y_pred, prev_values)
        results.update({f'directional_{k}': v for k, v in directional_metrics.items()})
        
        # For trading tasks, calculate trading metrics
        if task_type == 'trading':
            trading_metrics = calculate_trading_metrics(y_test, y_pred, prev_values)
            results.update({f'trading_{k}': v for k, v in trading_metrics.items()})
    
    # Calculate time series metrics
    if task_type in ['regression', 'trading', 'time_series']:
        ts_metrics = calculate_time_series_metrics(y_test, y_pred)
        results.update({f'time_series_{k}': v for k, v in ts_metrics.items()})
    
    # For probabilistic models with prediction intervals
    if hasattr(model, 'predict_std') or hasattr(model, 'predict_interval'):
        try:
            # Try to get standard deviations
            if hasattr(model, 'predict_std'):
                y_std = model.predict_std(X_test)
            elif hasattr(model, 'predict_interval'):
                lower, upper = model.predict_interval(X_test)
                # Estimate std from interval assuming normal distribution
                y_std = (upper - lower) / (2 * 1.96)  # 95% interval
                
            # Calculate calibration metrics
            calibration_metrics = calculate_calibration_metrics(y_test, y_pred, y_std)
            results.update({f'calibration_{k}': v for k, v in calibration_metrics.items()})
        except Exception as e:
            print(f"Error calculating calibration metrics: {str(e)}")
    
    # Add robustness metrics
    try:
        robustness_metrics = calculate_robustness_metrics(model, X_test, y_test)
        results.update({f'robustness_{k}': v for k, v in robustness_metrics.items()})
    except Exception as e:
        print(f"Error calculating robustness metrics: {str(e)}")
    
    # Add any additional custom metrics
    if additional_metrics:
        for name, metric_fn in additional_metrics.items():
            try:
                metric_value = metric_fn(y_test, y_pred)
                results[name] = metric_value
            except Exception as e:
                print(f"Error calculating custom metric '{name}': {str(e)}")
                results[name] = float('nan')
    
    return results


def generate_metrics_report(results: Dict[str, Any], 
                          include_plot: bool = True,
                          output_format: str = 'text') -> Union[str, Dict[str, Any]]:
    """
    Generate a formatted report from metrics results.
    
    Args:
        results: Dictionary of metric results
        include_plot: Whether to include performance plots
        output_format: Format of report ('text', 'json', 'html', 'markdown')
        
    Returns:
        Report in the specified format
    """
    if output_format == 'json':
        # Clean up any non-serializable values
        clean_results = {}
        for k, v in results.items():
            if isinstance(v, (int, float, str, bool, list, dict)) and v is not None:
                if isinstance(v, float) and (np.isnan(v) or np.isinf(v)):
                    clean_results[k] = str(v)  # Convert NaN/Inf to strings
                else:
                    clean_results[k] = v
            else:
                clean_results[k] = str(v)
                
        return clean_results
    
    # Create text/markdown/html report
    model_name = results.get('model_name', 'Unknown Model')
    task_type = results.get('task_type', 'Unknown Task')
    n_samples = results.get('n_samples', 0)
    
    if output_format == 'text':
        report = [
            f"==== Metrics Report for {model_name} ====",
            f"Task: {task_type}",
            f"Samples: {n_samples}",
            ""
        ]
        
        # Organize metrics by category
        categories = {
            'Regression Metrics': 'regression_',
            'Directional Metrics': 'directional_',
            'Trading Metrics': 'trading_',
            'Time Series Metrics': 'time_series_',
            'Calibration Metrics': 'calibration_',
            'Robustness Metrics': 'robustness_'
        }
        
        for category_name, prefix in categories.items():
            category_metrics = {k.replace(prefix, ''): v for k, v in results.items() if k.startswith(prefix)}
            
            if category_metrics:
                report.append(f"--- {category_name} ---")
                for metric_name, value in category_metrics.items():
                    if isinstance(value, float):
                        report.append(f"{metric_name}: {value:.6f}")
                    else:
                        report.append(f"{metric_name}: {value}")
                report.append("")
        
        # Add other metrics
        other_metrics = {k: v for k, v in results.items() 
                        if not any(k.startswith(prefix) for prefix in categories.values())
                        and k not in ['model_name', 'task_type', 'n_samples']}
        
        if other_metrics:
            report.append("--- Other Metrics ---")
            for metric_name, value in other_metrics.items():
                if isinstance(value, float):
                    report.append(f"{metric_name}: {value:.6f}")
                else:
                    report.append(f"{metric_name}: {value}")
        
        return "\n".join(report)
    
    elif output_format == 'markdown':
        report = [
            f"# Metrics Report for {model_name}",
            f"**Task:** {task_type}",
            f"**Samples:** {n_samples}",
            ""
        ]
        
        # Organize metrics by category
        categories = {
            'Regression Metrics': 'regression_',
            'Directional Metrics': 'directional_',
            'Trading Metrics': 'trading_',
            'Time Series Metrics': 'time_series_',
            'Calibration Metrics': 'calibration_',
            'Robustness Metrics': 'robustness_'
        }
        
        for category_name, prefix in categories.items():
            category_metrics = {k.replace(prefix, ''): v for k, v in results.items() if k.startswith(prefix)}
            
            if category_metrics:
                report.append(f"## {category_name}")
                report.append("| Metric | Value |")
                report.append("| ------ | ----- |")
                
                for metric_name, value in category_metrics.items():
                    if isinstance(value, float):
                        report.append(f"| {metric_name} | {value:.6f} |")
                    else:
                        report.append(f"| {metric_name} | {value} |")
                report.append("")
        
        # Add other metrics
        other_metrics = {k: v for k, v in results.items() 
                        if not any(k.startswith(prefix) for prefix in categories.values())
                        and k not in ['model_name', 'task_type', 'n_samples']}
        
        if other_metrics:
            report.append("## Other Metrics")
            report.append("| Metric | Value |")
            report.append("| ------ | ----- |")
            
            for metric_name, value in other_metrics.items():
                if isinstance(value, float):
                    report.append(f"| {metric_name} | {value:.6f} |")
                else:
                    report.append(f"| {metric_name} | {value} |")
        
        return "\n".join(report)
    
    elif output_format == 'html':
        # Basic HTML report
        report = [
            f"<h1>Metrics Report for {model_name}</h1>",
            f"<p><strong>Task:</strong> {task_type}</p>",
            f"<p><strong>Samples:</strong> {n_samples}</p>"
        ]
        
        # Organize metrics by category
        categories = {
            'Regression Metrics': 'regression_',
            'Directional Metrics': 'directional_',
            'Trading Metrics': 'trading_',
            'Time Series Metrics': 'time_series_',
            'Calibration Metrics': 'calibration_',
            'Robustness Metrics': 'robustness_'
        }
        
        for category_name, prefix in categories.items():
            category_metrics = {k.replace(prefix, ''): v for k, v in results.items() if k.startswith(prefix)}
            
            if category_metrics:
                report.append(f"<h2>{category_name}</h2>")
                report.append("<table border='1'>")
                report.append("<tr><th>Metric</th><th>Value</th></tr>")
                
                for metric_name, value in category_metrics.items():
                    if isinstance(value, float):
                        report.append(f"<tr><td>{metric_name}</td><td>{value:.6f}</td></tr>")
                    else:
                        report.append(f"<tr><td>{metric_name}</td><td>{value}</td></tr>")
                
                report.append("</table>")
        
        # Add other metrics
        other_metrics = {k: v for k, v in results.items() 
                        if not any(k.startswith(prefix) for prefix in categories.values())
                        and k not in ['model_name', 'task_type', 'n_samples']}
        
        if other_metrics:
            report.append("<h2>Other Metrics</h2>")
            report.append("<table border='1'>")
            report.append("<tr><th>Metric</th><th>Value</th></tr>")
            
            for metric_name, value in other_metrics.items():
                if isinstance(value, float):
                    report.append(f"<tr><td>{metric_name}</td><td>{value:.6f}</td></tr>")
                else:
                    report.append(f"<tr><td>{metric_name}</td><td>{value}</td></tr>")
            
            report.append("</table>")
        
        return "\n".join(report)
    
    else:
        raise ValueError(f"Unsupported output format: {output_format}")


# ==========================================
# Utility Functions
# ==========================================

def find_best_model(models_results: List[Dict[str, Any]], 
                  primary_metric: str,
                  secondary_metrics: Optional[List[str]] = None,
                  higher_is_better: bool = True) -> Dict[str, Any]:
    """
    Find the best model based on specified metrics.
    
    Args:
        models_results: List of metric results for different models
        primary_metric: Main metric to optimize
        secondary_metrics: Additional metrics for tie-breaking
        higher_is_better: Whether higher metric values are better
        
    Returns:
        Results dictionary for the best model
    """
    if not models_results:
        raise ValueError("Empty models_results list")
    
    if secondary_metrics is None:
        secondary_metrics = []
    
    # Check if primary metric exists in all results
    for results in models_results:
        if primary_metric not in results:
            raise ValueError(f"Primary metric '{primary_metric}' not found in all model results")
    
    # Find best model based on primary metric
    if higher_is_better:
        best_models = [results for results in models_results 
                      if results[primary_metric] == max(r[primary_metric] for r in models_results)]
    else:
        best_models = [results for results in models_results 
                      if results[primary_metric] == min(r[primary_metric] for r in models_results)]
    
    # If we have a tie, use secondary metrics
    best_model = best_models[0]
    if len(best_models) > 1 and secondary_metrics:
        for metric in secondary_metrics:
            valid_models = [results for results in best_models if metric in results]
            
            if not valid_models:
                continue
                
            if higher_is_better:
                best_model = max(valid_models, key=lambda x: x[metric])
            else:
                best_model = min(valid_models, key=lambda x: x[metric])
            
            # If we have a clear winner, break
            if len([results for results in valid_models if results[metric] == best_model[metric]]) == 1:
                break
    
    return best_model


def metric_improvement(baseline_value: float, new_value: float, higher_is_better: bool = True) -> float:
    """
    Calculate the relative improvement between two metric values.
    
    Args:
        baseline_value: Reference value
        new_value: New value to compare
        higher_is_better: Whether higher metric values are better
        
    Returns:
        Improvement percentage (positive means improvement)
    """
    if baseline_value == 0:
        return float('inf') if (new_value > 0 and higher_is_better) or (new_value < 0 and not higher_is_better) else float('-inf')
    
    relative_change = (new_value - baseline_value) / abs(baseline_value)
    
    # Convert to improvement (positive means improvement)
    if higher_is_better:
        return relative_change * 100
    else:
        return -relative_change * 100


# ==========================================
# Metrics Registry
# ==========================================

class MetricsRegistry:
    """
    Registry for tracking model performance metrics over time.
    
    This class provides a way to track and compare metrics across different
    model versions, environments, and time periods. It supports:
    - Logging metrics for different models and versions
    - Comparing metrics between models
    - Tracking metric evolution over time
    - Detecting performance degradation
    """
    
    def __init__(self, storage_path: Optional[str] = None):
        """
        Initialize metrics registry.
        
        Args:
            storage_path: Path to store metrics data (None for in-memory only)
        """
        self.storage_path = storage_path
        self.metrics_data = []
        
        # Load existing data if available
        if storage_path and os.path.exists(storage_path):
            try:
                with open(storage_path, 'r') as f:
                    self.metrics_data = json.load(f)
            except Exception as e:
                print(f"Error loading metrics data: {str(e)}")
    
    def log_metrics(self, metrics: Dict[str, Any], 
                  model_name: str,
                  model_version: str,
                  dataset_name: str,
                  environment: str = 'development',
                  tags: Optional[List[str]] = None) -> None:
        """
        Log metrics for a model.
        
        Args:
            metrics: Dictionary of metrics
            model_name: Name of the model
            model_version: Version of the model
            dataset_name: Name of the dataset used
            environment: Environment (development, staging, production)
            tags: Additional tags for filtering
        """
        # Ensure metrics are serializable
        clean_metrics = {}
        for k, v in metrics.items():
            if isinstance(v, (int, float, str, bool, list, dict)) and v is not None:
                if isinstance(v, float) and (np.isnan(v) or np.isinf(v)):
                    clean_metrics[k] = str(v)  # Convert NaN/Inf to strings
                else:
                    clean_metrics[k] = v
            else:
                clean_metrics[k] = str(v)
        
        # Create metrics entry
        entry = {
            'model_name': model_name,
            'model_version': model_version,
            'dataset_name': dataset_name,
            'environment': environment,
            'tags': tags or [],
            'timestamp': datetime.now().isoformat(),
            'metrics': clean_metrics
        }
        
        # Add to data
        self.metrics_data.append(entry)
        
        # Save to storage if available
        if self.storage_path:
            try:
                with open(self.storage_path, 'w') as f:
                    json.dump(self.metrics_data, f, indent=2)
            except Exception as e:
                print(f"Error saving metrics data: {str(e)}")
    
    def get_metrics(self, model_name: Optional[str] = None, 
                  model_version: Optional[str] = None,
                  dataset_name: Optional[str] = None,
                  environment: Optional[str] = None,
                  tags: Optional[List[str]] = None,
                  start_date: Optional[str] = None,
                  end_date: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get metrics matching the specified filters.
        
        Args:
            model_name: Filter by model name
            model_version: Filter by model version
            dataset_name: Filter by dataset name
            environment: Filter by environment
            tags: Filter by tags (must have all specified tags)
            start_date: Filter by start date (ISO format)
            end_date: Filter by end date (ISO format)
            
        Returns:
            List of matching metrics entries
        """
        results = self.metrics_data
        
        # Apply filters
        if model_name:
            results = [entry for entry in results if entry['model_name'] == model_name]
            
        if model_version:
            results = [entry for entry in results if entry['model_version'] == model_version]
            
        if dataset_name:
            results = [entry for entry in results if entry['dataset_name'] == dataset_name]
            
        if environment:
            results = [entry for entry in results if entry['environment'] == environment]
            
        if tags:
            results = [entry for entry in results if all(tag in entry['tags'] for tag in tags)]
            
        if start_date:
            results = [entry for entry in results if entry['timestamp'] >= start_date]
            
        if end_date:
            results = [entry for entry in results if entry['timestamp'] <= end_date]
            
        return results
    
    def compare_models(self, model_names: List[str], 
                      metric_name: str,
                      environment: Optional[str] = None,
                      dataset_name: Optional[str] = None,
                      latest_only: bool = True) -> Dict[str, Any]:
        """
        Compare specific metric across different models.
        
        Args:
            model_names: List of model names to compare
            metric_name: Metric to compare
            environment: Filter by environment
            dataset_name: Filter by dataset name
            latest_only: Use only the latest version of each model
            
        Returns:
            Dictionary with comparison results
        """
        comparison = {}
        
        for model_name in model_names:
            # Get metrics for this model
            model_metrics = self.get_metrics(model_name=model_name, environment=environment, dataset_name=dataset_name)
            
            if not model_metrics:
                comparison[model_name] = None
                continue
                
            if latest_only:
                # Find latest version by timestamp
                latest_entry = max(model_metrics, key=lambda x: x['timestamp'])
                comparison[model_name] = latest_entry['metrics'].get(metric_name)
            else:
                # Return all versions
                comparison[model_name] = [entry['metrics'].get(metric_name) for entry in model_metrics]
        
        return comparison
    
    def track_metric_evolution(self, model_name: str, 
                             metric_name: str,
                             dataset_name: Optional[str] = None,
                             environment: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Track evolution of a specific metric over time for a model.
        
        Args:
            model_name: Model name
            metric_name: Metric to track
            dataset_name: Filter by dataset name
            environment: Filter by environment
            
        Returns:
            List of {timestamp, version, metric_value} entries
        """
        # Get metrics for this model
        model_metrics = self.get_metrics(model_name=model_name, environment=environment, dataset_name=dataset_name)
        
        # Extract timestamp, version and metric value
        evolution = []
        for entry in model_metrics:
            if metric_name in entry['metrics']:
                evolution.append({
                    'timestamp': entry['timestamp'],
                    'version': entry['model_version'],
                    'value': entry['metrics'][metric_name]
                })
        
        # Sort by timestamp
        evolution.sort(key=lambda x: x['timestamp'])
        
        return evolution
    
    def detect_performance_degradation(self, model_name: str,
                                     metric_name: str,
                                     threshold: float = 0.05,
                                     higher_is_better: bool = True,
                                     environment: str = 'production') -> Dict[str, Any]:
        """
        Detect performance degradation for a model in production.
        
        Args:
            model_name: Model name
            metric_name: Metric to monitor
            threshold: Threshold for significant degradation
            higher_is_better: Whether higher metric values are better
            environment: Environment to monitor
            
        Returns:
            Dictionary with degradation analysis
        """
        # Get production metrics for this model
        model_metrics = self.get_metrics(model_name=model_name, environment=environment)
        
        if len(model_metrics) < 2:
            return {
                'degradation_detected': False,
                'message': f"Insufficient data for model {model_name} in {environment}",
                'current_value': model_metrics[0]['metrics'].get(metric_name) if model_metrics else None,
                'baseline_value': None,
                'relative_change': None
            }
        
        # Sort by timestamp
        model_metrics.sort(key=lambda x: x['timestamp'])
        
        # Get current and previous values
        current_entry = model_metrics[-1]
        baseline_entry = model_metrics[-2]
        
        current_value = current_entry['metrics'].get(metric_name)
        baseline_value = baseline_entry['metrics'].get(metric_name)
        
        if current_value is None or baseline_value is None:
            return {
                'degradation_detected': False,
                'message': f"Metric {metric_name} not found in both current and baseline entries",
                'current_value': current_value,
                'baseline_value': baseline_value,
                'relative_change': None
            }
        
        # Calculate relative change
        if baseline_value == 0:
            relative_change = float('inf') if current_value > 0 else float('-inf')
        else:
            relative_change = (current_value - baseline_value) / abs(baseline_value)
        
        # Determine if degradation occurred
        if higher_is_better:
            degradation_detected = relative_change < -threshold
        else:
            degradation_detected = relative_change > threshold
        
        return {
            'degradation_detected': degradation_detected,
            'message': f"Performance degradation detected: {metric_name} changed by {relative_change*100:.2f}%" if degradation_detected else "No significant degradation detected",
            'current_value': current_value,
            'baseline_value': baseline_value,
            'relative_change': relative_change,
            'current_version': current_entry['model_version'],
            'baseline_version': baseline_entry['model_version']
        }


# Example usage
if __name__ == "__main__":
    # Example data
    y_true = np.array([10.5, 11.2, 12.0, 12.8, 13.5, 14.2, 13.8, 14.5, 15.2, 15.8])
    y_pred = np.array([10.2, 11.5, 11.7, 13.0, 13.2, 14.0, 14.1, 14.8, 15.0, 16.0])
    prev_values = np.array([10.0, 10.5, 11.2, 12.0, 12.8, 13.5, 14.2, 13.8, 14.5, 15.2])
    
    # Calculate regression metrics
    regression_metrics = calculate_regression_metrics(y_true, y_pred)
    print("Regression Metrics:")
    for metric, value in regression_metrics.items():
        print(f"  {metric}: {value:.6f}")
    
    # Calculate directional metrics
    directional_metrics = calculate_directional_metrics(y_true, y_pred, prev_values)
    print("\nDirectional Metrics:")
    for metric, value in directional_metrics.items():
        print(f"  {metric}: {value:.6f}")
    
    # Calculate trading metrics
    trading_metrics = calculate_trading_metrics(y_true, y_pred, prev_values)
    print("\nTrading Metrics:")
    for metric, value in trading_metrics.items():
        print(f"  {metric}: {value:.6f}")
"""
Metrics Module for Minos-AI DeFi Strategy Platform

This module provides custom performance metrics and evaluation functions for ML models
on the Minos-AI platform. It extends beyond standard ML metrics to include
finance-specific metrics tailored for cryptocurrency trading and DeFi strategies.

The metrics fall into several categories:
- Regression metrics: Evaluate price/value prediction accuracy
- Classification metrics: Evaluate directional prediction accuracy
- Ranking metrics: Evaluate ordering of investment opportunities
- Financial metrics: Evaluate trading performance and risk-adjusted returns
- DeFi-specific metrics: Evaluate performance in DeFi contexts (yield, impermanent loss, etc.)

Integration Points:
- Used during model training for optimization objectives
- Used during model evaluation for performance reporting
- Used during backtesting to assess strategy performance
- Used in production monitoring to detect model drift

Author: Minos-AI Team
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Union, Tuple, Optional, Callable, Any
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error, 
    mean_absolute_percentage_error,
    r2_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    log_loss,
    roc_auc_score,
    precision_recall_curve,
    confusion_matrix
)
import warnings


# ==========================================
# Standard Regression Metrics
# ==========================================

def calculate_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray, 
                                sample_weight: Optional[np.ndarray] = None) -> Dict[str, float]:
    """
    Calculate standard regression metrics for evaluating price predictions.
    
    Args:
        y_true: Array of true values
        y_pred: Array of predicted values
        sample_weight: Optional weights for samples
        
    Returns:
        Dictionary containing calculated metrics
    """
    if len(y_true) != len(y_pred):
        raise ValueError(f"Length mismatch: y_true has {len(y_true)} elements, y_pred has {len(y_pred)}")
        
    if sample_weight is not None and len(sample_weight) != len(y_true):
        raise ValueError(f"Length mismatch: sample_weight has {len(sample_weight)} elements, expected {len(y_true)}")
    
    metrics = {}
    
    # Root Mean Squared Error
    metrics['rmse'] = np.sqrt(mean_squared_error(y_true, y_pred, sample_weight=sample_weight))
    
    # Mean Absolute Error
    metrics['mae'] = mean_absolute_error(y_true, y_pred, sample_weight=sample_weight)
    
    # Mean Absolute Percentage Error
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)  # Ignore div by zero warnings
        metrics['mape'] = mean_absolute_percentage_error(y_true, y_pred, sample_weight=sample_weight)
    
    # R-squared
    metrics['r2'] = r2_score(y_true, y_pred, sample_weight=sample_weight)
    
    # Custom Symmetric Mean Absolute Percentage Error (handles zero values better)
    with np.errstate(divide='ignore', invalid='ignore'):
        abs_diff = np.abs(y_true - y_pred)
        abs_sum = np.abs(y_true) + np.abs(y_pred)
        smape = 2 * abs_diff / abs_sum
        # Handle division by zero
        smape[abs_sum == 0] = 0
        
        if sample_weight is not None:
            metrics['smape'] = np.average(smape, weights=sample_weight) * 100
        else:
            metrics['smape'] = np.mean(smape) * 100
    
    # Median Absolute Error
    metrics['median_ae'] = np.median(np.abs(y_true - y_pred))
    
    # Maximum Absolute Error
    metrics['max_ae'] = np.max(np.abs(y_true - y_pred))
    
    # Coefficient of Variation of RMSE
    # Normalized RMSE by the mean of y_true to make it comparable across different scales
    with np.errstate(divide='ignore', invalid='ignore'):
        cv_rmse = metrics['rmse'] / np.mean(y_true) if np.mean(y_true) != 0 else np.inf
        metrics['cv_rmse'] = cv_rmse
    
    return metrics


# ==========================================
# Directional Accuracy Metrics
# ==========================================

def calculate_directional_metrics(y_true: np.ndarray, y_pred: np.ndarray, 
                                 prev_values: np.ndarray,
                                 threshold: float = 0.0,
                                 sample_weight: Optional[np.ndarray] = None) -> Dict[str, float]:
    """
    Calculate directional accuracy metrics for evaluating price movement predictions.
    
    Args:
        y_true: Array of true values (prices or returns)
        y_pred: Array of predicted values (prices or returns)
        prev_values: Array of previous values to calculate direction against
        threshold: Threshold for defining direction change (default 0.0 means any change)
        sample_weight: Optional weights for samples
        
    Returns:
        Dictionary containing calculated metrics
    """
    if len(y_true) != len(y_pred) or len(y_true) != len(prev_values):
        raise ValueError(f"Length mismatch between arrays")
    
    # Calculate actual and predicted directions
    true_direction = np.sign(y_true - prev_values)
    pred_direction = np.sign(y_pred - prev_values)
    
    # Apply threshold if needed
    if threshold > 0:
        true_pct_change = (y_true - prev_values) / prev_values
        pred_pct_change = (y_pred - prev_values) / prev_values
        
        true_direction = np.zeros_like(true_pct_change)
        true_direction[true_pct_change > threshold] = 1
        true_direction[true_pct_change < -threshold] = -1
        
        pred_direction = np.zeros_like(pred_pct_change)
        pred_direction[pred_pct_change > threshold] = 1
        pred_direction[pred_pct_change < -threshold] = -1
    
    # Convert to binary for standard classification metrics
    # 1 for up, 0 for down
    true_binary = (true_direction > 0).astype(int)
    pred_binary = (pred_direction > 0).astype(int)
    
    metrics = {}
    
    # Directional Accuracy
    metrics['directional_accuracy'] = accuracy_score(true_binary, pred_binary, 
                                                   sample_weight=sample_weight)
    
    # Precision (How many predicted ups are actual ups)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
        metrics['precision'] = precision_score(true_binary, pred_binary, 
                                             sample_weight=sample_weight, zero_division=0)
    
    # Recall (How many actual ups are predicted correctly)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
        metrics['recall'] = recall_score(true_binary, pred_binary, 
                                       sample_weight=sample_weight, zero_division=0)
    
    # F1 Score (Harmonic mean of precision and recall)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
        metrics['f1'] = f1_score(true_binary, pred_binary, 
                               sample_weight=sample_weight, zero_division=0)
    
    # Confusion Matrix derived metrics
    cm = confusion_matrix(true_binary, pred_binary, sample_weight=sample_weight)
    
    # Ensure we have a proper 2x2 matrix even if some classes aren't present
    if cm.shape == (1, 1):
        if true_binary[0] == 1:  # Only positive class
            cm = np.array([[0, 0], [0, cm[0, 0]]])
        else:  # Only negative class
            cm = np.array([[cm[0, 0], 0], [0, 0]])
    elif cm.shape == (2, 1):
        if pred_binary[0] == 1:  # Only predicted positive
            cm = np.array([[0, cm[0, 0]], [0, cm[1, 0]]])
        else:  # Only predicted negative
            cm = np.array([[cm[0, 0], 0], [cm[1, 0], 0]])
    
    # Extract values from confusion matrix
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        
        # Additional metrics useful for trading
        metrics['true_positive_rate'] = tp / (tp + fn) if (tp + fn) > 0 else 0
        metrics['false_positive_rate'] = fp / (fp + tn) if (fp + tn) > 0 else 0
        metrics['true_negative_rate'] = tn / (tn + fp) if (tn + fp) > 0 else 0
        metrics['false_negative_rate'] = fn / (fn + tp) if (fn + tp) > 0 else 0
        
        # Balanced accuracy (average of true positive rate and true negative rate)
        metrics['balanced_accuracy'] = (metrics['true_positive_rate'] + 
                                      metrics['true_negative_rate']) / 2
        
        # Positive predictive value and negative predictive value
        metrics['positive_predictive_value'] = tp / (tp + fp) if (tp + fp) > 0 else 0
        metrics['negative_predictive_value'] = tn / (tn + fn) if (tn + fn) > 0 else 0
    
    return metrics


# ==========================================
# Financial Trading Metrics
# ==========================================

def calculate_trading_metrics(y_true: np.ndarray, y_pred: np.ndarray, 
                             prev_values: np.ndarray,
                             initial_capital: float = 10000.0,
                             transaction_fee: float = 0.001,
                             risk_free_rate: float = 0.02 / 365) -> Dict[str, float]:
    """
    Calculate trading performance metrics for a simple long/short strategy.
    
    Args:
        y_true: Array of true prices
        y_pred: Array of predicted prices
        prev_values: Array of previous prices
        initial_capital: Initial capital for backtesting
        transaction_fee: Fee per transaction as a percentage (default 0.1%)
        risk_free_rate: Daily risk-free rate (default 2% annual converted to daily)
        
    Returns:
        Dictionary containing calculated metrics
    """
    if len(y_true) != len(y_pred) or len(y_true) != len(prev_values):
        raise ValueError(f"Length mismatch between arrays")
    
    # Calculate predicted directions (1 for long, -1 for short, 0 for hold)
    pred_direction = np.sign(y_pred - prev_values)
    
    # Calculate actual returns
    actual_returns = (y_true - prev_values) / prev_values
    
    # Strategy returns (excluding transaction costs for now)
    strategy_returns = pred_direction * actual_returns
    
    # Calculate transaction costs
    position_changes = np.diff(np.append(0, pred_direction))
    position_changes = np.abs(position_changes)
    transaction_costs = position_changes * transaction_fee
    
    # Adjust strategy returns for transaction costs
    strategy_returns_net = strategy_returns - transaction_costs
    
    # Calculate cumulative returns
    cumulative_returns = np.cumprod(1 + strategy_returns_net) - 1
    
    # Calculate portfolio value
    portfolio_value = initial_capital * (1 + cumulative_returns)
    
    metrics = {}
    
    # Total Return
    metrics['total_return'] = cumulative_returns[-1]
    
    # Annualized Return (assuming 252 trading days per year)
    num_days = len(strategy_returns_net)
    if num_days > 0:
        metrics['annualized_return'] = (1 + metrics['total_return']) ** (252 / num_days) - 1
    else:
        metrics['annualized_return'] = 0
    
    # Sharpe Ratio
    daily_returns = np.diff(np.append(1, 1 + strategy_returns_net))
    excess_returns = daily_returns - risk_free_rate
    if np.std(excess_returns) > 0:
        metrics['sharpe_ratio'] = (np.mean(excess_returns) / np.std(excess_returns)) * np.sqrt(252)
    else:
        metrics['sharpe_ratio'] = 0
    
    # Sortino Ratio (penalizes only negative volatility)
    downside_returns = excess_returns[excess_returns < 0]
    if len(downside_returns) > 0 and np.std(downside_returns) > 0:
        metrics['sortino_ratio'] = (np.mean(excess_returns) / np.std(downside_returns)) * np.sqrt(252)
    else:
        metrics['sortino_ratio'] = 0
    
    # Maximum Drawdown
    peak = np.maximum.accumulate(portfolio_value)
    drawdown = (peak - portfolio_value) / peak
    metrics['max_drawdown'] = np.max(drawdown) if len(drawdown) > 0 else 0
    
    # Calmar Ratio
    if metrics['max_drawdown'] > 0:
        metrics['calmar_ratio'] = metrics['annualized_return'] / metrics['max_drawdown']
    else:
        metrics['calmar_ratio'] = 0
    
    # Win Rate
    winning_trades = np.sum(strategy_returns_net > 0)
    total_trades = np.sum(position_changes > 0)
    metrics['win_rate'] = winning_trades / total_trades if total_trades > 0 else 0
    
    # Profit Factor
    gross_profits = np.sum(strategy_returns_net[strategy_returns_net > 0])
    gross_losses = np.abs(np.sum(strategy_returns_net[strategy_returns_net < 0]))
    metrics['profit_factor'] = gross_profits / gross_losses if gross_losses > 0 else float('inf')
    
    # Average Profit per Trade
    metrics['avg_profit_per_trade'] = np.mean(strategy_returns_net[position_changes > 0]) if total_trades > 0 else 0
    
    # Expectancy
    avg_win = np.mean(strategy_returns_net[strategy_returns_net > 0]) if len(strategy_returns_net[strategy_returns_net > 0]) > 0 else 0
    avg_loss = np.mean(strategy_returns_net[strategy_returns_net < 0]) if len(strategy_returns_net[strategy_returns_net < 0]) > 0 else 0
    metrics['expectancy'] = (metrics['win_rate'] * avg_win) - ((1 - metrics['win_rate']) * np.abs(avg_loss))
    
    # Returns Volatility (annualized)
    metrics['volatility'] = np.std(daily_returns) * np.sqrt(252)
    
    # Information Ratio
    # Using simple buy and hold as benchmark
    benchmark_returns = (y_true[-1] / y_true[0]) - 1
    benchmark_daily_returns = actual_returns
    if num_days > 0:
        annualized_benchmark_return = (1 + benchmark_returns) ** (252 / num_days) - 1
        active_returns = daily_returns - benchmark_daily_returns
        if np.std(active_returns) > 0:
            metrics['information_ratio'] = (metrics['annualized_return'] - annualized_benchmark_return) / (np.std(active_returns) * np.sqrt(252))
        else:
            metrics['information_ratio'] = 0
    else:
        metrics['information_ratio'] = 0
    
    # Beta
    cov = np.cov(daily_returns, benchmark_daily_returns)[0, 1]
    benchmark_var = np.var(benchmark_daily_returns)
    metrics['beta'] = cov / benchmark_var if benchmark_var > 0 else 0
    
    # Alpha (annualized)
    metrics['alpha'] = metrics['annualized_return'] - (risk_free_rate * 252) - (metrics['beta'] * (annualized_benchmark_return - (risk_free_rate * 252)))
    
    # Maximum Consecutive Wins and Losses
    wins = strategy_returns_net > 0
    losses = strategy_returns_net < 0
    
    # Count consecutive occurrences
    win_streaks, loss_streaks = [], []
    current_win_streak, current_loss_streak = 0, 0
    
    for win, loss in zip(wins, losses):
        if win:
            current_win_streak += 1
            if current_loss_streak > 0:
                loss_streaks.append(current_loss_streak)
                current_loss_streak = 0
        elif loss:
            current_loss_streak += 1
            if current_win_streak > 0:
                win_streaks.append(current_win_streak)
                current_win_streak = 0
        else:
            if current_win_streak > 0:
                win_streaks.append(current_win_streak)
                current_win_streak = 0
            if current_loss_streak > 0:
                loss_streaks.append(current_loss_streak)
                current_loss_streak = 0
    
    # Add any remaining streaks
    if current_win_streak > 0:
        win_streaks.append(current_win_streak)
    if current_loss_streak > 0:
        loss_streaks.append(current_loss_streak)
    
    metrics['max_consecutive_wins'] = max(win_streaks) if win_streaks else 0
    metrics['max_consecutive_losses'] = max(loss_streaks) if loss_streaks else 0
    
    return metrics


# ==========================================
# DeFi-Specific Metrics
# ==========================================

def calculate_defi_metrics(returns: np.ndarray, 
                         predicted_returns: np.ndarray,
                         risks: Optional[np.ndarray] = None,
                         liquidity: Optional[np.ndarray] = None,
                         impermanent_loss: Optional[np.ndarray] = None,
                         gas_costs: Optional[np.ndarray] = None) -> Dict[str, float]:
    """
    Calculate DeFi-specific metrics for evaluating strategies.
    
    Args:
        returns: Array of actual returns from the strategy
        predicted_returns: Array of predicted returns
        risks: Optional array of risk scores
        liquidity: Optional array of liquidity measures
        impermanent_loss: Optional array of impermanent loss values for LP positions
        gas_costs: Optional array of gas costs in base currency
        
    Returns:
        Dictionary containing calculated metrics
    """
    metrics = {}
    
    # Risk-adjusted return metrics
    if risks is not None:
        # Risk-adjusted Sharpe-like ratio
        risk_adj_returns = returns / risks
        metrics['risk_adjusted_return'] = np.mean(risk_adj_returns)
        
        # Risk-Return Efficiency
        total_return = np.sum(returns)
        total_risk = np.sum(risks)
        metrics['risk_return_efficiency'] = total_return / total_risk if total_risk > 0 else 0
    
    # Liquidity-adjusted metrics
    if liquidity is not None:
        # Liquidity-weighted return
        liq_weighted_returns = returns * liquidity
        metrics['liquidity_weighted_return'] = np.sum(liq_weighted_returns) / np.sum(liquidity) if np.sum(liquidity) > 0 else 0
        
        # Illiquidity premium
        high_liq_mask = liquidity > np.median(liquidity)
        low_liq_mask = ~high_liq_mask
        
        high_liq_return = np.mean(returns[high_liq_mask]) if np.any(high_liq_mask) else 0
        low_liq_return = np.mean(returns[low_liq_mask]) if np.any(low_liq_mask) else 0
        
        metrics['illiquidity_premium'] = low_liq_return - high_liq_return
    
    # Impermanent Loss metrics
    if impermanent_loss is not None:
        # Average impermanent loss
        metrics['avg_impermanent_loss'] = np.mean(impermanent_loss)
        
        # IL-adjusted return
        il_adjusted_returns = returns - impermanent_loss
        metrics['il_adjusted_return'] = np.mean(il_adjusted_returns)
        
        # IL ratio (IL as a percentage of returns)
        metrics['il_ratio'] = np.mean(impermanent_loss) / np.mean(returns) if np.mean(returns) > 0 else float('inf')
    
    # Gas cost metrics
    if gas_costs is not None:
        # Gas-adjusted return
        gas_adjusted_returns = returns - gas_costs
        metrics['gas_adjusted_return'] = np.mean(gas_adjusted_returns)
        
        # Gas efficiency (return per unit of gas)
        gas_efficiency = returns / gas_costs
        metrics['gas_efficiency'] = np.mean(gas_efficiency[gas_costs > 0]) if np.any(gas_costs > 0) else 0
        
        # Gas cost ratio (gas as a percentage of returns)
        metrics['gas_cost_ratio'] = np.sum(gas_costs) / np.sum(returns) if np.sum(returns) > 0 else float('inf')
    
    # Prediction accuracy for DeFi returns
    # RMSE for return prediction
    metrics['returns_rmse'] = np.sqrt(mean_squared_error(returns, predicted_returns))
    
    # Directional accuracy for returns
    true_direction = np.sign(returns)
    pred_direction = np.sign(predicted_returns)
    metrics['returns_directional_accuracy'] = np.mean(true_direction == pred_direction)
    
    # Opportunity cost (difference between optimal and actual strategy)
    sorted_indices_actual = np.argsort(-returns)  # Descending order
    sorted_indices_pred = np.argsort(-predicted_returns)  # Descending order
    
    # Calculate returns if we followed predicted ordering vs actual optimal ordering
    # Assuming we allocate capital linearly based on rank
    n = len(returns)
    weights = np.linspace(1, 0, n)  # Linear allocation from highest to lowest
    
    optimal_allocation_return = np.sum(returns[sorted_indices_actual] * weights) / np.sum(weights)
    predicted_allocation_return = np.sum(returns[sorted_indices_pred] * weights) / np.sum(weights)
    
    metrics['opportunity_cost'] = optimal_allocation_return - predicted_allocation_return
    metrics['allocation_efficiency'] = predicted_allocation_return / optimal_allocation_return if optimal_allocation_return > 0 else 0
    
    return metrics


# ==========================================
# Specialized Metrics for Portfolio Optimization
# ==========================================

def calculate_portfolio_metrics(asset_returns: np.ndarray, 
                              predicted_weights: np.ndarray,
                              optimal_weights: Optional[np.ndarray] = None,
                              risk_free_rate: float = 0.02 / 252) -> Dict[str, float]:
    """
    Calculate metrics for evaluating portfolio optimizations.
    
    Args:
        asset_returns: 2D array of asset returns [time_steps, n_assets]
        predicted_weights: 1D array of portfolio weights
        optimal_weights: Optional 1D array of optimal portfolio weights
        risk_free_rate: Risk-free rate (default is 2% annual converted to daily)
        
    Returns:
        Dictionary containing calculated metrics
    """
    if len(predicted_weights) != asset_returns.shape[1]:
        raise ValueError(f"Dimension mismatch: asset_returns has {asset_returns.shape[1]} assets, but predicted_weights has {len(predicted_weights)} weights")
    
    if optimal_weights is not None and len(optimal_weights) != asset_returns.shape[1]:
        raise ValueError(f"Dimension mismatch: asset_returns has {asset_returns.shape[1]} assets, but optimal_weights has {len(optimal_weights)} weights")
    
    # Normalize weights to ensure they sum to 1
    predicted_weights = predicted_weights / np.sum(predicted_weights) if np.sum(predicted_weights) > 0 else predicted_weights
    
    if optimal_weights is not None:
        optimal_weights = optimal_weights / np.sum(optimal_weights) if np.sum(optimal_weights) > 0 else optimal_weights
    
    metrics = {}
    
    # Portfolio returns
    portfolio_returns = np.dot(asset_returns, predicted_weights)
    
    # Expected return
    metrics['expected_return'] = np.mean(portfolio_returns)
    
    # Volatility (standard deviation of returns)
    metrics['volatility'] = np.std(portfolio_returns)
    
    # Sharpe ratio
    excess_returns = portfolio_returns - risk_free_rate
    metrics['sharpe_ratio'] = np.mean(excess_returns) / np.std(excess_returns) if np.std(excess_returns) > 0 else 0
    
    # Downside volatility (standard deviation of negative returns)
    downside_returns = portfolio_returns[portfolio_returns < 0]
    metrics['downside_volatility'] = np.std(downside_returns) if len(downside_returns) > 0 else 0
    
    # Sortino ratio
    metrics['sortino_ratio'] = np.mean(excess_returns) / metrics['downside_volatility'] if metrics['downside_volatility'] > 0 else 0
    
    # Maximum drawdown
    cumulative_returns = np.cumprod(1 + portfolio_returns)
    peak = np.maximum.accumulate(cumulative_returns)
    drawdown = (peak - cumulative_returns) / peak
    metrics['max_drawdown'] = np.max(drawdown) if len(drawdown) > 0 else 0
    
    # Calmar ratio
    if metrics['max_drawdown'] > 0:
        annual_return = (1 + metrics['expected_return']) ** 252 - 1
        metrics['calmar_ratio'] = annual_return / metrics['max_drawdown']
    else:
        metrics['calmar_ratio'] = 0
    
    # Portfolio diversification
    metrics['diversification'] = 1 - np.sum(predicted_weights ** 2)  # 1 - Herfindahl-Hirschman Index
    
    # If optimal weights are provided, calculate additional metrics
    if optimal_weights is not None:
        # Weight RMSE
        metrics['weight_rmse'] = np.sqrt(mean_squared_error(optimal_weights, predicted_weights))
        
        # Weight correlation
        metrics['weight_correlation'] = np.corrcoef(optimal_weights, predicted_weights)[0, 1] if np.std(optimal_weights) > 0 and np.std(predicted_weights) > 0 else 0
        
        # Optimal portfolio returns
        optimal_portfolio_returns = np.dot(asset_returns, optimal_weights)
        
        # Tracking error
        return_differences = portfolio_returns - optimal_portfolio_returns
        metrics['tracking_error'] = np.std(return_differences)
        
        # Information ratio
        metrics['information_ratio'] = np.mean(return_differences) / metrics['tracking_error'] if metrics['tracking_error'] > 0 else 0
        
        # Relative Sharpe ratio
        optimal_sharpe = np.mean(optimal_portfolio_returns - risk_free_rate) / np.std(optimal_portfolio_returns - risk_free_rate) if np.std(optimal_portfolio_returns - risk_free_rate) > 0 else 0
        metrics['relative_sharpe'] = metrics['sharpe_ratio'] / optimal_sharpe if optimal_sharpe > 0 else float('inf')
    
    return metrics


# ==========================================
# Model Calibration and Robustness Metrics
# ==========================================

def calculate_calibration_metrics(y_true: np.ndarray, 
                                y_pred: np.ndarray,
                                y_std: np.ndarray) -> Dict[str, float]:
    """
    Calculate calibration metrics for probabilistic predictions.
    
    Args:
        y_true: Array of true values
        y_pred: Array of predicted values (means)
        y_std: Array of predicted standard deviations
        
    Returns:
        Dictionary containing calculated metrics
    """
    if len(y_true) != len(y_pred) or len(y_true) != len(y_std):
        raise ValueError(f"Length mismatch between arrays")
    
    metrics = {}
    
    # Z-scores (how many standard deviations the true value is from the prediction)
    z_scores = (y_true - y_pred) / y_std
    
    # Mean and std of z-scores (should be 0 and 1 for well-calibrated model)
    metrics['z_score_mean'] = np.mean(z_scores)
    metrics['z_score_std'] = np.std(z_scores)
    
    # Negative log-likelihood (assuming Gaussian distribution)
    nll = 0.5 * np.log(2 * np.pi * y_std ** 2) + 0.5 * ((y_true - y_pred) / y_std) ** 2
    metrics['mean_nll'] = np.mean(nll)
    
    # Calibration error (difference between expected and actual coverage)
    coverages = []
    expected_coverages = [0.5, 0.8, 0.9, 0.95, 0.99]
    
    for p in expected_coverages:
        z_score = np.abs(z_scores)
        z_threshold = np.percentile(z_score, p * 100)
        actual_coverage = np.mean(z_score <= z_threshold)
        coverage_error = np.abs(actual_coverage - p)
        coverages.append(coverage_error)
    
    metrics['mean_coverage_error'] = np.mean(coverages)
    metrics['max_coverage_error'] = np.max(coverages)
    
    # Continuous Ranked Probability Score (CRPS)
    # Approximated using the closed-form solution for Gaussian distributions
    # CRPS = y_std * (1/sqrt(pi) - 2*pdf(z) - z*(2*cdf(z) - 1))
    # where z = (y_true - y_pred) / y_std, pdf is the standard normal PDF, and cdf is the standard normal CDF
    
    # Standard normal PDF and CDF
    pdf = np.exp(-0.5 * z_scores ** 2) / np.sqrt(2 * np.pi)
    cdf = 0.5 * (1 + np.erf(z_scores / np.sqrt(2)))
    
    crps = y_std * (1/np.sqrt(np.pi) - 2*pdf - z_scores*(2*cdf - 1))
    metrics['mean_cr