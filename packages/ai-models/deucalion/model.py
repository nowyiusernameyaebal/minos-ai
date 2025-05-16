"""
Deucalion AI Model - DeFi Investment Strategy Optimization
Sophisticated machine learning model for analyzing market trends, social sentiment,
and DeFi protocols to generate optimal investment strategies.

Architecture:
- Multi-modal transformer architecture combining numerical and textual data
- Ensemble methods for risk-adjusted returns prediction
- Reinforcement learning for adaptive strategy optimization
- Advanced feature engineering for DeFi-specific metrics
"""

import json
import logging
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, asdict
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModel,
    AutoConfig,
    BertModel,
    RobertaModel
)
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import IsolationForest
from scipy.stats import zscore
import optuna
import joblib
from datetime import datetime, timedelta
import asyncio
import aiohttp
from concurrent.futures import ThreadPoolExecutor
import warnings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore', category=FutureWarning)


@dataclass
class ModelConfig:
    """Configuration for the Deucalion model"""
    # Model architecture
    model_type: str = "transformer_ensemble"
    hidden_dim: int = 768
    num_heads: int = 12
    num_layers: int = 6
    dropout: float = 0.1
    
    # Training parameters
    learning_rate: float = 2e-5
    batch_size: int = 32
    max_epochs: int = 100
    early_stopping_patience: int = 10
    weight_decay: float = 0.01
    
    # Data processing
    sequence_length: int = 144  # 3 days of hourly data
    prediction_horizon: int = 24  # 24 hours
    feature_dim: int = 256
    
    # Strategy parameters
    risk_tolerance: float = 0.02
    max_leverage: float = 3.0
    rebalancing_frequency: int = 6  # hours
    confidence_threshold: float = 0.75
    
    # Model ensemble
    num_models: int = 5
    voting_strategy: str = "weighted"
    
    # Paths
    model_dir: str = "./models/"
    data_dir: str = "./data/"
    config_path: str = "./config.json"


@dataclass
class MarketState:
    """Represents current market state for decision making"""
    timestamp: datetime
    prices: Dict[str, float]
    volumes: Dict[str, float]
    liquidity: Dict[str, float]
    volatility: Dict[str, float]
    social_sentiment: Dict[str, float]
    technical_indicators: Dict[str, Dict[str, float]]
    protocol_metrics: Dict[str, Any]
    risk_metrics: Dict[str, float]


@dataclass
class Strategy:
    """Investment strategy output"""
    timestamp: datetime
    allocations: Dict[str, float]  # Protocol -> allocation percentage
    actions: List[Dict[str, Any]]  # List of suggested actions
    confidence: float
    risk_score: float
    expected_return: float
    max_drawdown: float
    reasoning: str


class MultiModalTransformer(nn.Module):
    """
    Multi-modal transformer for processing both numerical time series
    and textual social sentiment data
    """
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        
        # Text encoder (for social sentiment)
        self.text_encoder = BertModel.from_pretrained('bert-base-uncased')
        self.text_projection = nn.Linear(768, config.hidden_dim)
        
        # Numerical encoder (for market data)
        self.numerical_projection = nn.Linear(config.feature_dim, config.hidden_dim)
        
        # Position encoding
        self.position_embedding = nn.Embedding(config.sequence_length, config.hidden_dim)
        
        # Transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_dim,
            nhead=config.num_heads,
            dim_feedforward=config.hidden_dim * 4,
            dropout=config.dropout,
            activation='gelu'
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, config.num_layers)
        
        # Output heads
        self.price_head = nn.Linear(config.hidden_dim, 1)
        self.volatility_head = nn.Linear(config.hidden_dim, 1)
        self.risk_head = nn.Linear(config.hidden_dim, 1)
        self.confidence_head = nn.Linear(config.hidden_dim, 1)
        
        # Strategy generator
        self.strategy_head = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim // 2, config.hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(config.hidden_dim // 4, 100)  # Strategy allocation logits
        )
        
        self.dropout = nn.Dropout(config.dropout)
        self.layer_norm = nn.LayerNorm(config.hidden_dim)
    
    def forward(self, 
                numerical_data: torch.Tensor,
                text_data: Optional[Dict] = None,
                attention_mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the multi-modal transformer
        
        Returns:
            Dictionary containing predictions for price, volatility, risk, and strategies
        """
        batch_size, seq_len, _ = numerical_data.shape
        
        # Process numerical data
        numerical_embeddings = self.numerical_projection(numerical_data)
        
        # Process text data if available
        if text_data is not None:
            text_embeddings = self.text_encoder(**text_data).last_hidden_state
            text_embeddings = self.text_projection(text_embeddings)
            # Average pool text embeddings to match sequence length
            text_embeddings = text_embeddings.mean(dim=1, keepdim=True)
            text_embeddings = text_embeddings.expand(-1, seq_len, -1)
            
            # Combine modalities
            embeddings = numerical_embeddings + text_embeddings * 0.3
        else:
            embeddings = numerical_embeddings
        
        # Add position embeddings
        positions = torch.arange(seq_len, device=embeddings.device).unsqueeze(0)
        position_embeddings = self.position_embedding(positions)
        embeddings = embeddings + position_embeddings
        
        # Apply layer norm and dropout
        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)
        
        # Transpose for transformer (seq_len, batch_size, hidden_dim)
        embeddings = embeddings.transpose(0, 1)
        
        # Pass through transformer
        output = self.transformer(embeddings, src_key_padding_mask=attention_mask)
        
        # Use the last hidden state for predictions
        output = output[-1]  # (batch_size, hidden_dim)
        
        # Generate predictions
        predictions = {
            'price': self.price_head(output),
            'volatility': torch.sigmoid(self.volatility_head(output)),
            'risk': torch.sigmoid(self.risk_head(output)),
            'confidence': torch.sigmoid(self.confidence_head(output)),
            'strategy': F.softmax(self.strategy_head(output), dim=-1)
        }
        
        return predictions


class DeucalionDataset(Dataset):
    """
    Dataset class for training the Deucalion model
    Handles both numerical market data and textual social sentiment
    """
    
    def __init__(self, 
                 data: pd.DataFrame,
                 config: ModelConfig,
                 tokenizer: Optional[Any] = None,
                 is_training: bool = True):
        self.data = data
        self.config = config
        self.tokenizer = tokenizer
        self.is_training = is_training
        
        # Extract features and targets
        self._prepare_sequences()
    
    def _prepare_sequences(self):
        """Prepare input sequences and targets"""
        self.sequences = []
        self.targets = {}
        
        # Numerical features
        numerical_columns = [col for col in self.data.columns if 'text' not in col.lower()]
        numerical_data = self.data[numerical_columns].values
        
        # Normalize numerical data
        self.scaler = StandardScaler()
        numerical_data = self.scaler.fit_transform(numerical_data)
        
        # Create sequences
        for i in range(len(self.data) - self.config.sequence_length - self.config.prediction_horizon):
            sequence = numerical_data[i:i + self.config.sequence_length]
            
            # Extract targets
            future_prices = self.data.iloc[i + self.config.sequence_length + 1:
                                         i + self.config.sequence_length + self.config.prediction_horizon + 1]['price'].values
            future_volatility = np.std(future_prices)
            future_returns = (future_prices[-1] - future_prices[0]) / future_prices[0]
            
            self.sequences.append(sequence)
            if 'price' not in self.targets:
                self.targets['price'] = []
                self.targets['volatility'] = []
                self.targets['returns'] = []
            
            self.targets['price'].append(future_prices[-1])
            self.targets['volatility'].append(future_volatility)
            self.targets['returns'].append(future_returns)
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        sequence = torch.tensor(self.sequences[idx], dtype=torch.float32)
        
        item = {
            'numerical_data': sequence,
            'targets': {
                'price': torch.tensor(self.targets['price'][idx], dtype=torch.float32),
                'volatility': torch.tensor(self.targets['volatility'][idx], dtype=torch.float32),
                'returns': torch.tensor(self.targets['returns'][idx], dtype=torch.float32)
            }
        }
        
        # Add text data if available
        if 'sentiment_text' in self.data.columns:
            text = self.data.iloc[idx]['sentiment_text']
            if self.tokenizer:
                encoded = self.tokenizer(
                    text,
                    max_length=512,
                    padding='max_length',
                    truncation=True,
                    return_tensors='pt'
                )
                item['text_data'] = {
                    'input_ids': encoded['input_ids'].squeeze(),
                    'attention_mask': encoded['attention_mask'].squeeze()
                }
        
        return item


class ModelEnsemble:
    """
    Ensemble of multiple models for robust predictions
    Implements different aggregation strategies
    """
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.models = []
        self.weights = []
        self.performance_history = []
    
    def add_model(self, model: nn.Module, weight: float = 1.0):
        """Add a model to the ensemble"""
        self.models.append(model)
        self.weights.append(weight)
    
    def predict(self, *args, **kwargs) -> Dict[str, torch.Tensor]:
        """Generate ensemble predictions"""
        predictions = []
        
        for model in self.models:
            with torch.no_grad():
                pred = model(*args, **kwargs)
                predictions.append(pred)
        
        # Aggregate predictions
        if self.config.voting_strategy == "weighted":
            return self._weighted_average(predictions)
        elif self.config.voting_strategy == "median":
            return self._median_aggregate(predictions)
        else:
            return self._simple_average(predictions)
    
    def _weighted_average(self, predictions: List[Dict]) -> Dict[str, torch.Tensor]:
        """Weighted average aggregation"""
        result = {}
        total_weight = sum(self.weights)
        
        for key in predictions[0].keys():
            weighted_sum = sum(pred[key] * weight for pred, weight in zip(predictions, self.weights))
            result[key] = weighted_sum / total_weight
        
        return result
    
    def _median_aggregate(self, predictions: List[Dict]) -> Dict[str, torch.Tensor]:
        """Median aggregation for robustness"""
        result = {}
        
        for key in predictions[0].keys():
            stacked = torch.stack([pred[key] for pred in predictions])
            result[key] = torch.median(stacked, dim=0)[0]
        
        return result
    
    def _simple_average(self, predictions: List[Dict]) -> Dict[str, torch.Tensor]:
        """Simple average aggregation"""
        result = {}
        
        for key in predictions[0].keys():
            result[key] = torch.mean(torch.stack([pred[key] for pred in predictions]), dim=0)
        
        return result


class DeucalionModel:
    """
    Main Deucalion AI model class
    Orchestrates training, inference, and strategy generation
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize the Deucalion model"""
        self.config = self._load_config(config_path)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize components
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        self.model = MultiModalTransformer(self.config)
        self.ensemble = ModelEnsemble(self.config)
        self.optimizer = None
        self.scheduler = None
        
        # Risk management
        self.risk_manager = self._initialize_risk_manager()
        
        # Performance tracking
        self.performance_metrics = {
            'sharpe_ratio': [],
            'max_drawdown': [],
            'win_rate': [],
            'total_return': []
        }
        
        # Anomaly detection
        self.anomaly_detector = IsolationForest(contamination=0.1, random_state=42)
        
        logger.info(f"Deucalion model initialized on {self.device}")
    
    def _load_config(self, config_path: Optional[str]) -> ModelConfig:
        """Load configuration from file or use defaults"""
        if config_path and Path(config_path).exists():
            with open(config_path, 'r') as f:
                config_dict = json.load(f)
            return ModelConfig(**config_dict)
        else:
            return ModelConfig()
    
    def _initialize_risk_manager(self) -> Dict:
        """Initialize risk management parameters"""
        return {
            'max_position_size': 0.2,  # Maximum 20% per position
            'stop_loss': 0.05,         # 5% stop loss
            'take_profit': 0.15,       # 15% take profit
            'correlation_limit': 0.8,  # Maximum correlation between positions
            'volatility_threshold': 0.1 # Maximum volatility threshold
        }
    
    async def train(self, 
                   train_data: pd.DataFrame,
                   validation_data: pd.DataFrame,
                   save_path: Optional[str] = None) -> Dict[str, float]:
        """
        Train the Deucalion model
        
        Returns:
            Training metrics dictionary
        """
        logger.info("Starting model training...")
        
        # Prepare datasets
        train_dataset = DeucalionDataset(train_data, self.config, self.tokenizer, is_training=True)
        val_dataset = DeucalionDataset(validation_data, self.config, self.tokenizer, is_training=False)
        
        train_loader = DataLoader(train_dataset, batch_size=self.config.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.config.batch_size, shuffle=False)
        
        # Initialize optimizer and scheduler
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )
        
        self.scheduler = torch.optim.lr_scheduler.SchedulerLR(
            self.optimizer,
            step_size=10,
            gamma=0.9
        )
        
        # Training loop
        best_val_loss = float('inf')
        patience_counter = 0
        training_history = {'train_loss': [], 'val_loss': []}
        
        for epoch in range(self.config.max_epochs):
            # Training phase
            train_loss = await self._train_epoch(train_loader)
            
            # Validation phase
            val_loss, val_metrics = await self._validate(val_loader)
            
            # Update learning rate
            self.scheduler.step()
            
            # Track metrics
            training_history['train_loss'].append(train_loss)
            training_history['val_loss'].append(val_loss)
            
            logger.info(f"Epoch {epoch+1}/{self.config.max_epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                if save_path:
                    self._save_model(save_path)
            else:
                patience_counter += 1
                if patience_counter >= self.config.early_stopping_patience:
                    logger.info(f"Early stopping at epoch {epoch+1}")
                    break
        
        # Train ensemble with different architectures
        await self._train_ensemble(train_data, validation_data)
        
        logger.info("Model training completed")
        return training_history
    
    async def _train_epoch(self, train_loader: DataLoader) -> float:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        for batch in train_loader:
            self.optimizer.zero_grad()
            
            # Prepare inputs
            numerical_data = batch['numerical_data'].to(self.device)
            text_data = batch.get('text_data')
            if text_data:
                text_data = {k: v.to(self.device) for k, v in text_data.items()}
            
            # Forward pass
            predictions = self.model(numerical_data, text_data)
            
            # Calculate loss
            loss = self._calculate_loss(predictions, batch['targets'])
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        return total_loss / num_batches
    
    async def _validate(self, val_loader: DataLoader) -> Tuple[float, Dict]:
        """Validate the model"""
        self.model.eval()
        total_loss = 0
        num_batches = 0
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for batch in val_loader:
                numerical_data = batch['numerical_data'].to(self.device)
                text_data = batch.get('text_data')
                if text_data:
                    text_data = {k: v.to(self.device) for k, v in text_data.items()}
                
                predictions = self.model(numerical_data, text_data)
                loss = self._calculate_loss(predictions, batch['targets'])
                
                total_loss += loss.item()
                num_batches += 1
                
                # Store predictions for metrics calculation
                all_predictions.append(predictions)
                all_targets.append(batch['targets'])
        
        # Calculate validation metrics
        val_metrics = self._calculate_metrics(all_predictions, all_targets)
        
        return total_loss / num_batches, val_metrics
    
    def _calculate_loss(self, predictions: Dict, targets: Dict) -> torch.Tensor:
        """Calculate composite loss function"""
        # Price prediction loss (MSE)
        price_loss = F.mse_loss(predictions['price'], targets['price'].to(self.device))
        
        # Volatility prediction loss (MSE)
        vol_loss = F.mse_loss(predictions['volatility'], targets['volatility'].to(self.device))
        
        # Risk-adjusted loss
        risk_penalty = torch.mean(predictions['risk'] * torch.abs(predictions['price'] - targets['price'].to(self.device)))
        
        # Strategy consistency loss
        strategy_entropy = -torch.mean(torch.sum(predictions['strategy'] * torch.log(predictions['strategy'] + 1e-8), dim=1))
        
        # Combine losses
        total_loss = (
            price_loss * 0.4 +
            vol_loss * 0.3 +
            risk_penalty * 0.2 +
            strategy_entropy * 0.1
        )
        
        return total_loss
    
    def _calculate_metrics(self, predictions: List, targets: List) -> Dict:
        """Calculate validation metrics"""
        # Aggregate predictions and targets
        pred_prices = torch.cat([p['price'] for p in predictions])
        target_prices = torch.cat([t['price'] for t in targets])
        
        # Calculate metrics
        mse = F.mse_loss(pred_prices, target_prices).item()
        mae = F.l1_loss(pred_prices, target_prices).item()
        
        # Calculate directional accuracy
        pred_direction = torch.sign(pred_prices[1:] - pred_prices[:-1])
        target_direction = torch.sign(target_prices[1:] - target_prices[:-1])
        directional_accuracy = torch.mean((pred_direction == target_direction).float()).item()
        
        return {
            'mse': mse,
            'mae': mae,
            'directional_accuracy': directional_accuracy
        }
    
    async def _train_ensemble(self, train_data: pd.DataFrame, val_data: pd.DataFrame):
        """Train ensemble of models with different configurations"""
        logger.info("Training ensemble models...")
        
        # Different model configurations
        ensemble_configs = [
            {'hidden_dim': 512, 'num_layers': 4},
            {'hidden_dim': 1024, 'num_layers': 8},
            {'dropout': 0.2, 'learning_rate': 1e-5},
            {'num_heads': 8, 'dropout': 0.05},
            {'sequence_length': 96, 'prediction_horizon': 12}
        ]
        
        base_weights = []
        
        for i, config_updates in enumerate(ensemble_configs):
            logger.info(f"Training ensemble model {i+1}/5...")
            
            # Create modified config
            ensemble_config = ModelConfig(**{**asdict(self.config), **config_updates})
            
            # Create and train model
            ensemble_model = MultiModalTransformer(ensemble_config)
            ensemble_model.to(self.device)
            
            # Quick training with different hyperparameters
            optimizer = torch.optim.AdamW(
                ensemble_model.parameters(),
                lr=ensemble_config.learning_rate,
                weight_decay=ensemble_config.weight_decay
            )
            
            # Train for fewer epochs
            train_dataset = DeucalionDataset(train_data, ensemble_config, self.tokenizer)
            train_loader = DataLoader(train_dataset, batch_size=ensemble_config.batch_size, shuffle=True)
            
            # Training loop for ensemble model
            for epoch in range(10):  # Reduced epochs for ensemble
                ensemble_model.train()
                for batch in train_loader:
                    optimizer.zero_grad()
                    numerical_data = batch['numerical_data'].to(self.device)
                    predictions = ensemble_model(numerical_data)
                    loss = self._calculate_loss(predictions, batch['targets'])
                    loss.backward()
                    optimizer.step()
            
            # Add to ensemble
            self.ensemble.add_model(ensemble_model, weight=1.0)
            base_weights.append(1.0)
        
        # Optimize ensemble weights
        self.ensemble.weights = await self._optimize_ensemble_weights(val_data, base_weights)
        logger.info("Ensemble training completed")
    
    async def _optimize_ensemble_weights(self, val_data: pd.DataFrame, initial_weights: List[float]) -> List[float]:
        """Optimize ensemble weights using Optuna"""
        def objective(trial):
            weights = [trial.suggest_float(f'weight_{i}', 0.1, 2.0) for i in range(len(initial_weights))]
            
            # Normalize weights
            total_weight = sum(weights)
            weights = [w / total_weight for w in weights]
            
            # Evaluate ensemble with these weights
            self.ensemble.weights = weights
            val_loss = self._evaluate_ensemble(val_data)
            
            return val_loss
        
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=50, show_progress_bar=False)
        
        # Get best weights
        best_trial = study.best_trial
        best_weights = [best_trial.params[f'weight_{i}'] for i in range(len(initial_weights))]
        total_weight = sum(best_weights)
        best_weights = [w / total_weight for w in best_weights]
        
        logger.info(f"Optimized ensemble weights: {best_weights}")
        return best_weights
    
    def _evaluate_ensemble(self, val_data: pd.DataFrame) -> float:
        """Evaluate ensemble performance"""
        val_dataset = DeucalionDataset(val_data, self.config, self.tokenizer, is_training=False)
        val_loader = DataLoader(val_dataset, batch_size=self.config.batch_size, shuffle=False)
        
        total_loss = 0
        num_batches = 0
        
        with torch.no_grad():
            for batch in val_loader:
                numerical_data = batch['numerical_data'].to(self.device)
                predictions = self.ensemble.predict(numerical_data)
                loss = self._calculate_loss(predictions, batch['targets'])
                total_loss += loss.item()
                num_batches += 1
        
        return total_loss / num_batches
    
    async def predict(self, market_state: MarketState) -> Strategy:
        """
        Generate investment strategy based on current market state
        
        Args:
            market_state: Current market state with prices, volumes, sentiment, etc.
            
        Returns:
            Strategy object with allocations and recommendations
        """
        logger.info("Generating investment strategy...")
        
        # Prepare input data
        input_data = self._prepare_market_state(market_state)
        
        # Generate predictions using ensemble
        with torch.no_grad():
            predictions = self.ensemble.predict(input_data['numerical_data'])
        
        # Risk assessment
        risk_score = self._assess_risk(market_state, predictions)
        
        # Generate strategy
        strategy = self._generate_strategy(market_state, predictions, risk_score)
        
        # Validate strategy
        validated_strategy = self._validate_strategy(strategy)
        
        logger.info(f"Strategy generated with {validated_strategy.confidence:.2f} confidence")
        return validated_strategy
    
    def _prepare_market_state(self, market_state: MarketState) -> Dict[str, torch.Tensor]:
        """Prepare market state for model input"""
        # Extract numerical features
        features = []
        
        # Price features
        prices = list(market_state.prices.values())
        features.extend(prices)
        
        # Volume features
        volumes = list(market_state.volumes.values())
        features.extend(volumes)
        
        # Technical indicators
        for protocol, indicators in market_state.technical_indicators.items():
            features.extend(list(indicators.values()))
        
        # Social sentiment
        sentiments = list(market_state.social_sentiment.values())
        features.extend(sentiments)
        
        # Risk metrics
        risk_metrics = list(market_state.risk_metrics.values())
        features.extend(risk_metrics)
        
        # Pad or truncate to match expected feature dimension
        if len(features) < self.config.feature_dim:
            features.extend([0.0] * (self.config.feature_dim - len(features)))
        else:
            features = features[:self.config.feature_dim]
        
        # Create tensor
        numerical_data = torch.tensor(features, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        numerical_data = numerical_data.to(self.device)
        
        return {'numerical_data': numerical_data}
    
    def _assess_risk(self, market_state: MarketState, predictions: Dict) -> float:
        """Assess portfolio risk based on market state and predictions"""
        # Volatility risk
        volatility_risk = float(predictions['volatility'].item())
        
        # Market risk (using VaR calculation)
        returns = []
        for protocol in market_state.prices.keys():
            # Calculate recent returns (simplified)
            price = market_state.prices[protocol]
            volume = market_state.volumes[protocol]
            returns.append(price * volume)
        
        # Calculate Value at Risk (95% confidence)
        returns_array = np.array(returns)
        var_95 = np.percentile(returns_array, 5)
        market_risk = abs(var_95) / np.mean(returns_array) if np.mean(returns_array) != 0 else 0
        
        # Correlation risk
        correlation_risk = self._calculate_correlation_risk(market_state)
        
        # Liquidity risk
        liquidity_values = list(market_state.liquidity.values())
        liquidity_risk = 1.0 - (np.mean(liquidity_values) if liquidity_values else 0)
        
        # Combined risk score
        total_risk = (
            volatility_risk * 0.3 +
            market_risk * 0.3 +
            correlation_risk * 0.2 +
            liquidity_risk * 0.2
        )
        
        return min(max(total_risk, 0.0), 1.0)  # Clamp between 0 and 1
    
    def _calculate_correlation_risk(self, market_state: MarketState) -> float:
        """Calculate correlation risk between assets"""
        prices = list(market_state.prices.values())
        if len(prices) < 2:
            return 0.0
        
        # Calculate pairwise correlations (simplified using price movements)
        correlations = []
        for i in range(len(prices)):
            for j in range(i + 1, len(prices)):
                # Use price ratio as a proxy for correlation
                correlation = min(prices[i], prices[j]) / max(prices[i], prices[j])
                correlations.append(correlation)
        
        # Return average correlation as risk metric
        return np.mean(correlations) if correlations else 0.0
    
    def _generate_strategy(self, 
                          market_state: MarketState, 
                          predictions: Dict, 
                          risk_score: float) -> Strategy:
        """Generate investment strategy from predictions"""
        # Extract strategy logits from model predictions
        strategy_logits = predictions['strategy'].cpu().numpy()[0]
        
        # Get asset list
        assets = list(market_state.prices.keys())
        
        # Ensure we have enough logits for all assets
        if len(strategy_logits) < len(assets):
            # Pad with zeros if needed
            strategy_logits = np.pad(strategy_logits, (0, len(assets) - len(strategy_logits)))
        
        # Convert logits to allocations
        allocations = {}
        total_allocation = 0.0
        
        for i, asset in enumerate(assets):
            if i < len(strategy_logits):
                # Apply risk adjustment to allocation
                allocation = float(strategy_logits[i]) * (1 - risk_score)
                allocation = max(0.0, min(allocation, self.risk_manager['max_position_size']))
                allocations[asset] = allocation
                total_allocation += allocation
        
        # Normalize allocations
        if total_allocation > 0:
            for asset in allocations:
                allocations[asset] /= total_allocation
        
        # Generate specific actions
        actions = self._generate_actions(market_state, allocations, predictions)
        
        # Calculate expected returns
        expected_return = self._calculate_expected_return(allocations, predictions, market_state)
        
        # Calculate max drawdown estimate
        max_drawdown = self._estimate_max_drawdown(allocations, predictions, risk_score)
        
        # Generate reasoning
        reasoning = self._generate_reasoning(market_state, predictions, risk_score, allocations)
        
        # Extract confidence from model
        confidence = float(predictions['confidence'].item())
        
        return Strategy(
            timestamp=market_state.timestamp,
            allocations=allocations,
            actions=actions,
            confidence=confidence,
            risk_score=risk_score,
            expected_return=expected_return,
            max_drawdown=max_drawdown,
            reasoning=reasoning
        )
    
    def _generate_actions(self, 
                         market_state: MarketState, 
                         allocations: Dict[str, float],
                         predictions: Dict) -> List[Dict[str, Any]]:
        """Generate specific trading actions"""
        actions = []
        
        for asset, allocation in allocations.items():
            if allocation > 0.01:  # Only include significant allocations
                action = {
                    'action': 'allocate',
                    'asset': asset,
                    'allocation': allocation,
                    'predicted_price': float(predictions['price'].item()),
                    'confidence': float(predictions['confidence'].item()),
                    'suggested_entry': market_state.prices[asset] * 0.98,  # Slight discount
                    'stop_loss': market_state.prices[asset] * (1 - self.risk_manager['stop_loss']),
                    'take_profit': market_state.prices[asset] * (1 + self.risk_manager['take_profit'])
                }
                actions.append(action)
        
        # Add rebalancing suggestions
        actions.append({
            'action': 'rebalance',
            'frequency': f"every {self.config.rebalancing_frequency} hours",
            'trigger_conditions': [
                'allocation_drift > 10%',
                'volatility_spike > 20%',
                'confidence_drop > 15%'
            ]
        })
        
        return actions
    
    def _calculate_expected_return(self, 
                                  allocations: Dict[str, float],
                                  predictions: Dict,
                                  market_state: MarketState) -> float:
        """Calculate portfolio expected return"""
        total_return = 0.0
        
        # Get predicted price change
        price_prediction = float(predictions['price'].item())
        
        for asset, allocation in allocations.items():
            # Simple expected return calculation
            current_price = market_state.prices[asset]
            expected_price_change = (price_prediction - current_price) / current_price
            total_return += allocation * expected_price_change
        
        # Apply risk adjustment
        risk_adjustment = 1.0 - (predictions['risk'].item() * 0.5)
        return total_return * risk_adjustment
    
    def _estimate_max_drawdown(self, 
                              allocations: Dict[str, float],
                              predictions: Dict,
                              risk_score: float) -> float:
        """Estimate maximum potential drawdown"""
        # Base drawdown from volatility prediction
        base_drawdown = float(predictions['volatility'].item())
        
        # Adjust for concentration risk
        concentration_risk = max(allocations.values()) if allocations else 0
        concentration_penalty = concentration_risk * 2.0
        
        # Adjust for overall risk score
        risk_penalty = risk_score * 0.5
        
        # Combine factors
        max_drawdown = base_drawdown + concentration_penalty + risk_penalty
        return min(max_drawdown, 0.5)  # Cap at 50%
    
    def _generate_reasoning(self, 
                           market_state: MarketState,
                           predictions: Dict,
                           risk_score: float,
                           allocations: Dict[str, float]) -> str:
        """Generate human-readable reasoning for the strategy"""
        reasoning_parts = []
        
        # Market analysis
        avg_sentiment = np.mean(list(market_state.social_sentiment.values()))
        if avg_sentiment > 0.6:
            reasoning_parts.append("Strong positive sentiment detected across DeFi protocols.")
        elif avg_sentiment < 0.4:
            reasoning_parts.append("Bearish sentiment observed in social metrics.")
        
        # Volatility assessment
        vol_prediction = float(predictions['volatility'].item())
        if vol_prediction > 0.7:
            reasoning_parts.append("High volatility expected - recommended defensive positioning.")
        elif vol_prediction < 0.3:
            reasoning_parts.append("Low volatility environment - opportunity for higher allocations.")
        
        # Risk evaluation
        if risk_score > 0.7:
            reasoning_parts.append("Elevated risk levels detected - conservative strategy recommended.")
        elif risk_score < 0.3:
            reasoning_parts.append("Favorable risk-reward profile - aggressive positioning viable.")
        
        # Allocation insights
        top_allocation = max(allocations.items(), key=lambda x: x[1]) if allocations else ("", 0)
        if top_allocation[1] > 0.3:
            reasoning_parts.append(f"Overweight position in {top_allocation[0]} due to strong technical indicators.")
        
        # Confidence assessment
        confidence = float(predictions['confidence'].item())
        if confidence > 0.8:
            reasoning_parts.append("High confidence in predictions due to strong model consensus.")
        elif confidence < 0.5:
            reasoning_parts.append("Lower confidence suggests caution and closer monitoring required.")
        
        return " ".join(reasoning_parts) if reasoning_parts else "Strategy based on comprehensive multi-modal analysis."
    
    def _validate_strategy(self, strategy: Strategy) -> Strategy:
        """Validate and adjust strategy to ensure risk compliance"""
        # Check allocation constraints
        total_allocation = sum(strategy.allocations.values())
        if abs(total_allocation - 1.0) > 0.01:
            # Normalize allocations
            for asset in strategy.allocations:
                strategy.allocations[asset] /= total_allocation
        
        # Ensure no single position exceeds limit
        for asset, allocation in strategy.allocations.items():
            if allocation > self.risk_manager['max_position_size']:
                excess = allocation - self.risk_manager['max_position_size']
                strategy.allocations[asset] = self.risk_manager['max_position_size']
                
                # Redistribute excess to other assets
                other_assets = [a for a in strategy.allocations.keys() if a != asset]
                if other_assets:
                    excess_per_asset = excess / len(other_assets)
                    for other_asset in other_assets:
                        strategy.allocations[other_asset] += excess_per_asset
        
        # Apply confidence threshold
        if strategy.confidence < self.config.confidence_threshold:
            # Reduce allocations proportionally
            reduction_factor = strategy.confidence / self.config.confidence_threshold
            for asset in strategy.allocations:
                strategy.allocations[asset] *= reduction_factor
        
        # Final validation
        strategy = self._apply_risk_filters(strategy)
        
        return strategy
    
    def _apply_risk_filters(self, strategy: Strategy) -> Strategy:
        """Apply additional risk filters and constraints"""
        # Check correlation risk
        assets = list(strategy.allocations.keys())
        if len(assets) > 1:
            # Simplified correlation check - reduce allocations if too concentrated
            max_individual = max(strategy.allocations.values())
            if max_individual > 0.5:  # More than 50% in single asset
                # Diversify by moving some allocation to cash/stable assets
                reduction = (max_individual - 0.5) * 0.5
                max_asset = max(strategy.allocations.items(), key=lambda x: x[1])[0]
                strategy.allocations[max_asset] -= reduction
                
                # Add to stablecoin allocation (if not present, create it)
                if 'USDC' not in strategy.allocations:
                    strategy.allocations['USDC'] = reduction
                else:
                    strategy.allocations['USDC'] += reduction
        
        # Ensure minimum diversification
        if len([a for a, v in strategy.allocations.items() if v > 0.01]) < 3:
            strategy.reasoning += " Note: Limited diversification due to strong conviction signals."
        
        return strategy
    
    def _save_model(self, save_path: str):
        """Save the trained model and associated components"""
        model_data = {
            'model_state_dict': self.model.state_dict(),
            'config': asdict(self.config),
            'ensemble_weights': self.ensemble.weights if hasattr(self.ensemble, 'weights') else [],
            'risk_manager': self.risk_manager,
            'performance_metrics': self.performance_metrics
        }
        
        # Create directory if it doesn't exist
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Save model
        torch.save(model_data, save_path)
        
        # Save tokenizer
        tokenizer_path = str(Path(save_path).parent / "tokenizer")
        self.tokenizer.save_pretrained(tokenizer_path)
        
        # Save scalers and other preprocessing components
        preprocessing_path = str(Path(save_path).parent / "preprocessing.pkl")
        preprocessing_data = {
            'anomaly_detector': self.anomaly_detector,
            # Add other preprocessing components as needed
        }
        joblib.dump(preprocessing_data, preprocessing_path)
        
        logger.info(f"Model saved to {save_path}")
    
    def load_model(self, model_path: str):
        """Load a trained model from disk"""
        # Load model data
        model_data = torch.load(model_path, map_location=self.device)
        
        # Restore config
        self.config = ModelConfig(**model_data['config'])
        
        # Recreate and load model
        self.model = MultiModalTransformer(self.config)
        self.model.load_state_dict(model_data['model_state_dict'])
        self.model.to(self.device)
        
        # Restore ensemble weights
        if 'ensemble_weights' in model_data:
            self.ensemble.weights = model_data['ensemble_weights']
        
        # Restore other components
        self.risk_manager = model_data.get('risk_manager', self._initialize_risk_manager())
        self.performance_metrics = model_data.get('performance_metrics', {})
        
        # Load tokenizer
        tokenizer_path = str(Path(model_path).parent / "tokenizer")
        if Path(tokenizer_path).exists():
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        
        # Load preprocessing components
        preprocessing_path = str(Path(model_path).parent / "preprocessing.pkl")
        if Path(preprocessing_path).exists():
            preprocessing_data = joblib.load(preprocessing_path)
            self.anomaly_detector = preprocessing_data.get('anomaly_detector', IsolationForest())
        
        logger.info(f"Model loaded from {model_path}")
    
    async def backtest(self, 
                      historical_data: pd.DataFrame,
                      initial_capital: float = 100000.0,
                      start_date: Optional[datetime] = None,
                      end_date: Optional[datetime] = None) -> Dict[str, Any]:
        """
        Backtest the model on historical data
        
        Returns:
            Comprehensive backtest results with performance metrics
        """
        logger.info("Starting backtest...")
        
        # Filter data by date range if specified
        if start_date:
            historical_data = historical_data[historical_data.index >= start_date]
        if end_date:
            historical_data = historical_data[historical_data.index <= end_date]
        
        # Initialize backtest state
        capital = initial_capital
        positions = {}
        portfolio_value = [initial_capital]
        timestamps = []
        trades = []
        
        # Backtest loop
        for i in range(self.config.sequence_length, len(historical_data)):
            current_time = historical_data.index[i]
            timestamps.append(current_time)
            
            # Create market state from historical data
            market_state = self._create_market_state_from_history(historical_data, i)
            
            # Generate strategy
            strategy = await self.predict(market_state)
            
            # Execute trades
            capital, positions, new_trades = self._execute_backtest_trades(
                strategy, market_state, capital, positions
            )
            trades.extend(new_trades)
            
            # Calculate portfolio value
            portfolio_value.append(capital + sum(
                positions.get(asset, 0) * market_state.prices[asset] 
                for asset in market_state.prices
            ))
        
        # Calculate performance metrics
        results = self._calculate_backtest_metrics(
            portfolio_value, timestamps, trades, initial_capital
        )
        
        logger.info(f"Backtest complete. Total return: {results['total_return']:.2%}")
        return results
    
    def _create_market_state_from_history(self, data: pd.DataFrame, index: int) -> MarketState:
        """Create MarketState from historical data point"""
        current_row = data.iloc[index]
        
        # Extract basic market data
        prices = {}
        volumes = {}
        liquidity = {}
        volatility = {}
        
        # Parse columns to extract asset information
        for col in data.columns:
            if col.endswith('_price'):
                asset = col.replace('_price', '')
                prices[asset] = current_row[col]
            elif col.endswith('_volume'):
                asset = col.replace('_volume', '')
                volumes[asset] = current_row[col]
            elif col.endswith('_liquidity'):
                asset = col.replace('_liquidity', '')
                liquidity[asset] = current_row[col]
        
        # Calculate rolling volatility
        window_size = min(24, index)  # 24-hour window or available data
        for asset in prices.keys():
            price_col = f"{asset}_price"
            if price_col in data.columns:
                price_series = data[price_col].iloc[index-window_size:index]
                volatility[asset] = price_series.pct_change().std()
        
        # Mock social sentiment (in practice, this would come from social data)
        social_sentiment = {asset: 0.5 for asset in prices.keys()}  # Neutral sentiment
        
        # Technical indicators (simplified)
        technical_indicators = {}
        for asset in prices.keys():
            technical_indicators[asset] = {
                'rsi': 50.0,  # Placeholder
                'macd': 0.0,  # Placeholder
                'sma_20': prices[asset],  # Simplified
                'sma_50': prices[asset]   # Simplified
            }
        
        # Protocol metrics (dummy data)
        protocol_metrics = {asset: {'tvl': 1000000.0} for asset in prices.keys()}
        
        # Risk metrics
        risk_metrics = {
            'overall_volatility': np.mean(list(volatility.values())),
            'correlation_risk': 0.5,  # Placeholder
            'liquidity_risk': 1.0 - np.mean(list(liquidity.values()))
        }
        
        return MarketState(
            timestamp=current_row.name,
            prices=prices,
            volumes=volumes,
            liquidity=liquidity,
            volatility=volatility,
            social_sentiment=social_sentiment,
            technical_indicators=technical_indicators,
            protocol_metrics=protocol_metrics,
            risk_metrics=risk_metrics
        )
    
    def _execute_backtest_trades(self, 
                                strategy: Strategy,
                                market_state: MarketState,
                                capital: float,
                                positions: Dict[str, float]) -> Tuple[float, Dict[str, float], List[Dict]]:
        """Execute trades during backtest"""
        trades = []
        new_positions = positions.copy()
        remaining_capital = capital
        
        # Calculate current portfolio value
        current_portfolio_value = remaining_capital + sum(
            positions.get(asset, 0) * market_state.prices[asset] 
            for asset in market_state.prices
        )
        
        # Execute rebalancing
        for asset, target_allocation in strategy.allocations.items():
            if asset in market_state.prices:
                target_value = current_portfolio_value * target_allocation
                current_position = positions.get(asset, 0)
                current_value = current_position * market_state.prices[asset]
                
                # Calculate required trade
                trade_value = target_value - current_value
                trade_quantity = trade_value / market_state.prices[asset]
                
                # Execute trade (with transaction costs)
                transaction_cost = abs(trade_value) * 0.001  # 0.1% transaction cost
                
                if abs(trade_quantity) > 0.001:  # Only trade if significant
                    new_positions[asset] = current_position + trade_quantity
                    remaining_capital -= trade_value + transaction_cost
                    
                    trades.append({
                        'timestamp': market_state.timestamp,
                        'asset': asset,
                        'quantity': trade_quantity,
                        'price': market_state.prices[asset],
                        'value': trade_value,
                        'transaction_cost': transaction_cost,
                        'type': 'rebalance'
                    })
        
        return remaining_capital, new_positions, trades
    
    def _calculate_backtest_metrics(self, 
                                   portfolio_values: List[float],
                                   timestamps: List[datetime],
                                   trades: List[Dict],
                                   initial_capital: float) -> Dict[str, Any]:
        """Calculate comprehensive backtest performance metrics"""
        # Convert to numpy arrays for easier calculation
        values = np.array(portfolio_values)
        timestamps_array = np.array(timestamps)
        
        # Returns
        returns = np.diff(values) / values[:-1]
        total_return = (values[-1] - initial_capital) / initial_capital
        
        # Annualized metrics
        trading_days = len(returns)
        days_per_year = 365
        annualized_return = (1 + total_return) ** (days_per_year / trading_days) - 1
        
        # Risk metrics
        volatility = np.std(returns) * np.sqrt(days_per_year)
        downside_returns = returns[returns < 0]
        downside_volatility = np.std(downside_returns) * np.sqrt(days_per_year) if len(downside_returns) > 0 else 0
        
        # Sharpe ratio (assuming 2% risk-free rate)
        sharpe_ratio = (annualized_return - 0.02) / volatility if volatility > 0 else 0
        
        # Sortino ratio
        sortino_ratio = (annualized_return - 0.02) / downside_volatility if downside_volatility > 0 else 0
        
        # Maximum drawdown
        running_max = np.maximum.accumulate(values)
        drawdown = (values - running_max) / running_max
        max_drawdown = np.min(drawdown)
        
        # Win rate
        winning_trades = sum(1 for trade in trades if trade['value'] > 0)
        total_trades = len(trades)
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        # Transaction costs
        total_transaction_costs = sum(trade['transaction_cost'] for trade in trades)
        
        # Calmar ratio
        calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0
        
        return {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'total_transaction_costs': total_transaction_costs,
            'calmar_ratio': calmar_ratio,
            'final_portfolio_value': values[-1],
            'portfolio_evolution': values.tolist(),
            'timestamps': [t.isoformat() for t in timestamps],
            'trades': trades
        }
    
    async def optimize_hyperparameters(self, 
                                      train_data: pd.DataFrame,
                                      val_data: pd.DataFrame,
                                      n_trials: int = 100) -> Dict[str, Any]:
        """
        Optimize model hyperparameters using Optuna
        
        Returns:
            Best hyperparameters and performance metrics
        """
        logger.info(f"Starting hyperparameter optimization with {n_trials} trials...")
        
        def objective(trial):
            # Suggest hyperparameters
            config_updates = {
                'hidden_dim': trial.suggest_categorical('hidden_dim', [256, 512, 768, 1024]),
                'num_layers': trial.suggest_int('num_layers', 3, 12),
                'num_heads': trial.suggest_int('num_heads', 4, 16),
                'dropout': trial.suggest_float('dropout', 0.05, 0.3),
                'learning_rate': trial.suggest_float('learning_rate', 1e-6, 1e-3, log=True),
                'batch_size': trial.suggest_categorical('batch_size', [16, 32, 64, 128]),
                'sequence_length': trial.suggest_int('sequence_length', 48, 288),
                'prediction_horizon': trial.suggest_int('prediction_horizon', 6, 48)
            }
            
            # Create modified config
            trial_config = ModelConfig(**{**asdict(self.config), **config_updates})
            
            # Create and train trial model
            trial_model = MultiModalTransformer(trial_config)
            trial_model.to(self.device)
            
            # Quick training
            trial_dataset = DeucalionDataset(train_data, trial_config, self.tokenizer)
            trial_loader = DataLoader(trial_dataset, batch_size=trial_config.batch_size, shuffle=True)
            
            optimizer = torch.optim.AdamW(
                trial_model.parameters(),
                lr=trial_config.learning_rate,
                weight_decay=trial_config.weight_decay
            )
            
            # Train for limited epochs
            for epoch in range(5):  # Limited epochs for hyperparameter search
                trial_model.train()
                for batch in trial_loader:
                    optimizer.zero_grad()
                    numerical_data = batch['numerical_data'].to(self.device)
                    predictions = trial_model(numerical_data)
                    loss = self._calculate_loss(predictions, batch['targets'])
                    loss.backward()
                    optimizer.step()
            
            # Evaluate on validation set
            val_dataset = DeucalionDataset(val_data, trial_config, self.tokenizer, is_training=False)
            val_loader = DataLoader(val_dataset, batch_size=trial_config.batch_size, shuffle=False)
            
            total_loss = 0
            num_batches = 0
            
            trial_model.eval()
            with torch.no_grad():
                for batch in val_loader:
                    numerical_data = batch['numerical_data'].to(self.device)
                    predictions = trial_model(numerical_data)
                    loss = self._calculate_loss(predictions, batch['targets'])
                    total_loss += loss.item()
                    num_batches += 1
            
            return total_loss / num_batches
        
        # Run optimization
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=n_trials)
        
        # Get best parameters
        best_params = study.best_params
        best_value = study.best_value
        
        logger.info(f"Hyperparameter optimization complete. Best validation loss: {best_value:.4f}")
        
        return {
            'best_params': best_params,
            'best_value': best_value,
            'study': study
        }
    
    async def explain_prediction(self, 
                                market_state: MarketState,
                                strategy: Optional[Strategy] = None) -> Dict[str, Any]:
        """
        Generate explainable AI insights for predictions
        
        Returns:
            Dictionary with detailed explanations and feature importance
        """
        if strategy is None:
            strategy = await self.predict(market_state)
        
        # Feature importance analysis
        feature_importance = await self._analyze_feature_importance(market_state)
        
        # Attention visualization
        attention_weights = await self._extract_attention_weights(market_state)
        
        # Sensitivity analysis
        sensitivity_analysis = await self._perform_sensitivity_analysis(market_state)
        
        # Risk factor breakdown
        risk_breakdown = self._explain_risk_factors(market_state, strategy)
        
        # Confidence sources
        confidence_factors = self._analyze_confidence_sources(market_state, strategy)
        
        return {
            'strategy': asdict(strategy),
            'feature_importance': feature_importance,
            'attention_weights': attention_weights,
            'sensitivity_analysis': sensitivity_analysis,
            'risk_breakdown': risk_breakdown,
            'confidence_factors': confidence_factors,
            'market_regime': self._identify_market_regime(market_state),
            'key_drivers': self._identify_key_drivers(market_state, strategy)
        }
    
    async def _analyze_feature_importance(self, market_state: MarketState) -> Dict[str, float]:
        """Analyze feature importance using gradient-based methods"""
        # Prepare input
        input_data = self._prepare_market_state(market_state)
        numerical_data = input_data['numerical_data']
        numerical_data.requires_grad_(True)
        
        # Forward pass
        predictions = self.model(numerical_data)
        
        # Calculate gradients
        total_loss = sum(predictions[key].sum() for key in predictions if key != 'strategy')
        total_loss.backward()
        
        # Get gradients and calculate importance
        gradients = numerical_data.grad.abs().cpu().numpy()[0, 0]
        
        # Create feature importance dictionary
        feature_names = [
            'prices', 'volumes', 'technical_indicators', 
            'social_sentiment', 'risk_metrics'
        ]
        
        # Distribute gradients across feature groups
        features_per_group = len(gradients) // len(feature_names)
        importance = {}
        
        for i, feature_name in enumerate(feature_names):
            start_idx = i * features_per_group
            end_idx = (i + 1) * features_per_group
            importance[feature_name] = float(np.mean(gradients[start_idx:end_idx]))
        
        # Normalize importance scores
        total_importance = sum(importance.values())
        if total_importance > 0:
            importance = {k: v / total_importance for k, v in importance.items()}
        
        return importance
    
    async def _extract_attention_weights(self, market_state: MarketState) -> Dict[str, List[float]]:
        """Extract attention weights from transformer layers"""
        # This is a simplified implementation
        # In practice, you'd need to modify the model to return attention weights
        
        # Mock attention weights for demonstration
        attention_weights = {
            'layer_1': [0.3, 0.2, 0.2, 0.2, 0.1],
            'layer_2': [0.25, 0.25, 0.2, 0.2, 0.1],
            'layer_3': [0.2, 0.3, 0.2, 0.2, 0.1]
        }
        
        return attention_weights
    
    async def _perform_sensitivity_analysis(self, market_state: MarketState) -> Dict[str, float]:
        """Perform sensitivity analysis by perturbing inputs"""
        baseline_strategy = await self.predict(market_state)
        sensitivities = {}
        
        # Test sensitivity to volume changes
        for asset in modified_state.volumes:
            # Increase volume by 10%
            original_volume = modified_state.volumes[asset]
            modified_state.volumes[asset] *= 1.1
            
            new_strategy = await self.predict(modified_state)
            allocation_change = abs(new_strategy.allocations.get(asset, 0) - 
                                  baseline_strategy.allocations.get(asset, 0))
            sensitivities[f"{asset}_volume"] = allocation_change
            
            # Restore original volume
            modified_state.volumes[asset] = original_volume
        
        # Test sensitivity to social sentiment
        for asset in modified_state.social_sentiment:
            # Increase sentiment by 0.1
            original_sentiment = modified_state.social_sentiment[asset]
            modified_state.social_sentiment[asset] = min(1.0, original_sentiment + 0.1)
            
            new_strategy = await self.predict(modified_state)
            allocation_change = abs(new_strategy.allocations.get(asset, 0) - 
                                  baseline_strategy.allocations.get(asset, 0))
            sensitivities[f"{asset}_sentiment"] = allocation_change
            
            # Restore original sentiment
            modified_state.social_sentiment[asset] = original_sentiment
        
        return sensitivities
    
    def _explain_risk_factors(self, market_state: MarketState, strategy: Strategy) -> Dict[str, Any]:
        """Break down risk factors contributing to the overall risk score"""
        risk_breakdown = {}
        
        # Volatility risk
        volatilities = list(market_state.volatility.values())
        avg_volatility = np.mean(volatilities) if volatilities else 0
        risk_breakdown['volatility_risk'] = {
            'value': avg_volatility,
            'contribution': avg_volatility * 0.3,  # 30% weight in risk calculation
            'explanation': f"Average volatility of {avg_volatility:.1%} across assets"
        }
        
        # Concentration risk
        allocations = list(strategy.allocations.values())
        max_allocation = max(allocations) if allocations else 0
        concentration_risk = max_allocation - 0.2  # Risk if >20% in single asset
        risk_breakdown['concentration_risk'] = {
            'value': max(0, concentration_risk),
            'contribution': max(0, concentration_risk) * 0.25,
            'explanation': f"Maximum allocation: {max_allocation:.1%} to single asset"
        }
        
        # Liquidity risk
        liquidity_values = list(market_state.liquidity.values())
        avg_liquidity = np.mean(liquidity_values) if liquidity_values else 1.0
        liquidity_risk = 1.0 - avg_liquidity
        risk_breakdown['liquidity_risk'] = {
            'value': liquidity_risk,
            'contribution': liquidity_risk * 0.2,
            'explanation': f"Average liquidity: {avg_liquidity:.1%}"
        }
        
        # Market conditions risk
        sentiment_values = list(market_state.social_sentiment.values())
        avg_sentiment = np.mean(sentiment_values) if sentiment_values else 0.5
        sentiment_risk = 0.5 - abs(avg_sentiment - 0.5)  # Risk highest at extreme sentiment
        risk_breakdown['sentiment_risk'] = {
            'value': sentiment_risk,
            'contribution': sentiment_risk * 0.15,
            'explanation': f"Market sentiment: {avg_sentiment:.1%}"
        }
        
        # Correlation risk
        correlation_risk = self._calculate_correlation_risk(market_state)
        risk_breakdown['correlation_risk'] = {
            'value': correlation_risk,
            'contribution': correlation_risk * 0.1,
            'explanation': f"Asset correlation risk: {correlation_risk:.1%}"
        }
        
        return risk_breakdown
    
    def _analyze_confidence_sources(self, market_state: MarketState, strategy: Strategy) -> Dict[str, Any]:
        """Analyze sources of prediction confidence"""
        confidence_factors = {}
        
        # Model consensus
        # This would require storing individual model predictions
        confidence_factors['model_consensus'] = {
            'value': 0.85,  # Placeholder
            'explanation': "High agreement between ensemble models"
        }
        
        # Data quality
        # Check for missing or anomalous data
        data_quality_score = 1.0
        for asset in market_state.prices:
            if market_state.prices[asset] <= 0:
                data_quality_score -= 0.1
        confidence_factors['data_quality'] = {
            'value': max(0, min(1, data_quality_score)),
            'explanation': f"Data quality score: {data_quality_score:.1%}"
        }
        
        # Historical pattern recognition
        # This would analyze how similar current conditions are to historical patterns
        confidence_factors['pattern_recognition'] = {
            'value': 0.7,  # Placeholder
            'explanation': "Current market conditions match historical patterns"
        }
        
        # Feature stability
        # Check if key features are stable or rapidly changing
        feature_stability = 0.8  # Placeholder calculation
        confidence_factors['feature_stability'] = {
            'value': feature_stability,
            'explanation': f"Market features show {feature_stability:.1%} stability"
        }
        
        return confidence_factors
    
    def _identify_market_regime(self, market_state: MarketState) -> Dict[str, Any]:
        """Identify current market regime"""
        # Calculate regime indicators
        avg_volatility = np.mean(list(market_state.volatility.values())) if market_state.volatility else 0
        avg_sentiment = np.mean(list(market_state.social_sentiment.values())) if market_state.social_sentiment else 0.5
        avg_volume = np.mean(list(market_state.volumes.values())) if market_state.volumes else 0
        
        # Classify regime
        if avg_volatility > 0.05 and avg_sentiment < 0.3:
            regime = "BEAR_MARKET"
            description = "High volatility with negative sentiment indicating bear market conditions"
        elif avg_volatility < 0.02 and avg_sentiment > 0.7:
            regime = "BULL_MARKET"
            description = "Low volatility with positive sentiment indicating bull market conditions"
        elif avg_volatility > 0.04:
            regime = "HIGH_VOLATILITY"
            description = "Elevated volatility suggests uncertain market conditions"
        elif avg_volume < np.percentile(list(market_state.volumes.values()), 25):
            regime = "LOW_LIQUIDITY"
            description = "Below-average trading volumes indicating low liquidity"
        else:
            regime = "RANGING_MARKET"
            description = "Market showing sideways movement with moderate volatility"
        
        return {
            'regime': regime,
            'description': description,
            'confidence': 0.75,  # Placeholder confidence score
            'indicators': {
                'volatility': avg_volatility,
                'sentiment': avg_sentiment,
                'volume': avg_volume
            }
        }
    
    def _identify_key_drivers(self, market_state: MarketState, strategy: Strategy) -> List[Dict[str, Any]]:
        """Identify key market drivers influencing the strategy"""
        drivers = []
        
        # Price momentum driver
        price_changes = []
        for asset in market_state.prices:
            # Calculate recent price momentum (simplified)
            current_price = market_state.prices[asset]
            # This would normally use historical data
            momentum = 0.02  # Placeholder 2% momentum
            price_changes.append((asset, momentum))
        
        # Sort by momentum
        price_changes.sort(key=lambda x: abs(x[1]), reverse=True)
        
        if price_changes:
            top_mover = price_changes[0]
            drivers.append({
                'type': 'price_momentum',
                'asset': top_mover[0],
                'value': top_mover[1],
                'impact': 'high',
                'description': f"{top_mover[0]} showing {top_mover[1]:.1%} momentum"
            })
        
        # Volume driver
        if market_state.volumes:
            volume_ratios = {asset: vol / np.mean(list(market_state.volumes.values())) 
                           for asset, vol in market_state.volumes.items()}
            top_volume = max(volume_ratios.items(), key=lambda x: x[1])
            
            if top_volume[1] > 1.5:  # 50% above average
                drivers.append({
                    'type': 'volume_surge',
                    'asset': top_volume[0],
                    'value': top_volume[1],
                    'impact': 'medium',
                    'description': f"{top_volume[0]} volume {top_volume[1]:.1f}x above average"
                })
        
        # Sentiment driver
        if market_state.social_sentiment:
            extreme_sentiment = {asset: abs(sent - 0.5) * 2 
                               for asset, sent in market_state.social_sentiment.items()}
            top_sentiment = max(extreme_sentiment.items(), key=lambda x: x[1])
            
            if top_sentiment[1] > 0.5:  # Extreme sentiment
                original_sentiment = market_state.social_sentiment[top_sentiment[0]]
                sentiment_direction = "positive" if original_sentiment > 0.5 else "negative"
                drivers.append({
                    'type': 'social_sentiment',
                    'asset': top_sentiment[0],
                    'value': original_sentiment,
                    'impact': 'medium',
                    'description': f"{top_sentiment[0]} showing {sentiment_direction} sentiment ({original_sentiment:.1%})"
                })
        
        # Allocation driver
        if strategy.allocations:
            top_allocation = max(strategy.allocations.items(), key=lambda x: x[1])
            if top_allocation[1] > 0.3:  # Significant allocation
                drivers.append({
                    'type': 'strategic_allocation',
                    'asset': top_allocation[0],
                    'value': top_allocation[1],
                    'impact': 'high',
                    'description': f"High conviction {top_allocation[1]:.1%} allocation to {top_allocation[0]}"
                })
        
        return drivers
    
    async def continuous_monitoring(self, 
                                  monitoring_config: Dict[str, Any],
                                  callback: Optional[callable] = None) -> None:
        """
        Continuously monitor market conditions and trigger rebalancing
        
        Args:
            monitoring_config: Configuration for monitoring parameters
            callback: Optional callback function for notifications
        """
        logger.info("Starting continuous monitoring...")
        
        monitoring_state = {
            'last_rebalance': datetime.now(),
            'alerts_triggered': [],
            'performance_tracking': []
        }
        
        while True:
            try:
                # Get current market state (this would connect to real data sources)
                current_market_state = await self._fetch_current_market_state()
                
                # Generate current strategy
                current_strategy = await self.predict(current_market_state)
                
                # Check for rebalancing triggers
                should_rebalance = self._check_rebalancing_triggers(
                    current_strategy, 
                    current_market_state,
                    monitoring_state,
                    monitoring_config
                )
                
                if should_rebalance:
                    logger.info("Rebalancing triggered")
                    
                    # Execute rebalancing (this would connect to trading systems)
                    rebalance_result = await self._execute_rebalancing(current_strategy)
                    
                    # Update monitoring state
                    monitoring_state['last_rebalance'] = datetime.now()
                    monitoring_state['performance_tracking'].append({
                        'timestamp': datetime.now(),
                        'strategy': current_strategy,
                        'rebalance_result': rebalance_result
                    })
                    
                    # Send notification if callback provided
                    if callback:
                        await callback({
                            'type': 'rebalance_executed',
                            'timestamp': datetime.now(),
                            'strategy': current_strategy,
                            'result': rebalance_result
                        })
                
                # Check for alerts
                alerts = self._check_alerts(current_market_state, current_strategy, monitoring_config)
                
                for alert in alerts:
                    if alert not in monitoring_state['alerts_triggered']:
                        logger.warning(f"Alert triggered: {alert['message']}")
                        monitoring_state['alerts_triggered'].append(alert)
                        
                        if callback:
                            await callback({
                                'type': 'alert',
                                'alert': alert,
                                'timestamp': datetime.now()
                            })
                
                # Wait before next check
                await asyncio.sleep(monitoring_config.get('check_interval', 60))  # Default 1 minute
                
            except Exception as e:
                logger.error(f"Error in continuous monitoring: {e}")
                if callback:
                    await callback({
                        'type': 'error',
                        'error': str(e),
                        'timestamp': datetime.now()
                    })
                await asyncio.sleep(60)  # Wait before retrying
    
    async def _fetch_current_market_state(self) -> MarketState:
        """Fetch current market state from data sources"""
        # This would connect to real data sources (APIs, blockchain, etc.)
        # For now, return a placeholder
        
        return MarketState(
            timestamp=datetime.now(),
            prices={'ETH': 2000.0, 'BTC': 45000.0, 'USDC': 1.0},
            volumes={'ETH': 1000000.0, 'BTC': 500000.0, 'USDC': 2000000.0},
            liquidity={'ETH': 0.9, 'BTC': 0.85, 'USDC': 0.99},
            volatility={'ETH': 0.03, 'BTC': 0.04, 'USDC': 0.001},
            social_sentiment={'ETH': 0.6, 'BTC': 0.7, 'USDC': 0.5},
            technical_indicators={
                'ETH': {'rsi': 55, 'macd': 0.02},
                'BTC': {'rsi': 60, 'macd': 0.05},
                'USDC': {'rsi': 50, 'macd': 0.0}
            },
            protocol_metrics={
                'ETH': {'tvl': 10000000.0},
                'BTC': {'tvl': 15000000.0},
                'USDC': {'tvl': 20000000.0}
            },
            risk_metrics={
                'overall_volatility': 0.025,
                'correlation_risk': 0.4,
                'liquidity_risk': 0.1
            }
        )
    
    def _check_rebalancing_triggers(self, 
                                   strategy: Strategy,
                                   market_state: MarketState,
                                   monitoring_state: Dict,
                                   config: Dict) -> bool:
        """Check if rebalancing should be triggered"""
        # Time-based trigger
        time_since_rebalance = datetime.now() - monitoring_state['last_rebalance']
        if time_since_rebalance.seconds / 3600 >= config.get('max_hours_between_rebalance', 24):
            return True
        
        # Volatility trigger
        current_volatility = np.mean(list(market_state.volatility.values()))
        if current_volatility > config.get('volatility_threshold', 0.05):
            return True
        
        # Allocation drift trigger
        if monitoring_state['performance_tracking']:
            last_strategy = monitoring_state['performance_tracking'][-1]['strategy']
            allocation_drift = self._calculate_allocation_drift(strategy, last_strategy)
            if allocation_drift > config.get('max_allocation_drift', 0.1):
                return True
        
        # Confidence drop trigger
        if strategy.confidence < config.get('min_confidence', 0.6):
            return True
        
        return False
    
    def _calculate_allocation_drift(self, current: Strategy, previous: Strategy) -> float:
        """Calculate the maximum allocation drift between strategies"""
        max_drift = 0.0
        
        all_assets = set(current.allocations.keys()) | set(previous.allocations.keys())
        
        for asset in all_assets:
            current_alloc = current.allocations.get(asset, 0.0)
            previous_alloc = previous.allocations.get(asset, 0.0)
            drift = abs(current_alloc - previous_alloc)
            max_drift = max(max_drift, drift)
        
        return max_drift
    
    async def _execute_rebalancing(self, strategy: Strategy) -> Dict[str, Any]:
        """Execute the rebalancing according to strategy"""
        # This would connect to actual trading systems
        # For now, return a simulated result
        
        logger.info("Executing rebalancing...")
        
        # Simulate execution
        execution_results = {
            'success': True,
            'trades_executed': [],
            'total_cost': 0.0,
            'execution_time': datetime.now()
        }
        
        for asset, allocation in strategy.allocations.items():
            trade = {
                'asset': asset,
                'target_allocation': allocation,
                'executed': True,
                'price': 100.0,  # Placeholder
                'cost': allocation * 100.0 * 0.001  # 0.1% trading cost
            }
            execution_results['trades_executed'].append(trade)
            execution_results['total_cost'] += trade['cost']
        
        return execution_results
    
    def _check_alerts(self, 
                     market_state: MarketState,
                     strategy: Strategy,
                     config: Dict) -> List[Dict[str, Any]]:
        """Check for various alert conditions"""
        alerts = []
        
        # High volatility alert
        avg_volatility = np.mean(list(market_state.volatility.values()))
        if avg_volatility > config.get('high_volatility_threshold', 0.1):
            alerts.append({
                'type': 'high_volatility',
                'severity': 'warning',
                'message': f"High volatility detected: {avg_volatility:.1%}",
                'value': avg_volatility,
                'timestamp': datetime.now()
            })
        
        # Low confidence alert
        if strategy.confidence < config.get('low_confidence_threshold', 0.5):
            alerts.append({
                'type': 'low_confidence',
                'severity': 'warning',
                'message': f"Low prediction confidence: {strategy.confidence:.1%}",
                'value': strategy.confidence,
                'timestamp': datetime.now()
            })
        
        # High risk alert
        if strategy.risk_score > config.get('high_risk_threshold', 0.8):
            alerts.append({
                'type': 'high_risk',
                'severity': 'danger',
                'message': f"High risk score detected: {strategy.risk_score:.1%}",
                'value': strategy.risk_score,
                'timestamp': datetime.now()
            })
        
        # Concentration alert
        max_allocation = max(strategy.allocations.values()) if strategy.allocations else 0
        if max_allocation > config.get('concentration_threshold', 0.5):
            max_asset = max(strategy.allocations.items(), key=lambda x: x[1])[0]
            alerts.append({
                'type': 'high_concentration',
                'severity': 'warning',
                'message': f"High concentration in {max_asset}: {max_allocation:.1%}",
                'value': max_allocation,
                'asset': max_asset,
                'timestamp': datetime.now()
            })
        
        return alerts
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get comprehensive information about the model"""
        return {
            'model_type': self.config.model_type,
            'architecture': {
                'hidden_dim': self.config.hidden_dim,
                'num_layers': self.config.num_layers,
                'num_heads': self.config.num_heads,
                'dropout': self.config.dropout
            },
            'training_config': {
                'learning_rate': self.config.learning_rate,
                'batch_size': self.config.batch_size,
                'sequence_length': self.config.sequence_length,
                'prediction_horizon': self.config.prediction_horizon
            },
            'ensemble_info': {
                'num_models': len(self.ensemble.models),
                'weights': self.ensemble.weights
            },
            'risk_management': self.risk_manager,
            'performance_metrics': self.performance_metrics,
            'device': str(self.device),
            'training_status': hasattr(self, 'optimizer') and self.optimizer is not None
        }


# Utility functions for model management
async def create_model_ensemble(configs: List[ModelConfig]) -> ModelEnsemble:
    """Create an ensemble of models with different configurations"""
    ensemble = ModelEnsemble(configs[0])
    
    for config in configs:
        model = MultiModalTransformer(config)
        ensemble.add_model(model)
    
    return ensemble


def load_pretrained_model(model_path: str) -> DeucalionModel:
    """Load a pretrained Deucalion model"""
    model = DeucalionModel()
    model.load_model(model_path)
    return model


# Export main classes and functions
__all__ = [
    'DeucalionModel',
    'MultiModalTransformer',
    'ModelConfig',
    'MarketState',
    'Strategy',
    'DeucalionDataset',
    'ModelEnsemble',
    'create_model_ensemble',
    'load_pretrained_model'
] sensitivity to price changes
        modified_state = MarketState(**asdict(market_state))
        for asset in modified_state.prices:
            # Increase price by 1%
            original_price = modified_state.prices[asset]
            modified_state.prices[asset] *= 1.01
            
            new_strategy = await self.predict(modified_state)
            allocation_change = abs(new_strategy.allocations.get(asset, 0) - 
                                  baseline_strategy.allocations.get(asset, 0))
            sensitivities[f"{asset}_price"] = allocation_change
            
            # Restore original price
            modified_state.prices[asset] = original_price
        
        # Test