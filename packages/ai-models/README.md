# Minos-AI: Machine Learning Models for DeFi on Solana

## Overview
This package contains the machine learning infrastructure for the Minos-AI DeFi strategy platform. Our models analyze market data, on-chain metrics, and external signals to generate optimization strategies for Solana-based investments.

## Architecture

```
packages/ai-models/
├── models/                   # Model definitions and implementations
│   ├── price_prediction.py   # Price movement forecasting models
│   ├── volatility.py         # Volatility estimation models
│   ├── sentiment.py          # Market sentiment analysis models
│   └── risk.py               # Risk assessment models
├── training/                 # Training pipelines and configurations
│   ├── train.py              # Main training orchestration
│   ├── dataset.py            # Dataset preparation and augmentation
│   ├── config.py             # Training hyperparameters
│   └── callbacks.py          # Custom training callbacks
├── utils/                    # Utility functions and helpers
│   ├── data_loader.py        # Data ingestion from various sources
│   ├── metrics.py            # Custom performance metrics
│   ├── preprocessing.py      # Feature engineering and normalization
│   └── visualization.py      # Result visualization tools
├── inference/                # Model serving and inference
│   ├── predictor.py          # Prediction service
│   ├── ensemble.py           # Model ensemble techniques
│   └── streaming.py          # Real-time prediction pipeline
├── experiments/              # Experiment tracking and versioning
│   ├── experiment.py         # Experiment configuration
│   └── tracking.py           # MLflow integration
└── deployment/               # Model deployment utilities
    ├── packaging.py          # Model packaging for deployment
    ├── monitoring.py         # Production monitoring
    └── versioning.py         # Model versioning and rollback
```

## Key Components

### Models
- **Price Prediction**: LSTM-based models for short/medium-term price movement forecasting
- **Volatility Estimation**: GARCH and ML-based models for volatility prediction
- **Sentiment Analysis**: NLP models for market sentiment extraction from social/news sources
- **Risk Assessment**: Bayesian models for risk quantification and portfolio optimization

### Data Pipeline
1. **Data Collection**: Integration with Solana blockchain, CEXs/DEXs, and external data providers
2. **Feature Engineering**: Market-specific indicators, on-chain metrics, and sentiment signals
3. **Model Training**: Automated pipelines with hyperparameter optimization
4. **Deployment**: Continuous training and deployment to prediction services

## Integration Points

- **Backend API**: Models expose prediction endpoints consumed by the strategy engine
- **Smart Contracts**: Risk metrics inform contract execution parameters
- **Frontend**: Visualizations and insights derived from model predictions
- **Data Warehouse**: Historical predictions stored for backtesting and strategy refinement

## Getting Started

### Prerequisites
- Python 3.10+
- CUDA-compatible GPU (recommended)
- Access to Minos-AI data warehouse

### Installation
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set up environment
cp .env.example .env
# Edit .env with your configuration
```

### Training a Model
```bash
# Basic training with default parameters
python -m training.train --model price_prediction --dataset sol_usdc_1h

# Advanced configuration
python -m training.train --model price_prediction \
    --dataset sol_usdc_1h \
    --lookback 48 \
    --horizon 24 \
    --features price,volume,funding_rate,market_depth \
    --layers 3 \
    --units 128 \
    --dropout 0.2 \
    --batch_size 64 \
    --epochs 100
```

### Running Inference
```bash
# One-time prediction
python -m inference.predictor --model path/to/model.h5 --input path/to/input.json

# Start prediction service
python -m inference.predictor --serve --port 8080
```

## Development Guidelines

### Code Standards
- Follow PEP 8 style guide
- Use type hints for all function signatures
- Document classes and functions with docstrings (Google style)
- Maintain test coverage above 85%

### Model Development Process
1. Create experiment configuration in `experiments/`
2. Implement model architecture in `models/`
3. Add custom metrics if needed in `utils/metrics.py`
4. Train and evaluate model using `training/train.py`
5. Log performance metrics to MLflow
6. Review results and iterate
7. When satisfied, package model for deployment

### Contribution Workflow
1. Create feature branch from `develop`
2. Implement changes with tests
3. Run validation suite: `make validate`
4. Submit PR with experiment results
5. Peer review including performance validation
6. Merge to `develop` and deploy to staging

## Performance Benchmarks

| Model | Asset | Horizon | RMSE | Directional Accuracy | Sharpe Ratio |
|-------|-------|---------|------|---------------------|--------------|
| LSTM-Attention | SOL/USDC | 24h | 0.027 | 68.2% | 1.87 |
| LSTM-Attention | SOL/USDC | 7d | 0.054 | 61.3% | 1.42 |
| GRU-Wavenet | BTC/USDC | 24h | 0.019 | 65.7% | 1.65 |
| Transformer | Market Index | 24h | 0.022 | 63.9% | 1.56 |

## License
Proprietary and confidential. Copyright © 2025 Minos-AI.