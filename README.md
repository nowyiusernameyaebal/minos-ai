<div align="center">
  <img src="docs/assets/minos-ai-logo.png" alt="Minos-AI Logo" width="400" />
  <h1>Minos-AI: Advanced AI-Powered DeFi Strategy Platform</h1>
  
  [![Build Status](https://github.com/minos-ai/minos-ai/workflows/CI/badge.svg)](https://github.com/minos-ai/minos-ai/actions)
  [![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
  [![npm version](https://img.shields.io/npm/v/@minos-ai/sdk.svg?style=flat)](https://www.npmjs.com/package/@minos-ai/sdk)
  [![Solana](https://img.shields.io/badge/Solana-v1.16-blueviolet)](https://solana.com/)
  [![Discord](https://img.shields.io/discord/956782867415343174?label=Discord&logo=discord&logoColor=white)](https://discord.gg/minos-ai)
  [![Twitter Follow](https://img.shields.io/twitter/follow/MinosAI?style=social)](https://twitter.com/MinosAI)
</div>

## 🧠 Overview

Minos-AI is a revolutionary DeFi strategy platform that leverages artificial intelligence to optimize trading strategies on the Solana blockchain. Our platform combines state-of-the-art machine learning models with on-chain vaults and smart contracts to create a secure, transparent, and efficient trading ecosystem.

### Key Features

- **AI-Powered Trading Strategies**: Deploy three specialized AI models:
  - **Ariadne**: Time-series prediction for market movements
  - **Androgeus**: Technical analysis with risk management
  - **Deucalion**: Social sentiment and on-chain analytics
  
- **Secure On-Chain Vaults**: Self-custodial smart contracts for managing assets with transparent execution

- **Decentralized Governance**: Community-driven protocol decisions through on-chain voting

- **Composable SDK**: Build custom applications and strategies with our TypeScript SDK

- **Real-Time Analytics**: Monitor performance with comprehensive analytics dashboard

## 🚀 Getting Started

### Prerequisites

- Node.js v18+
- Rust & Cargo v1.69+
- Solana CLI v1.16+
- Python 3.10+ (for AI model development)
- Docker & Docker Compose

### Installation

```bash
# Clone the repository
git clone https://github.com/minos-ai/minos-ai.git
cd minos-ai

# Install dependencies
npm install

# Setup environment
cp .env.example .env
# Edit .env with your configuration

# Build all packages
npm run build
```

### Development Environment

```bash
# Start local development environment
npm run dev

# In another terminal, start the indexer
npm run indexer:dev
```

### Deploy to Devnet

```bash
# Deploy Solana programs to devnet
npm run deploy:devnet

# Initialize the protocol
npm run init:devnet
```

## 📦 Project Structure

```
minos-ai/
├── packages/
│   ├── contracts/       # Solana smart contracts in Rust
│   ├── ai-models/       # ML models in Python
│   ├── indexer/         # Blockchain indexer service
│   └── backend/         # API and services
├── sdk/                 # TypeScript SDK for integrations
├── scripts/             # Development and deployment scripts
└── docker/              # Docker configurations
```

## 🛠️ Core Components

### Smart Contracts

Our protocol consists of three primary Solana programs:

1. **Vault Program**: Manages asset allocation, deposits, withdrawals, and strategy execution
2. **AI Agent Program**: Handles on-chain model inference and strategy implementation
3. **Governance Program**: Facilitates decentralized decision-making through proposals and voting

### AI Models

Minos-AI features three specialized AI models, each targeting different aspects of market analysis:

1. **Ariadne**: Time series forecasting model that predicts short to medium-term price movements using LSTM and Transformer architectures
2. **Androgeus**: Technical analysis model that identifies patterns and generates trading signals with risk-adjusted position sizing
3. **Deucalion**: Sentiment analysis model that processes social media, news, and on-chain data to gauge market sentiment

### SDK

Our TypeScript SDK provides a simple interface for interacting with the Minos-AI protocol:

```typescript
import { MinosClient, Vault, Strategy } from '@minos-ai/sdk';

// Initialize client
const client = new MinosClient({ cluster: 'mainnet-beta' });

// Create a vault with an AI strategy
const vault = await client.createVault({
  name: 'My AI Strategy',
  initialDeposit: 10, // SOL
  strategy: Strategy.ARIADNE_MOMENTUM,
  riskLevel: 'medium'
});

// Monitor performance
const performance = await vault.getPerformance();
console.log(`7-day ROI: ${performance.roi7d}%`);
```

## 📊 Performance

Our AI models have demonstrated strong performance in backtesting and live environments:

| Model      | 30-Day ROI | Sharpe Ratio | Max Drawdown | Win Rate |
|------------|------------|--------------|--------------|----------|
| Ariadne    | 14.3%      | 2.1          | 8.7%         | 68%      |
| Androgeus  | 11.8%      | 1.8          | 7.2%         | 63%      |
| Deucalion  | 9.2%       | 1.6          | 6.1%         | 59%      |

*Note: Past performance is not indicative of future results. Trading involves risk.*

## 🔒 Security

Security is our top priority. Our protocol implements:

- Comprehensive test coverage with unit and integration tests
- Multiple independent security audits
- Time-locked upgrades with multi-sig authorization
- Emergency pause mechanisms for critical vulnerabilities
- Bug bounty program for responsible disclosure

## 🌐 Community & Governance

Minos-AI is governed by the community through:

- On-chain governance proposals and voting
- Community forums for discussion and feedback
- Regular community calls and development updates
- Transparent roadmap and development process

Join our [Discord](https://discord.gg/minos-ai) to participate in the community.

## 📝 License

Minos-AI is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👥 Contributing

We welcome contributions from the community! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on how to contribute.

## 🔗 Links

- [Website](https://minos-ai.org)
- [Documentation](https://docs.minos-ai.org)
- [GitHub](https://github.com/minos-ai)
- [Discord](https://discord.gg/minos-ai)
- [Twitter](https://twitter.com/MinosAI)

---

<div align="center">
  <sub>Built with ❤️ by the Minos-AI Team</sub>
</div>