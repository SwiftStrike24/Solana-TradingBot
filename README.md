# 🚀 Solana Advanced Trading Bot

A high-performance Solana trading bot leveraging Helius & Jito RPC for MEV protection, with Jupiter API integration for optimal swap routing.

## ✨ Features

- **Dual RPC Support**
  - Helius RPC for high performance
  - Jito RPC for MEV protection
  - Automatic RPC failover
  - Performance tracking & analytics

- **Advanced Swap Features**
  - Dynamic slippage adjustment
  - Multi-route optimization
  - Price impact analysis
  - Liquidity verification
  - Gas optimization

- **Real-time Analytics**
  - Token price tracking
  - Wallet balance monitoring
  - Transaction cost analysis
  - Performance metrics
  - Rich console output

- **MEV Protection**
  - Jito bundles support
  - Priority fees optimization
  - Transaction protection
  - Frontrunning prevention

## 🛠 Setup

1. Create virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # Windows: .\venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Configure `.env`:
```env
HELIUS_API_KEY=your_helius_api_key
SOLANA_WALLET_PRIVATE_KEY=your_wallet_private_key
```

## 📁 Project Structure

```
├── src/
│   ├── config/            # Configuration & settings
│   ├── core/             # Core trading logic
│   │   └── trading.py    # Main trading implementation
│   ├── utils/           # Helper functions
│   └── services/        # External API integrations
├── tests/               # Test suite
│   ├── test_swap.py    # Swap testing interface
│   └── test_token.py   # Token analysis tools
├── logs/               # Performance & error logs
└── requirements.txt    # Dependencies
```

## 💫 Usage

### Interactive Testing
```bash
python tests/test_swap.py
```

### Production Bot
```bash
python src/main.py
```

## ⚡ Performance Features

- Concurrent RPC requests for minimal latency
- Dynamic compute unit adjustment
- Optimized priority fees
- MEV protection via Jito bundles
- Performance analytics & tracking

## 🔒 Security Features

- Dynamic slippage protection
- Liquidity verification
- Token validation
- MEV protection
- Transaction simulation

## 📊 Analytics

- RPC performance tracking
- Transaction cost analysis
- Token price monitoring
- Wallet balance tracking
- Success rate metrics

## 🛡 Transaction Protection

- Priority fees: 0.00021 SOL (70%)
- Jito MEV tip: 0.00009 SOL (30%)
- Dynamic slippage adjustment
- Route optimization
- Frontrunning protection

## 📈 Monitoring

Rich console output includes:
- Wallet status
- Token information
- Swap quotes
- Fee breakdowns
- Transaction status
- Performance metrics

## Requirements

- Python 3.8+
- Solana wallet
- Helius API key 