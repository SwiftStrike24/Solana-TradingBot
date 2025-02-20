# 🚀 Solana Advanced Trading Bot

A high-performance Solana trading bot leveraging Helius & Jito RPC for MEV protection, with Jupiter API integration for optimal swap routing and QuestDB analytics.

## ✨ Features

- **Multi-Source Price Oracle**
  - Jupiter Price API (primary)
  - USDC Quote Calculation (secondary)
  - CoinGecko API Integration (fallback)
  - Price caching for optimization
  - Real-time USD value conversion

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
  - Token validation & verification

- **QuestDB Analytics Integration**
  - Transaction metrics storage
  - RPC performance tracking
  - Success rate monitoring
  - Latency analysis
  - Fee analytics

- **Real-time Analytics**
  - Token price tracking
  - Wallet balance monitoring
  - Transaction cost analysis
  - Performance metrics
  - Rich console output
  - Dynamic fee calculations

- **MEV Protection**
  - Jito bundles support
  - Priority fees optimization
  - Transaction protection
  - Frontrunning prevention
  - Bundle ID tracking

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
COINGECKO_API_KEY=your_coingecko_api_key  # Optional
```

## 📁 Project Structure

```
├── src/
│   ├── config/            # Configuration & settings
│   ├── core/             # Core trading logic
│   │   └── trading.py    # Main trading implementation
│   ├── db/              # Database integration
│   │   └── questdb.py   # QuestDB metrics tracking
│   ├── utils/           # Helper functions
│   └── services/        # External API integrations
│       └── coingecko.py # CoinGecko price service
├── tests/               # Test suite
│   ├── test_swap.py    # Interactive swap testing
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
- Price caching system
- Efficient token info caching

## 🔒 Security Features

- Dynamic slippage protection
- Liquidity verification
- Token validation & verification
- MEV protection
- Transaction simulation
- Multiple price sources
- Error handling & logging

## 📊 Analytics

- QuestDB metrics integration
- RPC performance tracking
- Transaction cost analysis
- Token price monitoring
- Wallet balance tracking
- Success rate metrics
- Latency measurements

## 🛡 Transaction Protection

- Priority fees: 0.00021 SOL (70%)
- Jito MEV tip: 0.00009 SOL (30%)
- Dynamic slippage adjustment
- Route optimization
- Frontrunning protection
- Transaction bundling

## 📈 Monitoring

Rich console output includes:
- Wallet status with USD values
- Token information & verification
- Detailed swap quotes
- Fee breakdowns in SOL & USD
- Transaction status & confirmations
- Performance metrics
- Dynamic slippage analysis
- MEV protection details

## 🔧 Advanced Features

- Token price impact analysis
- Multi-source price fetching
- Automatic fee optimization
- Dynamic route selection
- Comprehensive error handling
- Detailed logging system
- Transaction simulation
- Token validation checks

## Requirements

- Python 3.8+
- Solana wallet
- Helius API key
- QuestDB instance
- CoinGecko API key (optional)

## 📝 Notes

- Supports both mainnet and devnet
- Includes comprehensive error handling
- Features detailed logging system
- Optimized for memecoin trading
- Built-in protection mechanisms 