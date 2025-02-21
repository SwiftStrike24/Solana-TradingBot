# 🚀 Solana Advanced Trading Bot

A high-performance Solana trading bot with advanced MEV protection, multi-RPC failover, and comprehensive analytics. Built for memecoin trading with robust safety features and real-time performance monitoring.

## ✨ Core Features

### 🔄 Trading Engine
- **Multi-RPC Architecture**
  - Primary: Helius RPC (High Performance)
  - Secondary: Jito RPC (MEV Protection)
  - Automatic failover with latency-based routing
  - Parallel RPC request execution
  - Dynamic compute unit adjustment

- **Advanced Swap Features**
  - Dynamic slippage calculation based on token category
  - Multi-route optimization with parallel execution
  - Real-time price impact analysis
  - Automatic gas fee optimization
  - MEV protection via Jito bundles
  - Transaction simulation & validation

### 📊 Analytics & Monitoring

- **QuestDB Integration**
  - Real-time transaction metrics
  - RPC performance tracking
  - Success rate monitoring
  - Latency analysis
  - Fee analytics
  - Token metrics storage

- **Performance Metrics**
  - RPC latency tracking (min/max/avg)
  - Success rate monitoring
  - Compute unit optimization
  - Gas fee analysis
  - Route efficiency tracking

### 🛡️ Safety Features

- **Transaction Protection**
  - Priority Fee: 0.000007 SOL (70%) - [Configure in settings.py](src/config/settings.py)
  - Jito MEV Tip: 0.000003 SOL (30%) - [Configure in settings.py](src/config/settings.py)
  - Dynamic slippage adjustment
  - Frontrunning protection
  - Transaction bundling
  - Automatic retry mechanism

- **Token Validation**
  - Liquidity verification
  - Market cap analysis
  - Price impact checks
  - Token contract verification
  - Trading volume analysis

### 💹 Price Oracle System

- **Multi-Source Price Feed**
  - Jupiter Price API (Primary)
  - USDC Quote Calculation (Secondary)
  - CoinGecko API (Fallback)
  - Real-time price updates
  - Price deviation checks

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

## 🛠 Setup

1. **Environment Setup**
```bash
python -m venv venv
source venv/bin/activate  # Windows: .\venv\Scripts\activate
pip install -r requirements.txt
```

2. **Configuration**
```env
HELIUS_API_KEY=your_helius_api_key
SOLANA_WALLET_PRIVATE_KEY=your_wallet_private_key
COINGECKO_API_KEY=your_coingecko_api_key  # Optional
```

3. **QuestDB Setup**
```env
QUESTDB_HOST=localhost
QUESTDB_PORT=8812
QUESTDB_USER=admin
QUESTDB_PASSWORD=quest
```

## 📊 Interactive Testing

```bash
python tests/test_swap.py
```

Features:
- Single swap testing ($1 trades)
- Batch testing (5x $1 swaps)
- Real-time performance metrics
- Detailed fee breakdown
- Token analysis
- Wallet monitoring

## 🔍 Monitoring Features

- **Rich Console Output**
  - Live wallet status with USD values
  - Token verification status
  - Detailed swap quotes
  - Fee breakdowns (SOL & USD)
  - Transaction status tracking
  - RPC performance metrics
  - Slippage analysis

- **Transaction Analytics**
  - Route optimization details
  - MEV protection status
  - Gas optimization metrics
  - Price impact analysis
  - Liquidity depth indicators
  - Token market metrics

## 🚀 Performance Optimizations

- **RPC Optimization**
  - Parallel request execution
  - Connection pooling
  - DNS caching
  - Automatic retry logic
  - Dynamic timeout adjustment
  - Load balancing

- **Transaction Processing**
  - Async/await architecture
  - Thread pool for CPU tasks
  - Optimized memory usage
  - Efficient error handling
  - Resource cleanup

## 📈 Advanced Features

- **Batch Testing**
  - Multiple swap execution
  - Performance aggregation
  - Success rate tracking
  - RPC distribution analysis
  - Cost analysis
  - Token metrics

- **Dynamic Adjustments**
  - Slippage calculation
  - Gas fee optimization
  - Route selection
  - RPC failover
  - Error recovery
  - Performance tuning

## 🔧 System Requirements

- Python 3.8+
- QuestDB instance
- Solana wallet
- Helius API key
- Windows/Linux/MacOS support
- 2GB RAM minimum

## 🛡️ Safety Notes

- Built-in slippage protection
- MEV protection via Jito
- Multiple price sources
- Transaction simulation
- Token validation
- Error recovery
- Resource cleanup

## 📝 Latest Updates

- Windows event loop compatibility
- Enhanced error handling
- Improved QuestDB integration
- Better RPC failover
- Optimized batch testing
- Advanced metrics tracking
- Rich console output
- Transaction protection

## 🔗 Dependencies

Core dependencies:
- `solders`: Solana transaction handling
- `psycopg`: QuestDB async connection
- `aiohttp`: Async HTTP client
- `rich`: Console output formatting
- `numba`: Performance optimization
- `asyncio`: Async operations

## 📚 Documentation

For detailed documentation on each component:
- [Trading Engine](src/core/trading.py)
- [QuestDB Integration](src/db/questdb.py)
- [Test Suite](tests/test_swap.py)
- [Configuration](src/config/settings.py) 