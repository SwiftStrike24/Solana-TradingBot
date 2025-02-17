# Solana Memecoin Sniping Bot

A Python-based trading bot for sniping Solana memecoins using Helius RPC and Jupiter Swap API.

## Setup

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: .\venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Create a `.env` file in the root directory with your keys:
```
HELIUS_API_KEY=your_helius_api_key_here
SOLANA_WALLET_PRIVATE_KEY=your_solana_wallet_private_key_here
```

## Project Structure

```
├── src/                    # Source code
│   ├── config/            # Configuration files
│   ├── core/              # Core bot logic
│   ├── utils/             # Utility functions
│   └── strategies/        # Trading strategies
├── logs/                  # Log files
├── tests/                 # Test files
├── .env                   # Environment variables
└── requirements.txt       # Project dependencies
```

## Usage

1. Configure your settings in `.env`
2. Run the bot:
```bash
python src/main.py
``` 