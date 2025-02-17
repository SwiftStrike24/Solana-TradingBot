import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# API Keys
HELIUS_API_KEY = os.getenv("HELIUS_API_KEY")
SOLANA_WALLET_PRIVATE_KEY = os.getenv("SOLANA_WALLET_PRIVATE_KEY")

# RPC Endpoints
HELIUS_RPC_URL = f"https://rpc.helius.xyz/?api-key={HELIUS_API_KEY}"

# Trading Parameters
MIN_LIQUIDITY_SOL = 1.0  # Minimum liquidity in SOL
MAX_SLIPPAGE = 0.05  # Maximum slippage tolerance (5%)
GAS_ADJUSTMENT = 1.5  # Gas price adjustment factor

# Webhook Settings
WEBHOOK_PORT = 8080
WEBHOOK_HOST = "0.0.0.0"

# Logging
LOG_LEVEL = "INFO"
LOG_FILE = "logs/trading_bot.log" 