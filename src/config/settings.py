import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# API Keys
HELIUS_API_KEY = os.getenv("HELIUS_API_KEY")
SOLANA_WALLET_PRIVATE_KEY = os.getenv("SOLANA_WALLET_PRIVATE_KEY")

# RPC Endpoints
HELIUS_RPC_URL = f"https://rpc.helius.xyz/?api-key={HELIUS_API_KEY}"
# Using Salt Lake City endpoint for Calgary (closest region)
JITO_RPC_URL = "https://slc.mainnet.block-engine.jito.wtf"

# Trading Parameters
MIN_LIQUIDITY_SOL = 1.0  # Minimum liquidity in SOL
MAX_SLIPPAGE = 0.05  # Maximum slippage tolerance (5%)
GAS_ADJUSTMENT = 1.5  # Gas price adjustment factor
PRIORITY_FEE_LAMPORTS = 7_000  # 0.000007 SOL (70% of total fee)
JITO_TIP_LAMPORTS = 3_000  # 0.000003 SOL (30% of total fee)

# Webhook Settings
WEBHOOK_PORT = 8080
WEBHOOK_HOST = "0.0.0.0"

# Logging
LOG_LEVEL = "INFO"
LOG_FILE = "logs/trading_bot.log"

# QuestDB Settings
QUESTDB_HOST = "localhost"
QUESTDB_PORT = 8812
QUESTDB_USER = "admin"
QUESTDB_PASSWORD = "quest" 