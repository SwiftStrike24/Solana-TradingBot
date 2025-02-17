import logging
from solana.rpc.api import Client
from src.config.settings import HELIUS_RPC_URL, LOG_LEVEL
import sys

def setup_logging(log_file=None):
    """Configure logging for the bot."""
    if log_file is None:
        log_file = "logs/trading_bot.log"
        
    log_format = (
        '%(asctime)s | '
        '%(levelname)-8s | '
        '%(name)-12s | '
        '%(message)s'
    )
    
    date_format = '%Y-%m-%d %I:%M:%S %p'
    
    logging.basicConfig(
        level=LOG_LEVEL,
        format=log_format,
        datefmt=date_format,
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)

def get_solana_client():
    """Create and return a Solana client instance."""
    return Client(HELIUS_RPC_URL)

def format_amount(amount: float, decimals: int = 9) -> int:
    """Convert human-readable amount to lamports."""
    return int(amount * (10 ** decimals))

def calculate_price_impact(input_amount: float, output_amount: float) -> float:
    """Calculate the price impact of a trade."""
    return abs(1 - (output_amount / input_amount))

def is_valid_token_address(client: Client, address: str) -> bool:
    """Verify if an address is a valid SPL token."""
    try:
        response = client.get_account_info(address)
        return bool(response['result']['value'])
    except Exception:
        return False 