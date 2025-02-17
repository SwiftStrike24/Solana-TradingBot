import logging
from solana.rpc.api import Client
from src.config.settings import HELIUS_RPC_URL, LOG_FILE, LOG_LEVEL

def setup_logging():
    """Configure logging for the bot."""
    logging.basicConfig(
        level=LOG_LEVEL,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(LOG_FILE),
            logging.StreamHandler()
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