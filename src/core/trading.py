import logging
from typing import Dict, Optional
import requests
from solana.rpc.api import Client
from src.config.settings import MAX_SLIPPAGE, MIN_LIQUIDITY_SOL
from src.utils.helpers import get_solana_client, format_amount, calculate_price_impact

logger = logging.getLogger(__name__)

class TradingBot:
    def __init__(self):
        self.client = get_solana_client()
        self.jupiter_base_url = "https://quote-api.jup.ag/v6"
    
    def get_token_price(self, token_address: str) -> Optional[float]:
        """Get token price in SOL."""
        try:
            # Use Jupiter's quote endpoint instead of price endpoint
            url = f"{self.jupiter_base_url}/quote"
            params = {
                "inputMint": "So11111111111111111111111111111111111111112",  # SOL
                "outputMint": token_address,
                "amount": "1000000000",  # 1 SOL in lamports
                "slippageBps": 50
            }
            response = requests.get(url, params=params)
            data = response.json()
            
            # Calculate price from the quote
            if 'outAmount' in data:
                return float(data['outAmount']) / 1000000000  # Convert to human readable
            return None
        except Exception as e:
            logger.error(f"Error getting token price: {e}")
            return None

    def check_liquidity(self, token_address: str) -> bool:
        """Check if token has sufficient liquidity."""
        try:
            # Get quote for 1 SOL
            url = f"{self.jupiter_base_url}/quote"
            params = {
                "inputMint": "So11111111111111111111111111111111111111112",  # SOL
                "outputMint": token_address,
                "amount": "1000000000",  # 1 SOL
                "slippageBps": 50
            }
            response = requests.get(url, params=params)
            data = response.json()
            
            # Check if route exists and price impact is reasonable
            if 'priceImpactPct' in data:
                price_impact = float(data['priceImpactPct'])
                return price_impact < 5.0  # Less than 5% price impact
            return False
        except Exception as e:
            logger.error(f"Error checking liquidity: {e}")
            return False

    def get_swap_quote(self, input_mint: str, output_mint: str, amount: int) -> Optional[Dict]:
        """Get swap quote from Jupiter."""
        try:
            url = f"{self.jupiter_base_url}/quote"
            params = {
                "inputMint": input_mint,
                "outputMint": output_mint,
                "amount": amount,
                "slippageBps": int(MAX_SLIPPAGE * 10000)
            }
            response = requests.get(url, params=params)
            return response.json()
        except Exception as e:
            logger.error(f"Error getting swap quote: {e}")
            return None

    def execute_swap(self, quote: Dict) -> bool:
        """Execute a swap transaction."""
        # TODO: Implement actual swap execution using Jupiter Swap API
        # This requires wallet integration and transaction signing
        logger.info("Swap execution not yet implemented")
        return False 