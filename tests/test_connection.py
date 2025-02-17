import os
import sys
import asyncio

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.trading import TradingBot
from src.utils.helpers import setup_logging

async def test_basic_functions():
    # Setup logging
    logger = setup_logging()
    logger.info("Starting test...")
    
    bot = TradingBot()
    
    # Test with BONK token address (a known Solana memecoin)
    bonk_address = "DezXAZ8z7PnrnRJjz3wXBoRgixCa6xjnB7YaB1pPB263"
    
    # Test price check
    price = bot.get_token_price(bonk_address)
    print(f"BONK Price in SOL: {price}")
    
    # Test liquidity check
    has_liquidity = bot.check_liquidity(bonk_address)
    print(f"BONK has sufficient liquidity: {has_liquidity}")
    
    # Test swap quote
    quote = bot.get_swap_quote(
        input_mint="So11111111111111111111111111111111111111112",  # SOL
        output_mint=bonk_address,  # BONK
        amount=1000000000  # 1 SOL in lamports
    )
    print(f"Swap quote for 1 SOL to BONK: {quote}")

if __name__ == "__main__":
    asyncio.run(test_basic_functions())
