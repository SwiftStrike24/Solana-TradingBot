import os
import aiohttp
from typing import Optional

class CoinGeckoAPI:
    def __init__(self):
        self.api_key = os.getenv("COINGECKO_API_KEY")
        self.base_url = "https://api.coingecko.com/api/v3"
        self.headers = {"x-cg-api-key": self.api_key}

    async def get_sol_price(self) -> Optional[float]:
        """Get SOL price in USD from CoinGecko"""
        try:
            url = f"{self.base_url}/simple/price"
            params = {
                "ids": "solana",
                "vs_currencies": "usd"
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params, headers=self.headers) as response:
                    if response.status == 200:
                        data = await response.json()
                        return float(data["solana"]["usd"])
            return None
        except Exception as e:
            return None 