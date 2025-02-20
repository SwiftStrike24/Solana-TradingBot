import os
import aiohttp
from typing import Optional, Dict
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class CoinGeckoAPI:
    def __init__(self):
        self.api_key = os.getenv("COINGECKO_API_KEY")
        self.base_url = "https://api.coingecko.com/api/v3"
        self.headers = {"x-cg-api-key": self.api_key} if self.api_key else {}
        self._price_cache = {}
        self._cache_duration = timedelta(minutes=2)  # Cache prices for 2 minutes

    async def get_sol_price(self) -> Optional[float]:
        """Get SOL price in USD from CoinGecko with caching"""
        try:
            # Check cache first
            cache_entry = self._price_cache.get('solana')
            if cache_entry and datetime.now() - cache_entry['timestamp'] < self._cache_duration:
                return cache_entry['price']

            url = f"{self.base_url}/simple/price"
            params = {
                "ids": "solana",
                "vs_currencies": "usd"
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params, headers=self.headers) as response:
                    if response.status == 200:
                        data = await response.json()
                        price = float(data["solana"]["usd"])
                        # Update cache
                        self._price_cache['solana'] = {
                            'price': price,
                            'timestamp': datetime.now()
                        }
                        return price
                    else:
                        logger.error(f"CoinGecko API error: {response.status} - {await response.text()}")
            return None
        except Exception as e:
            logger.error(f"Error getting SOL price from CoinGecko: {str(e)}")
            return None

    async def get_token_price(self, contract_address: str, vs_currency: str = "usd") -> Optional[Dict]:
        """Get token price and market data from CoinGecko"""
        try:
            url = f"{self.base_url}/simple/token_price/solana"
            params = {
                "contract_addresses": contract_address,
                "vs_currencies": vs_currency,
                "include_market_cap": "true",
                "include_24hr_vol": "true",
                "include_24hr_change": "true"
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params, headers=self.headers) as response:
                    if response.status == 200:
                        data = await response.json()
                        if contract_address in data:
                            return {
                                'price': data[contract_address].get(vs_currency),
                                'market_cap': data[contract_address].get(f"{vs_currency}_market_cap"),
                                'volume_24h': data[contract_address].get(f"{vs_currency}_24h_vol"),
                                'price_change_24h': data[contract_address].get(f"{vs_currency}_24h_change")
                            }
            return None
        except Exception as e:
            logger.error(f"Error getting token price from CoinGecko: {str(e)}")
            return None 