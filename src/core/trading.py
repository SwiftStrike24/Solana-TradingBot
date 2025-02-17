import logging
from typing import Dict, Optional
import requests
from solana.rpc.api import Client
from solders.pubkey import Pubkey
from src.config.settings import MAX_SLIPPAGE, MIN_LIQUIDITY_SOL
from src.utils.helpers import get_solana_client, format_amount, calculate_price_impact
import asyncio
import json

logger = logging.getLogger(__name__)

class TradingBot:
    def __init__(self):
        self.client = get_solana_client()
        self.jupiter_swap_url = "https://api.jup.ag/swap/v1"  # Updated swap endpoint
        self.jupiter_price_url = "https://api.jup.ag/price/v2"  # New price endpoint
        self.jupiter_token_url = "https://api.jup.ag/tokens/v1"  # Already correct
        self.request_delay = 0.5  # 500ms delay between requests
        self.sol_mint = "So11111111111111111111111111111111111111112"
        self._token_cache = {}  # Cache for token info
    
    def _validate_address(self, address: str) -> Optional[str]:
        """Validate and convert address to string format."""
        try:
            return str(Pubkey.from_string(address))
        except ValueError as e:
            logger.error(f"Invalid address format: {str(e)}")
            return None
        except Exception as e:
            logger.error(f"Error validating address: {str(e)}")
            return None
    
    async def get_token_info(self, token_address: str) -> Optional[dict]:
        """Get token information from Jupiter."""
        try:
            validated_address = self._validate_address(token_address)
            if not validated_address:
                return None
                
            url = f"{self.jupiter_token_url}/token/{validated_address}"
            logger.info(f"\n{'='*80}\nFetching token info for: {validated_address}\nURL: {url}\n{'='*80}")
            
            response = requests.get(url)
            
            if not response.ok:
                logger.error(f"Failed to get token info: Status {response.status_code}\nResponse: {response.text}")
                return None
                
            token_info = response.json()
            logger.info(f"\n{'='*80}\nToken Info Response:\n{json.dumps(token_info, indent=2)}\n{'='*80}")
            
            return {
                'symbol': token_info.get('symbol', 'Unknown'),
                'name': token_info.get('name', 'Unknown Token'),
                'decimals': token_info.get('decimals', 6),
                'tags': token_info.get('tags', []),
                'daily_volume': token_info.get('daily_volume', 0),
                'logo_uri': token_info.get('logoURI'),
                'verified': 'verified' in token_info.get('tags', []),
                'created_at': token_info.get('created_at'),
                'mint_authority': token_info.get('mint_authority'),
                'freeze_authority': token_info.get('freeze_authority'),
                'permanent_delegate': token_info.get('permanent_delegate')
            }
        except Exception as e:
            logger.error(f"Error getting token info:\n{str(e)}\n{'='*80}")
            return None
    
    async def get_token_price(self, token_address: str) -> Optional[dict]:
        """Get comprehensive token price data."""
        try:
            # Validate token address
            validated_address = self._validate_address(token_address)
            if not validated_address:
                return None
            
            # Get token info first
            token_info = await self.get_token_info(validated_address)
            decimals = token_info['decimals'] if token_info else 6
            
            # Get price for 1 SOL worth of tokens
            sol_quote = await self._get_quote(
                input_mint=self.sol_mint,
                output_mint=validated_address,
                amount=1_000_000_000,  # 1 SOL in lamports
            )
            
            if not sol_quote:
                logger.error(f"Failed to get SOL quote for {validated_address}")
                return None
                
            # Calculate tokens per SOL
            tokens_per_sol = float(sol_quote['outAmount']) / (10 ** decimals)
            
            # Get reverse quote (1 token worth of SOL)
            token_quote = await self._get_quote(
                input_mint=validated_address,
                output_mint=self.sol_mint,
                amount=10 ** decimals  # 1 token in base units
            )
            
            if not token_quote:
                logger.warning(f"Failed to get token quote, using fallback calculation")
                sol_per_token = 1.0 / tokens_per_sol if tokens_per_sol != 0 else 0
            else:
                # Calculate SOL per token directly from quote
                sol_per_token = float(token_quote['outAmount']) / 1e9  # Convert from lamports to SOL
            
            return {
                'token_info': token_info,
                'tokens_per_sol': tokens_per_sol,
                'sol_per_token': sol_per_token,
                'price_impact': sol_quote.get('priceImpactPct', 0),
                'routes': sol_quote.get('routePlan', []),
                'market_info': {
                    'slippage': sol_quote.get('slippageBps', 0) / 100,
                    'fee': sum(float(r['swapInfo']['feeAmount']) for r in sol_quote.get('routePlan', []))
                }
            }
            
        except Exception as e:
            logger.error(f"Error getting token price: {str(e)}")
            return None

    async def _get_quote(self, input_mint: str, output_mint: str, amount: int) -> Optional[dict]:
        """Get quote from Jupiter."""
        try:
            await asyncio.sleep(self.request_delay)
            url = f"{self.jupiter_swap_url}/quote"
            params = {
                "inputMint": input_mint,
                "outputMint": output_mint,
                "amount": str(amount),
                "slippageBps": 50
            }
            response = requests.get(url, params=params)
            return response.json() if response.ok else None
        except Exception as e:
            logger.error(f"Error getting quote: {str(e)}")
            return None

    def check_liquidity(self, token_address: str) -> dict:
        """Check token liquidity metrics."""
        try:
            # Validate token address
            validated_address = self._validate_address(token_address)
            if not validated_address:
                return {'has_liquidity': False, 'reason': 'Invalid token address'}
            
            quote = self.get_swap_quote(
                input_mint=self.sol_mint,
                output_mint=validated_address,
                amount=int(MIN_LIQUIDITY_SOL * 1e9)
            )
            
            if not quote:
                return {'has_liquidity': False, 'reason': 'No route found'}
                
            price_impact = float(quote.get('priceImpactPct', 100))
            
            return {
                'has_liquidity': price_impact < MAX_SLIPPAGE * 100,
                'price_impact': price_impact,
                'routes_available': len(quote.get('routePlan', [])),
                'total_fees': sum(float(r['swapInfo']['feeAmount']) for r in quote.get('routePlan', []))
            }
        except Exception as e:
            logger.error(f"Error checking liquidity: {str(e)}")
            return {'has_liquidity': False, 'reason': str(e)}

    def get_swap_quote(self, input_mint: str, output_mint: str, amount: int) -> Optional[Dict]:
        """Get swap quote from Jupiter."""
        try:
            # Validate input and output mints
            validated_input = self._validate_address(input_mint)
            validated_output = self._validate_address(output_mint)
            
            if not validated_input or not validated_output:
                return None
            
            url = f"{self.jupiter_swap_url}/quote"
            params = {
                "inputMint": validated_input,
                "outputMint": validated_output,
                "amount": str(amount),
                "slippageBps": int(MAX_SLIPPAGE * 10000)
            }
            
            logger.info(f"\n{'='*80}\nFetching swap quote\nURL: {url}\nParams: {json.dumps(params, indent=2)}\n{'='*80}")
            
            response = requests.get(url, params=params)
            
            if not response.ok:
                logger.error(f"Failed to get swap quote: Status {response.status_code}\nResponse: {response.text}")
                return None
                
            quote = response.json()
            logger.info(f"\n{'='*80}\nSwap Quote Response:\n{json.dumps(quote, indent=2)}\n{'='*80}")
            
            return quote
        except Exception as e:
            logger.error(f"Error getting swap quote:\n{str(e)}\n{'='*80}")
            return None

    def execute_swap(self, quote: Dict) -> bool:
        """Execute a swap transaction."""
        # TODO: Implement actual swap execution using Jupiter Swap API
        # This requires wallet integration and transaction signing
        logger.info("Swap execution not yet implemented")
        return False

    async def get_new_tokens(self, limit: int = 100, offset: int = 0) -> Optional[list]:
        """Get list of new tokens from Jupiter API."""
        try:
            url = f"{self.jupiter_token_url}/new"
            logger.info(f"\n{'='*80}\nFetching new tokens\nURL: {url}\nLimit: {limit}, Offset: {offset}\n{'='*80}")
            
            params = {
                "limit": limit,
                "offset": offset
            }
            response = requests.get(url, params=params)
            
            if not response.ok:
                logger.error(f"Failed to get new tokens: Status {response.status_code}\nResponse: {response.text}")
                return None
                
            tokens = response.json()
            logger.info(f"\n{'='*80}\nNew Tokens Response:\n{json.dumps(tokens[:5], indent=2)}\n\nTotal tokens received: {len(tokens)}\n{'='*80}")
            
            # Transform response to match expected format
            return [{
                'mint': token.get('mint'),
                'symbol': token.get('symbol', 'Unknown'),
                'name': token.get('name', 'Unknown Token'),
                'decimals': token.get('decimals', 6),
                'logo_uri': token.get('logo_uri'),
                'created_at': token.get('created_at'),
                'known_markets': token.get('known_markets', []),
                'mint_authority': token.get('mint_authority'),
                'freeze_authority': token.get('freeze_authority')
            } for token in tokens]
            
        except Exception as e:
            logger.error(f"Error getting new tokens:\n{str(e)}\n{'='*80}")
            return None

    async def get_token_info_batch(self, token_addresses: list) -> dict:
        """Get token information for multiple tokens at once."""
        token_info = {}
        
        for address in token_addresses:
            if address in self._token_cache:
                token_info[address] = self._token_cache[address]
                continue
                
            try:
                validated_address = self._validate_address(address)
                if not validated_address:
                    continue
                    
                url = f"{self.jupiter_token_url}/token/{validated_address}"
                response = requests.get(url)
                
                if response.ok:
                    data = response.json()
                    info = {
                        'symbol': data.get('symbol', '...'),
                        'name': data.get('name', 'Unknown Token'),
                        'decimals': data.get('decimals', 6)
                    }
                    token_info[address] = info
                    self._token_cache[address] = info  # Cache the result
                    
                await asyncio.sleep(0.1)  # Small delay between requests
                
            except Exception as e:
                logger.error(f"Error fetching token info for {address}: {str(e)}")
                token_info[address] = {'symbol': '...', 'name': 'Unknown Token', 'decimals': 6}
        
        return token_info 