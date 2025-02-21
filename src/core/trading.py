import logging
from typing import Dict, Optional, List, Tuple
import requests
from solana.rpc.api import Client
from solders.pubkey import Pubkey
from solders.transaction import VersionedTransaction
from solders.message import MessageV0
from solders.signature import Signature
from solana.rpc.commitment import Commitment
from src.config.settings import (
    MAX_SLIPPAGE, MIN_LIQUIDITY_SOL, HELIUS_RPC_URL, HELIUS_API_KEY,
    JITO_RPC_URL, PRIORITY_FEE_LAMPORTS, JITO_TIP_LAMPORTS
)
from src.utils.helpers import get_solana_client, format_amount, calculate_price_impact
import asyncio
import json
import base64
from solders.transaction import Transaction
from solders.keypair import Keypair
import time
from src.db.questdb import QuestDBClient, RPCMetrics
from datetime import datetime
from src.services.coingecko import CoinGeckoAPI
import aiohttp
from numba import jit, prange
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import functools
import os

logger = logging.getLogger(__name__)

# Numba optimized functions for CPU-intensive operations
@jit(nopython=True, parallel=True)
def _calculate_price_impacts(amounts: np.ndarray, pool_sizes: np.ndarray) -> np.ndarray:
    """Optimized parallel price impact calculation"""
    impacts = np.zeros_like(amounts, dtype=np.float64)
    for i in prange(len(amounts)):
        impacts[i] = amounts[i] / (pool_sizes[i] + amounts[i]) * 100
    return impacts

@jit(nopython=True)
def _calculate_optimal_route(routes: np.ndarray, fees: np.ndarray) -> Tuple[int, float]:
    """Optimized route selection based on fees and latency"""
    route_scores = routes * (1 - fees)
    best_idx = np.argmax(route_scores)
    return best_idx, route_scores[best_idx]

class TradingBot:
    def __init__(self, keypair=None):
        # Initialize with proper RPC URL and headers
        self.client = Client(HELIUS_RPC_URL, commitment=Commitment("confirmed"))
        self.jito_client = Client(JITO_RPC_URL, commitment=Commitment("confirmed"))
        self.headers = {
            "Content-Type": "application/json",
        }
        self.jupiter_swap_url = "https://api.jup.ag/swap/v1"
        self.jupiter_price_url = "https://api.jup.ag/price/v2"
        self.jupiter_token_url = "https://api.jup.ag/tokens/v1"
        self.request_delay = 0.5
        self.sol_mint = "So11111111111111111111111111111111111111112"
        self.keypair = keypair
        self._token_cache = {}
        self.questdb = QuestDBClient()
        self._session = None
        self._latency_history = []
        self.MAX_LATENCY_HISTORY = 100
        
        # Add thread pool for CPU-intensive tasks
        self._thread_pool = ThreadPoolExecutor(
            max_workers=min(32, (os.cpu_count() or 1) * 4),
            thread_name_prefix="trading_worker"
        )
        
        # Optimize session management
        self._session_pool = {}
        self._session_semaphore = asyncio.Semaphore(100)  # Limit concurrent connections
    
    async def _get_session(self, endpoint: str) -> aiohttp.ClientSession:
        """Get or create optimized aiohttp session with connection pooling"""
        if endpoint not in self._session_pool or self._session_pool[endpoint].closed:
            connector = aiohttp.TCPConnector(
                limit=0,  # No limit on connections
                ttl_dns_cache=300,  # Cache DNS results for 5 minutes
                use_dns_cache=True,
                ssl=False  # Disable SSL for internal RPCs if secure
            )
            self._session_pool[endpoint] = aiohttp.ClientSession(
                connector=connector,
                headers=self.headers,
                timeout=aiohttp.ClientTimeout(total=10)
            )
        return self._session_pool[endpoint]

    async def _parallel_rpc_request(self, method: str, params: list, endpoints: List[str] = None) -> Tuple[Optional[dict], str, float]:
        """Execute RPC request across multiple endpoints in parallel"""
        if endpoints is None:
            endpoints = [HELIUS_RPC_URL, JITO_RPC_URL]
        elif isinstance(endpoints, str):
            endpoints = [endpoints]  # Convert single endpoint to list
            
        def get_rpc_name(endpoint: str) -> str:
            """Get clean RPC name from endpoint URL"""
            if "helius" in endpoint.lower():
                return "Helius"
            elif "jito" in endpoint.lower():
                return "Jito"
            return "Unknown"
            
        async def try_endpoint(endpoint: str) -> Tuple[Optional[dict], str, float]:
            start_time = time.perf_counter()
            try:
                session = await self._get_session(endpoint)
                async with self._session_semaphore:
                    payload = {
                        "jsonrpc": "2.0",
                        "id": str(int(time.time() * 1000)),
                        "method": method,
                        "params": params
                    }
                    logger.info(f"Sending RPC request to {get_rpc_name(endpoint)}...")
                    
                    async with session.post(endpoint, json=payload) as response:
                        if response.status == 200:
                            result = await response.json()
                            latency = (time.perf_counter() - start_time) * 1000
                            
                            # Check for RPC-specific errors
                            if "error" in result:
                                error = result["error"]
                                logger.error(f"RPC error from {get_rpc_name(endpoint)}: {error}")
                                return None, get_rpc_name(endpoint), float('inf')
                                
                            return result, get_rpc_name(endpoint), latency
                        else:
                            text = await response.text()
                            logger.error(f"RPC request failed for {get_rpc_name(endpoint)} with status {response.status}: {text}")
            except Exception as e:
                logger.error(f"RPC request failed for {get_rpc_name(endpoint)}: {str(e)}")
            return None, get_rpc_name(endpoint), float('inf')

        try:
            # Execute requests in parallel
            tasks = [try_endpoint(endpoint) for endpoint in endpoints]
            responses = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Filter successful responses and handle exceptions
            valid_responses = []
            for response in responses:
                if isinstance(response, Exception):
                    logger.error(f"RPC request failed with exception: {str(response)}")
                    continue
                if response[0] is not None:  # Check if we got a valid response
                    valid_responses.append(response)
            
            if not valid_responses:
                logger.error("All RPC endpoints failed")
                return None, "Unknown", float('inf')
                
            # Get fastest successful response
            return min(valid_responses, key=lambda x: x[2])
            
        except Exception as e:
            logger.error(f"Parallel RPC request failed: {str(e)}")
            return None, "Unknown", float('inf')

    async def _fetch(self, url: str, method: str = "get", **kwargs) -> Optional[dict]:
        """Generic fetch method with latency tracking"""
        session = await self._get_session(url)
        start_time = time.perf_counter()
        try:
            async with getattr(session, method)(url, **kwargs) as response:
                latency = (time.perf_counter() - start_time) * 1000  # Convert to ms
                self._update_latency_history(latency)
                
                if response.status == 200:
                    return await response.json()
                logger.error(f"API request failed: {response.status} - {await response.text()}")
                return None
        except Exception as e:
            logger.error(f"Error in _fetch: {str(e)}")
            return None

    def _update_latency_history(self, latency: float):
        """Update rolling latency history"""
        self._latency_history.append(latency)
        if len(self._latency_history) > self.MAX_LATENCY_HISTORY:
            self._latency_history.pop(0)

    async def get_avg_latency(self) -> float:
        """Get average latency from history"""
        if not self._latency_history:
            return 0
        return sum(self._latency_history) / len(self._latency_history)

    async def get_dynamic_delay(self) -> float:
        """Calculate dynamic delay based on latency history"""
        avg_latency = await self.get_avg_latency()
        # Convert ms to seconds and cap at reasonable limits
        return min(max(avg_latency / 2000, 0.01), 0.1)  # Min 10ms, Max 100ms

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
            
            # Handle special addresses
            if validated_address in ["11111111111111111111111111111111", "So11111111111111111111111111111111111111112"]:
                return {
                    'symbol': 'SOL',
                    'name': 'Solana',
                    'decimals': 9,
                    'tags': ['native'],
                    'daily_volume': None,
                    'logo_uri': 'https://raw.githubusercontent.com/solana-labs/token-list/main/assets/mainnet/So11111111111111111111111111111111111111112/logo.png',
                    'verified': True,
                    'created_at': None,
                    'mint_authority': None,
                    'freeze_authority': None,
                    'permanent_delegate': None
                }

            url = f"{self.jupiter_token_url}/token/{validated_address}"
            logger.info(f"\n{'='*80}\nFetching token info for: {validated_address}\nURL: {url}\n{'='*80}")
            
            # Use dynamic delay based on latency history
            delay = await self.get_dynamic_delay()
            await asyncio.sleep(delay)
            
            response = await self._fetch(url)
            if not response:
                logger.warning(f"Failed to get token info for {validated_address}")
                return None
                
            logger.info(f"\n{'='*80}\nToken Info Response:\n{json.dumps(response, indent=2)}\n{'='*80}")
            
            return {
                'symbol': response.get('symbol', 'Unknown'),
                'name': response.get('name', 'Unknown Token'),
                'decimals': response.get('decimals', 6),
                'tags': response.get('tags', []),
                'daily_volume': response.get('daily_volume', 0),
                'logo_uri': response.get('logoURI'),
                'verified': 'verified' in response.get('tags', []),
                'created_at': response.get('created_at'),
                'mint_authority': response.get('mint_authority'),
                'freeze_authority': response.get('freeze_authority'),
                'permanent_delegate': response.get('permanent_delegate')
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
            
            # Get market cap from Helius
            market_cap = await self.get_token_market_cap(validated_address)
            if market_cap is not None:
                token_info['market_cap'] = market_cap
            
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

    async def _get_quote(self, input_mint: str, output_mint: str, amount: int, slippage_bps: int = 50) -> Optional[dict]:
        """Get quote from Jupiter with configurable slippage."""
        try:
            # Use dynamic delay based on latency history
            delay = await self.get_dynamic_delay()
            await asyncio.sleep(delay)
            
            url = f"{self.jupiter_swap_url}/quote"
            params = {
                "inputMint": input_mint,
                "outputMint": output_mint,
                "amount": str(amount),
                "slippageBps": slippage_bps
            }
            return await self._fetch(url, params=params)
        except Exception as e:
            logger.error(f"Error getting quote: {str(e)}")
            return None

    async def check_liquidity(self, token_address: str) -> dict:
        """Check token liquidity metrics."""
        try:
            # Validate token address
            validated_address = self._validate_address(token_address)
            if not validated_address:
                return {'has_liquidity': False, 'reason': 'Invalid token address'}
            
            quote = await self.get_swap_quote(
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

    async def get_swap_quote(self, input_mint: str, output_mint: str, amount: int) -> Optional[Dict]:
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
            
            quote = await self._fetch(url, params=params)
            if not quote:
                logger.error("Failed to get swap quote")
                return None
                
            logger.info(f"\n{'='*80}\nSwap Quote Response:\n{json.dumps(quote, indent=2)}\n{'='*80}")
            
            return quote
        except Exception as e:
            logger.error(f"Error getting swap quote:\n{str(e)}\n{'='*80}")
            return None

    async def wait_for_confirmation(self, signature: Signature, max_retries: int = 15, retry_delay: float = 1) -> bool:
        """Enhanced confirmation checking with retries."""
        for attempt in range(max_retries):
            try:
                await asyncio.sleep(retry_delay * (attempt + 0.5))
                response = self.client.get_signature_statuses([signature], search_transaction_history=True)
                if not response or not response.value:
                    logger.error("Invalid response from get_signature_statuses")
                    return False
                    
                status = response.value[0]
                
                # If status is None, transaction not found
                if status is None:
                    logger.info("⏳ Transaction submitted, check Solscan for status")
                    return True  # Return true since transaction was sent
                
                # Check for transaction error
                if status.err:
                    error_json = json.dumps(status.err, indent=2) if isinstance(status.err, dict) else str(status.err)
                    logger.error(f"❌ Transaction failed with error: {error_json}")
                    return False
                
                # Check confirmation status
                if status.confirmation_status:
                    if status.confirmation_status in ["confirmed", "finalized"]:
                        logger.info(f"✅ Transaction {status.confirmation_status}")
                        return True
                    elif status.confirmation_status == "processed":
                        logger.info("⏳ Transaction processed, check Solscan for final status")
                        return True
                
                # No confirmation status but transaction exists
                logger.info("⏳ Transaction submitted, check Solscan for status")
                return True
                
            except Exception as e:
                if "commitment" not in str(e):  # Don't log the commitment error
                    logger.warning(f"Confirmation check attempt {attempt+1} failed: {str(e)}")
                    
        logger.error("Max confirmation retries exceeded")
        return False

    async def execute_swap(self, quote: dict) -> Optional[dict]:
        """Optimized swap execution with parallel processing"""
        max_retries = 3
        base_slippage = MAX_SLIPPAGE
        
        for attempt in range(max_retries):
            try:
                if not self.keypair:
                    logger.error("No keypair provided for swap execution")
                    return {"success": False, "error": "No keypair provided", "retries": attempt}
                    
                # Refresh quote with adjusted slippage using parallel RPC
                adjusted_slippage = base_slippage * (1 + (attempt * 0.25))
                fresh_quote = await self._get_updated_quote(quote, adjusted_slippage)
                
                if not fresh_quote:
                    logger.error(f"Failed to refresh quote on attempt {attempt+1}")
                    continue

                # Prepare swap transaction with optimized parameters
                swap_payload = {
                    "quoteResponse": fresh_quote,
                    "userPublicKey": str(self.keypair.pubkey()),
                    "asLegacyTransaction": False,
                    "restrictIntermediateTokens": True,
                    "dynamicComputeUnitLimit": True,
                    "dynamicSlippage": True,
                    "prioritizationFeeLamports": int(PRIORITY_FEE_LAMPORTS * (1 + attempt/10)),
                    "jitoTipLamports": int(JITO_TIP_LAMPORTS * (1 + attempt/10))
                }

                # Get swap transaction data with parallel RPC request
                logger.info("Getting swap transaction data...")
                swap_result = await self._fetch(
                    f"{self.jupiter_swap_url}/swap",
                    method="post",
                    json=swap_payload
                )
                
                if not swap_result:
                    logger.error("Failed to get swap transaction data")
                    continue

                # Process transaction data
                encoded_transaction = swap_result.get('swapTransaction')
                if not encoded_transaction:
                    logger.error("No swap transaction in response")
                    continue

                # Process transaction synchronously since these are CPU-bound operations
                try:
                    # Decode transaction
                    transaction_bytes = base64.b64decode(encoded_transaction)
                    unsigned_tx = VersionedTransaction.from_bytes(transaction_bytes)

                    # Sign transaction
                    message = unsigned_tx.message
                    if isinstance(message, MessageV0):
                        signed_tx = VersionedTransaction(message, [self.keypair])
                        serialized_tx = base64.b64encode(bytes(signed_tx)).decode('utf-8')
                        
                        # Send transaction to multiple RPCs in parallel
                        rpc_request = {
                            "jsonrpc": "2.0",
                            "id": str(int(time.time() * 1000)),
                            "method": "sendTransaction",
                            "params": [
                                serialized_tx,
                                {
                                    "encoding": "base64",
                                    "skipPreflight": True,
                                    "maxRetries": 5,
                                    "minContextSlot": swap_result.get('simulationSlot'),
                                    "lastValidBlockHeight": swap_result.get('lastValidBlockHeight'),
                                    "preflightCommitment": "confirmed"
                                }
                            ]
                        }

                        # Try both RPCs in parallel
                        result = await self._parallel_rpc_request(
                            "sendTransaction",
                            rpc_request["params"]
                        )
                        
                        if not result:
                            logger.error("Failed to send transaction to any RPC")
                            continue
                            
                        response, rpc_type, latency = result

                        if response and "result" in response:
                            tx_sig = response["result"]
                            if not tx_sig:
                                logger.error("No transaction signature in response")
                                continue
                                
                            logger.info(f"Transaction sent via {rpc_type} RPC with latency {latency:.2f}ms")
                            logger.info(f"Signature: {tx_sig}")

                            # Wait for confirmation and process metrics
                            confirmation_result = await self.wait_for_confirmation(Signature.from_string(tx_sig))
                            if not confirmation_result:
                                logger.error("Transaction failed to confirm")
                                continue

                            # Calculate fees and record metrics
                            priority_fee = swap_result.get("prioritizationFeeLamports", PRIORITY_FEE_LAMPORTS)
                            jito_tip = swap_result.get("jitoTipLamports", JITO_TIP_LAMPORTS)
                            
                            # Calculate route fees
                            total_route_fees = await self._calculate_route_fees(quote.get('routePlan', []))
                            
                            # Get SOL price
                            sol_price = await self.get_sol_price()
                            if sol_price is None or sol_price <= 0:
                                sol_price = float(quote['swapUsdValue']) / (float(quote['inAmount']) / 1e9)

                            total_fees_sol = total_route_fees + (priority_fee + jito_tip) / 1e9
                            total_fee_usd = total_fees_sol * sol_price

                            # Log RPC metrics
                            logger.info(f"\n{'='*80}\n🔄 RPC Performance Metrics:\n")
                            logger.info(f"• RPC Provider: {rpc_type}")
                            logger.info(f"• Latency: {latency:.2f}ms")
                            logger.info(f"• Compute Units: {swap_result.get('computeUnitLimit')}")
                            logger.info(f"• Priority Fee: {priority_fee / 1e9:.6f} SOL")
                            logger.info(f"\n{'='*80}")

                            # Record metrics
                            try:
                                metrics = RPCMetrics(
                                    timestamp=datetime.utcnow(),
                                    rpc_type=rpc_type,
                                    latency_ms=latency,
                                    success=True,
                                    tx_signature=tx_sig,
                                    route_count=len(quote.get('routePlan', [])),
                                    slippage_bps=int(float(quote.get('slippageBps', 0))),
                                    compute_units=swap_result.get('computeUnitLimit'),
                                    priority_fee=priority_fee,
                                    final_slippage_bps=int(swap_result.get('dynamicSlippageReport', {}).get('slippageBps', 100)),
                                    total_fee_usd=total_fee_usd,
                                    swap_usd_value=float(fresh_quote.get('swapUsdValue', 0)),
                                    retry_count=attempt,
                                    slippage_adjustment=adjusted_slippage
                                )
                                await self.questdb.record_rpc_metrics(metrics)
                            except Exception as e:
                                logger.error(f"Failed to record success metrics: {str(e)}")
                            
                            return {
                                "success": True,
                                "tx_sig": tx_sig,
                                "retries": attempt,
                                "adjusted_slippage": adjusted_slippage,
                                "dynamic_slippage_report": swap_result.get('dynamicSlippageReport'),
                                "rpc_used": rpc_type,
                                "rpc_latency": latency,
                                "total_fee_usd": total_fee_usd,
                                "compute_units": swap_result.get('computeUnitLimit'),
                                "priority_fee": priority_fee,
                                "jito_tip": jito_tip
                            }
                except Exception as e:
                    logger.error(f"Error processing transaction: {str(e)}")
                    continue
                    
            except Exception as e:
                logger.error(f"Attempt {attempt+1} failed: {str(e)}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(0.1 * (attempt + 1))
                    
        logger.error(f"Transaction failed after {max_retries} attempts")
        return {
            "success": False,
            "error": "Max retries exceeded",
            "retries": max_retries
        }

    async def _calculate_route_fees(self, route_plan: List[dict]) -> float:
        """Calculate total route fees in SOL."""
        total_route_fees = 0
        
        for route in route_plan:
            try:
                fee_amount = float(route['swapInfo']['feeAmount'])
                fee_mint = route['swapInfo']['feeMint']
                
                if fee_mint == "11111111111111111111111111111111":
                    continue
                    
                if fee_mint == "So11111111111111111111111111111111111111112":
                    total_route_fees += fee_amount / 1e9
                    continue
                    
                # For non-SOL fees, get SOL equivalent
                try:
                    reverse_quote = await self._get_quote(
                        input_mint=fee_mint,
                        output_mint="So11111111111111111111111111111111111111112",
                        amount=int(fee_amount)
                    )
                    if reverse_quote and 'outAmount' in reverse_quote:
                        fee_in_sol = float(reverse_quote['outAmount']) / 1e9
                        total_route_fees += fee_in_sol
                except Exception as e:
                    logger.warning(f"Failed to convert fee to SOL: {e}")
                    
            except (KeyError, ValueError, TypeError) as e:
                logger.warning(f"Error calculating route fee: {e}")
                continue
                
        return total_route_fees

    async def _get_updated_quote(self, original_quote: dict, slippage: float) -> Optional[dict]:
        """Refresh quote with updated parameters."""
        try:
            return await self._get_quote(
                input_mint=original_quote['inputMint'],
                output_mint=original_quote['outputMint'],
                amount=int(original_quote['inAmount']),
                slippage_bps=int(slippage * 10000)
            )
        except Exception as e:
            logger.error(f"Error refreshing quote: {str(e)}")
            return None

    async def get_new_tokens(self, limit: int = 100, offset: int = 0) -> Optional[list]:
        """Get list of new tokens from Jupiter API."""
        try:
            url = f"{self.jupiter_token_url}/new"
            logger.info(f"\n{'='*80}\nFetching new tokens\nURL: {url}\nLimit: {limit}, Offset: {offset}\n{'='*80}")
            
            params = {
                "limit": limit,
                "offset": offset
            }
            response = await self._fetch(url, params=params)
            
            if not response:
                logger.error("Failed to get new tokens")
                return None
                
            tokens = response
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
        """Get token information for multiple tokens concurrently."""
        token_info = {}
        
        # Filter out cached tokens and prepare tasks for uncached ones
        uncached_addresses = [addr for addr in token_addresses if addr not in self._token_cache]
        
        if uncached_addresses:
            tasks = []
            for address in uncached_addresses:
                validated_address = self._validate_address(address)
                if not validated_address:
                    continue
                    
                url = f"{self.jupiter_token_url}/token/{validated_address}"
                tasks.append(self._fetch(url))
            
            # Execute all requests concurrently
            responses = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Process responses
            for address, response in zip(uncached_addresses, responses):
                if isinstance(response, Exception):
                    logger.error(f"Error fetching token info for {address}: {str(response)}")
                    continue
                    
                if response:
                    info = {
                        'symbol': response.get('symbol', '...'),
                        'name': response.get('name', 'Unknown Token'),
                        'decimals': response.get('decimals', 6)
                    }
                    token_info[address] = info
                    self._token_cache[address] = info
        
        # Add cached tokens to result
        for address in token_addresses:
            if address in self._token_cache:
                token_info[address] = self._token_cache[address]
            elif address not in token_info:
                token_info[address] = {'symbol': '...', 'name': 'Unknown Token', 'decimals': 6}
        
        return token_info

    async def get_wallet_balance(self) -> float:
        """Get SOL balance of the wallet using Helius RPC"""
        try:
            if not self.keypair:
                return 0.0
            
            payload = {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "getAssetsByOwner",
                "params": {
                    "ownerAddress": str(self.keypair.pubkey()),
                    "page": 1,
                    "limit": 1000,
                    "displayOptions": {
                        "showNativeBalance": True,
                        "showFungible": True
                    }
                }
            }

            response = await self._fetch(HELIUS_RPC_URL, method="post", json=payload)
            if response:
                native_balance = response.get("result", {}).get("nativeBalance", {})
                return float(native_balance.get("lamports", 0)) / 1e9
            return 0.0
            
        except Exception as e:
            logger.error(f"Error getting wallet balance: {str(e)}")
            return 0.0

    async def get_token_holdings(self) -> list:
        """Get token holdings using Helius RPC"""
        try:
            if not self.keypair:
                return []
            
            payload = {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "getAssetsByOwner",
                "params": {
                    "ownerAddress": str(self.keypair.pubkey()),
                    "page": 1,
                    "limit": 1000,
                    "displayOptions": {
                        "showNativeBalance": True,
                        "showFungible": True
                    }
                }
            }

            response = await self._fetch(HELIUS_RPC_URL, method="post", json=payload)
            holdings = []
            
            if response:
                items = response.get("result", {}).get("items", [])
                
                for item in items:
                    # Only process FungibleToken items
                    if item.get("interface") == "FungibleToken":
                        token_info = item.get("token_info", {})
                        if token_info:
                            balance = float(token_info.get("balance", 0))
                            decimals = int(token_info.get("decimals", 6))
                            price_info = token_info.get("price_info", {})
                            
                            if balance > 0:  # Only include non-zero balances
                                holdings.append({
                                    "symbol": token_info.get("symbol", "Unknown"),
                                    "amount": balance / (10 ** decimals),
                                    "value_usd": float(price_info.get("total_price", 0))
                                })
            
            return holdings
            
        except Exception as e:
            logger.error(f"Error getting token holdings: {str(e)}")
            return []

    async def get_sol_price(self) -> float:
        """Get current SOL price in USD using Jupiter's price API with fallback to CoinGecko."""
        try:
            # First attempt: Use Jupiter's price API
            url = f"{self.jupiter_price_url}/price"
            params = {'ids': self.sol_mint}
            response = await self._fetch(url, params=params)
            
            if response and 'data' in response and 'price' in response['data']:
                return float(response['data']['price'])
            
            # Second attempt: Calculate from a 1 SOL to USDC quote
            usdc_mint = "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v"
            quote = await self._get_quote(
                input_mint=self.sol_mint,
                output_mint=usdc_mint,
                amount=1_000_000_000  # 1 SOL in lamports
            )
            
            if quote and 'outAmount' in quote:
                return float(quote['outAmount']) / 1_000_000  # USDC has 6 decimals
            
            # Third attempt: Use CoinGecko service
            coingecko = CoinGeckoAPI()
            price = await coingecko.get_sol_price()
            if price is not None:
                return price
            
            logger.warning("Failed to get SOL price from all sources")
            return 0.0
            
        except Exception as e:
            logger.error(f"Error getting SOL price: {str(e)}")
            return 0.0 

    async def get_token_market_cap(self, token_address: str) -> Optional[float]:
        """Get token market cap from Helius API."""
        try:
            validated_address = self._validate_address(token_address)
            if not validated_address:
                return None

            payload = {
                "jsonrpc": "2.0",
                "id": "jup-token-info",
                "method": "getAsset",
                "params": {
                    "id": validated_address,
                    "displayOptions": {
                        "showFungible": True
                    }
                }
            }

            response = await self._fetch(HELIUS_RPC_URL, method="post", json=payload)
            if response:
                result = response.get("result", {})
                token_info = result.get("token_info", {})
                
                # Get total supply and price info
                total_supply = token_info.get("supply")
                price_info = token_info.get("price_info", {})
                price_per_token = price_info.get("price_per_token")
                
                if total_supply is not None and price_per_token is not None:
                    # Convert supply from base units to actual tokens using decimals
                    decimals = token_info.get("decimals", 6)
                    actual_supply = float(total_supply) / (10 ** decimals)
                    
                    # Calculate market cap
                    market_cap = actual_supply * float(price_per_token)
                    return market_cap
                    
            return None
            
        except Exception as e:
            logger.error(f"Error getting token market cap: {str(e)}")
            return None 

    async def cleanup(self):
        """Cleanup resources"""
        try:
            # Close thread pool
            if hasattr(self, '_thread_pool'):
                self._thread_pool.shutdown(wait=True)
            
            # Close all sessions
            if hasattr(self, '_session_pool'):
                for endpoint, session in self._session_pool.items():
                    if not session.closed:
                        try:
                            await session.close()
                            logger.info(f"Closed session for endpoint: {endpoint}")
                        except Exception as e:
                            logger.error(f"Error closing session for {endpoint}: {str(e)}")
                self._session_pool.clear()
            
            # Close individual session if exists
            if hasattr(self, '_session') and self._session and not self._session.closed:
                await self._session.close()
                logger.info("Closed individual session")
                
            # Clear latency history
            if hasattr(self, '_latency_history'):
                self._latency_history.clear()
                
        except Exception as e:
            logger.error(f"Error during cleanup: {str(e)}")
        finally:
            # Ensure thread pool is shut down
            if hasattr(self, '_thread_pool') and not self._thread_pool._shutdown:
                self._thread_pool.shutdown(wait=False) 
