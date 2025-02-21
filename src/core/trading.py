import logging
from typing import Dict, Optional
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

logger = logging.getLogger(__name__)

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
                    'daily_volume': None,  # Could fetch from another source if needed
                    'logo_uri': 'https://raw.githubusercontent.com/solana-labs/token-list/main/assets/mainnet/So11111111111111111111111111111111111111112/logo.png',
                    'verified': True,
                    'created_at': None,
                    'mint_authority': None,
                    'freeze_authority': None,
                    'permanent_delegate': None
                }

            url = f"{self.jupiter_token_url}/token/{validated_address}"
            logger.info(f"\n{'='*80}\nFetching token info for: {validated_address}\nURL: {url}\n{'='*80}")
            
            response = requests.get(url)
            
            if not response.ok:
                logger.warning(f"Failed to get token info: Status {response.status_code}\nResponse: {response.text}")
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
            await asyncio.sleep(self.request_delay)
            url = f"{self.jupiter_swap_url}/quote"
            params = {
                "inputMint": input_mint,
                "outputMint": output_mint,
                "amount": str(amount),
                "slippageBps": slippage_bps
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
        """Execute a swap transaction with retry logic and dynamic parameter adjustment."""
        max_retries = 3
        base_slippage = MAX_SLIPPAGE
        retry_delay = 0.5  # seconds
        
        for attempt in range(max_retries):
            try:
                if not self.keypair:
                    logger.error("No keypair provided for swap execution")
                    return None
                    
                # Refresh quote with adjusted slippage before each attempt
                adjusted_slippage = base_slippage * (1 + (attempt * 0.25))  # 25% increase per attempt
                fresh_quote = await self._get_updated_quote(quote, adjusted_slippage)
                
                if not fresh_quote:
                    logger.error(f"Failed to refresh quote on attempt {attempt+1}")
                    continue

                swap_url = f"{self.jupiter_swap_url}/swap"
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

                logger.info(f"Swap attempt {attempt+1} with params: {json.dumps(swap_payload, indent=2)}")
                
                response = requests.post(swap_url, json=swap_payload)
                if not response.ok:
                    logger.error(f"Failed to get swap transaction: {response.text}")
                    continue
                
                swap_data = response.json()
                logger.info(f"Received swap response: {json.dumps(swap_data, indent=2)}")

                # Check for simulation errors before proceeding
                simulation_error = swap_data.get("simulationError")
                if simulation_error:
                    error_code = simulation_error.get("errorCode")
                    error_msg = simulation_error.get("error")
                    logger.error(f"Simulation failed on attempt {attempt+1}: {error_code} - {error_msg}")
                    
                    # Record failed attempt in QuestDB
                    metrics = RPCMetrics(
                        timestamp=datetime.utcnow(),
                        rpc_type="simulation",
                        latency_ms=0,
                        success=False,
                        tx_signature=None,
                        route_count=len(quote.get('routePlan', [])),
                        slippage_bps=int(adjusted_slippage * 10000),
                        compute_units=swap_data.get('computeUnitLimit'),
                        priority_fee=swap_payload['prioritizationFeeLamports'],
                        final_slippage_bps=None,
                        total_fee_usd=0,
                        swap_usd_value=float(fresh_quote.get('swapUsdValue', 0)),
                        retry_count=attempt,
                        slippage_adjustment=adjusted_slippage
                    )
                    self.questdb.record_rpc_metrics(metrics)
                    
                    # If this is the last attempt, return failure details
                    if attempt == max_retries - 1:
                        return {
                            "success": False,
                            "error": error_msg,
                            "error_code": error_code,
                            "retries": attempt + 1
                        }
                    continue

                dynamic_slippage_report = swap_data.get("dynamicSlippageReport")
                encoded_transaction = swap_data.get('swapTransaction')
                if not encoded_transaction:
                    logger.error("No swap transaction in response")
                    continue

                transaction_bytes = base64.b64decode(encoded_transaction)
                unsigned_tx = VersionedTransaction.from_bytes(transaction_bytes)
                
                last_valid_block_height = swap_data.get('lastValidBlockHeight')
                if not last_valid_block_height:
                    logger.error("No lastValidBlockHeight in response")
                    continue
                
                message = unsigned_tx.message
                if isinstance(message, MessageV0):
                    signed_tx = VersionedTransaction(
                        message,
                        [self.keypair]
                    )
                    
                    serialized_tx = base64.b64encode(bytes(signed_tx)).decode('utf-8')
                    
                    rpc_request = {
                        "jsonrpc": "2.0",
                        "id": "1",
                        "method": "sendTransaction",
                        "params": [
                            serialized_tx,
                            {
                                "encoding": "base64",
                                "skipPreflight": True,
                                "maxRetries": 5,
                                "minContextSlot": swap_data.get('simulationSlot'),
                                "lastValidBlockHeight": last_valid_block_height,
                                "preflightCommitment": "confirmed"
                            }
                        ]
                    }
                    
                    logger.info(f"Sending transaction concurrently to both RPCs")
                    
                    const_jito_url = f"{JITO_RPC_URL}/api/v1/transactions"
                    jito_headers = {
                        "Content-Type": "application/json",
                        "User-Agent": "jito-client",
                        "X-API-Version": "1"
                    }
                    helius_url = HELIUS_RPC_URL
                    helius_headers = { "Content-Type": "application/json" }

                    async def send_transaction(url: str, headers: dict) -> Optional[tuple[requests.Response, str, float]]:
                        try:
                            start_time = time.perf_counter()
                            loop = asyncio.get_running_loop()
                            response = await loop.run_in_executor(
                                None,
                                lambda: requests.post(url, json=rpc_request, headers=headers)
                            )
                            end_time = time.perf_counter()
                            latency = (end_time - start_time) * 1000  # Convert to milliseconds
                            
                            if response and response.ok and 'result' in response.json():
                                rpc_type = "jito" if url.startswith(JITO_RPC_URL) else "helius"
                                return response, rpc_type, latency
                        except Exception as e:
                            logger.error(f"Error sending transaction to {url}: {str(e)}")
                        return None

                    # Create tasks for both RPCs
                    jito_task = asyncio.create_task(send_transaction(const_jito_url, jito_headers))
                    helius_task = asyncio.create_task(send_transaction(helius_url, helius_headers))
                    
                    # Wait for first successful response
                    done, pending = await asyncio.wait(
                        {jito_task, helius_task},
                        return_when=asyncio.FIRST_COMPLETED
                    )

                    valid_response = None
                    rpc_type = None
                    latency = None

                    for task in done:
                        try:
                            result = await task
                            if result:
                                valid_response, rpc_type, latency = result
                                break
                        except Exception as e:
                            logger.error(f"Task error: {str(e)}")

                    # Cancel pending tasks
                    for task in pending:
                        task.cancel()
                        try:
                            await task
                        except asyncio.CancelledError:
                            pass
                    
                    if valid_response:
                        tx_sig = valid_response.json()["result"]
                        request_url = valid_response.request.url if valid_response.request else ""
                        jito_bundle_id = valid_response.headers.get("x-bundle-id") if request_url.startswith(JITO_RPC_URL) else None
                        logger.info("🚀 Transaction Details:")
                        logger.info(f"✅ Transaction sent successfully: {tx_sig}")
                        if jito_bundle_id:
                            logger.info(f"📦 Bundle ID: {jito_bundle_id}")

                        try:
                            # Wait for transaction confirmation before calculating fees
                            confirmation_result = await self.wait_for_confirmation(Signature.from_string(tx_sig))
                            if not confirmation_result:
                                logger.error("Transaction failed to confirm")
                                if attempt < max_retries - 1:
                                    continue
                                return {
                                    "success": False,
                                    "error": "Transaction failed to confirm",
                                    "tx_sig": tx_sig,
                                    "retries": attempt + 1
                                }

                            # Get actual fees from swap_data with fallbacks
                            priority_fee_lamports = swap_data.get("prioritizationFeeLamports", PRIORITY_FEE_LAMPORTS)
                            jito_tip_lamports = swap_data.get("jitoTipLamports", JITO_TIP_LAMPORTS)
                            
                            # Calculate total route fees with safety checks
                            total_route_fees = 0
                            for route in quote.get('routePlan', []):
                                try:
                                    fee_amount = float(route['swapInfo']['feeAmount'])
                                    fee_mint = route['swapInfo']['feeMint']
                                    
                                    # Skip system program fees (all zeros)
                                    if fee_mint == "11111111111111111111111111111111":
                                        continue
                                        
                                    # For SOL fees
                                    if fee_mint == "So11111111111111111111111111111111111111112":
                                        total_route_fees += fee_amount / 1e9
                                        continue
                                        
                                    # For non-SOL fees, we need to get their SOL equivalent
                                    try:
                                        # Get quote to convert fee token to SOL
                                        reverse_quote = await self._get_quote(
                                            input_mint=fee_mint,
                                            output_mint="So11111111111111111111111111111111111111112",
                                            amount=int(fee_amount)  # Use original amount in base units
                                        )
                                        if reverse_quote and 'outAmount' in reverse_quote:
                                            fee_in_sol = float(reverse_quote['outAmount']) / 1e9
                                            total_route_fees += fee_in_sol
                                    except Exception as e:
                                        logger.warning(f"Failed to convert fee to SOL: {e}")
                                        
                                except (KeyError, ValueError, TypeError) as e:
                                    logger.warning(f"Error calculating route fee: {e}")
                                    continue

                            # Calculate total fees in SOL
                            total_fees_sol = (total_route_fees + (priority_fee_lamports + jito_tip_lamports) / 1e9)

                            # Get SOL price with fallback
                            try:
                                sol_price = await self.get_sol_price()
                                if sol_price is None or sol_price <= 0:
                                    logger.warning("Invalid SOL price, using fallback from quote")
                                    # Try to derive price from quote's USD value
                                    if quote.get('swapUsdValue') and quote.get('inAmount'):
                                        sol_price = float(quote['swapUsdValue']) / (float(quote['inAmount']) / 1e9)
                                    else:
                                        sol_price = 0
                            except Exception as e:
                                logger.warning(f"Error getting SOL price: {e}")
                                sol_price = 0

                            # Calculate total fee in USD with safety check
                            total_fee_usd = total_fees_sol * sol_price if sol_price > 0 else 0

                            # Log the fee calculation for debugging
                            logger.info(f"Fee Calculation Details:")
                            logger.info(f"- Total Route Fees (SOL): {total_route_fees}")
                            logger.info(f"- Priority Fee (SOL): {priority_fee_lamports/1e9}")
                            logger.info(f"- Jito Tip (SOL): {jito_tip_lamports/1e9}")
                            logger.info(f"- Total Fees (SOL): {total_fees_sol}")
                            logger.info(f"- SOL Price: ${sol_price}")
                            logger.info(f"- Total Fees (USD): ${total_fee_usd}")

                            # Create metrics object with safe values
                            metrics = RPCMetrics(
                                timestamp=datetime.utcnow(),
                                rpc_type=rpc_type,
                                latency_ms=latency or 0,
                                success=True,
                                tx_signature=tx_sig,
                                route_count=len(quote.get('routePlan', [])),
                                slippage_bps=int(float(quote.get('slippageBps', 0))),
                                compute_units=swap_data.get('computeUnitLimit'),
                                priority_fee=priority_fee_lamports,
                                final_slippage_bps=int(swap_data.get('dynamicSlippageReport', {}).get('slippageBps', 100)),
                                total_fee_usd=total_fee_usd,
                                swap_usd_value=float(fresh_quote.get('swapUsdValue', 0)),
                                retry_count=attempt,
                                slippage_adjustment=adjusted_slippage
                            )

                            self.questdb.record_rpc_metrics(metrics)
                            
                            return {
                                "success": True,
                                "tx_sig": tx_sig,
                                "retries": attempt,
                                "adjusted_slippage": adjusted_slippage,
                                "dynamic_slippage_report": dynamic_slippage_report,
                                "jito_bundle_id": jito_bundle_id,
                                "rpc_used": rpc_type,
                                "rpc_latency": latency,
                                "total_fee_usd": total_fee_usd
                            }

                        except Exception as e:
                            logger.error(f"Error processing fees and metrics: {e}")
                            # Record failed attempt in QuestDB
                            metrics = RPCMetrics(
                                timestamp=datetime.utcnow(),
                                rpc_type=rpc_type or "unknown",
                                latency_ms=latency or 0,
                                success=False,
                                tx_signature=tx_sig,
                                route_count=len(quote.get('routePlan', [])),
                                slippage_bps=int(adjusted_slippage * 10000),
                                compute_units=swap_data.get('computeUnitLimit'),
                                priority_fee=priority_fee_lamports,
                                final_slippage_bps=None,
                                total_fee_usd=0,
                                swap_usd_value=float(fresh_quote.get('swapUsdValue', 0)),
                                retry_count=attempt,
                                slippage_adjustment=adjusted_slippage
                            )
                            self.questdb.record_rpc_metrics(metrics)
                            
                            if attempt < max_retries - 1:
                                continue
                            return {
                                "success": False,
                                "error": str(e),
                                "tx_sig": tx_sig,
                                "retries": attempt + 1
                            }

                else:
                    logger.error("Unexpected transaction message format")
                    return None
                
            except Exception as e:
                logger.error(f"Attempt {attempt+1} failed: {str(e)}")
                if attempt < max_retries - 1:
                    logger.info(f"Retrying in {retry_delay}s...")
                    await asyncio.sleep(retry_delay * (attempt + 1))
                    
        logger.error(f"Transaction failed after {max_retries} attempts")
        return {
            "success": False,
            "error": "Max retries exceeded",
            "retries": max_retries
        }

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

            response = requests.post(
                HELIUS_RPC_URL,
                json=payload,
                headers={"Content-Type": "application/json"}
            )

            if response.ok:
                data = response.json()
                native_balance = data.get("result", {}).get("nativeBalance", {})
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

            response = requests.post(
                HELIUS_RPC_URL,
                json=payload,
                headers={"Content-Type": "application/json"}
            )

            holdings = []
            if response.ok:
                data = response.json()
                items = data.get("result", {}).get("items", [])
                
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
            response = requests.get(url, params=params)
            
            if response.ok:
                data = response.json()
                if 'data' in data and 'price' in data['data']:
                    return float(data['data']['price'])
            
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

            response = requests.post(
                HELIUS_RPC_URL,
                json=payload,
                headers={"Content-Type": "application/json"}
            )

            if response.ok:
                data = response.json()
                result = data.get("result", {})
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
