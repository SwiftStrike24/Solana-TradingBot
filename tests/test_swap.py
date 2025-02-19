#!/usr/bin/env python3
import os
import sys
import asyncio
from datetime import datetime
from rich.console import Console
from rich.panel import Panel
from solders.keypair import Keypair
import base64
from dotenv import load_dotenv
import base58
from solders.signature import Signature

# Add project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.trading import TradingBot
from src.utils.helpers import setup_logging
from tests.test_token import format_token_info, format_swap_quote
from src.services.coingecko import CoinGeckoAPI
from src.config.settings import PRIORITY_FEE_LAMPORTS, JITO_TIP_LAMPORTS

console = Console()
load_dotenv()

def format_wallet_info(balance: float, token_holdings: list, sol_price: float = 0) -> Panel:
    """Format wallet information into a rich panel"""
    sol_value = balance * sol_price if sol_price else 0
    
    details = [
        f"[cyan]SOL Balance:[/cyan] {balance:.4f} SOL (${sol_value:.2f})",
        "",
        "[yellow]Token Holdings:[/yellow]"
    ]
    
    for token in token_holdings:
        details.append(f"• {token['symbol']}: {token['amount']} (${token['value_usd']:.2f})")
    
    return Panel(
        "\n".join(details),
        title="[bold green]Wallet Information[/bold green]",
        border_style="green"
    )

async def interactive_swap_test():
    timestamp = datetime.now().strftime("%Y-%m-%d_%I-%M-%S_%p")
    logger = setup_logging(f"logs/test_swap_{timestamp}.log")
    
    # Get private key from environment
    private_key_b58 = os.getenv("SOLANA_WALLET_PRIVATE_KEY")
    if not private_key_b58:
        console.print("[red]Error: SOLANA_WALLET_PRIVATE_KEY not found in environment[/red]")
        return
        
    try:
        keypair = Keypair.from_base58_string(private_key_b58)
        bot = TradingBot(keypair)
        coingecko = CoinGeckoAPI()
        
        console.print("\n[bold cyan]🚀 Solana Swap Tester[/bold cyan]\n")
        
        # Get SOL price and display it prominently
        sol_price = await coingecko.get_sol_price()
        console.print(Panel(
            f"✨ [bold magenta]1 SOL[/bold magenta] = [bold white]${sol_price:.2f}[/bold white] [dim]USD[/dim] ✨",
            title="[bold purple]💫 SOLANA LIVE PRICE 💫[/bold purple]",
            border_style="magenta",
            padding=(1, 4),
            style="purple"
        ))
        # Get wallet info
        balance = await bot.get_wallet_balance()
        token_holdings = await bot.get_token_holdings()
        
        console.print(format_wallet_info(balance, token_holdings, sol_price))
        
        while True:
            choice = console.input("\n[bold green]Choose action ([cyan]1[/cyan]: Test $1 Swap, [cyan]q[/cyan]: Quit):[/bold green] ").strip()
            
            if choice.lower() == 'q':
                break
                
            if choice == '1':
                token_address = console.input("[bold green]Enter Token Address to swap to:[/bold green] ").strip()
                
                # Get token info and display
                price_data = await bot.get_token_price(token_address)
                liquidity = bot.check_liquidity(token_address)
                console.print(format_token_info(token_address, price_data, liquidity))
                
                if liquidity['has_liquidity']:
                    # Calculate amount of SOL for $1
                    if not sol_price or sol_price == 0:
                        console.print("[red]Error: Unable to get valid SOL price[/red]")
                        continue
                        
                    sol_amount = int((1.0 / sol_price) * 1e9)  # Convert to lamports
                    
                    # Get swap quote
                    quote = bot.get_swap_quote(
                        input_mint="So11111111111111111111111111111111111111112",
                        output_mint=token_address,
                        amount=sol_amount
                    )
                    
                    if quote:
                        # Get all token info for the route
                        token_addresses = set()
                        for route in quote['routePlan']:
                            swap = route['swapInfo']
                            token_addresses.add(swap['inputMint'])
                            token_addresses.add(swap['outputMint'])
                            if swap['feeMint'] != "11111111111111111111111111111111":
                                token_addresses.add(swap['feeMint'])
                        
                        all_tokens_info = await bot.get_token_info_batch(list(token_addresses))
                        console.print(format_swap_quote(quote, price_data.get('token_info'), all_tokens_info))
                        
                        # Calculate and display fee breakdown
                        total_route_fees = sum(float(route['swapInfo']['feeAmount']) for route in quote['routePlan'])
                        priority_fee_sol = PRIORITY_FEE_LAMPORTS / 1e9
                        jito_tip_sol = JITO_TIP_LAMPORTS / 1e9
                        total_fees_sol = (total_route_fees + PRIORITY_FEE_LAMPORTS + JITO_TIP_LAMPORTS) / 1e9
                        
                        # Get current SOL price for USD conversion
                        sol_price = await coingecko.get_sol_price()
                        
                        fee_breakdown = (
                            f"[bold white]💰 Transaction Fee Breakdown[/bold white]\n\n"
                            f"[cyan]Network Fees:[/cyan]\n"
                            f"├─ 🏃 Priority Fee: [green]{priority_fee_sol:.6f} SOL[/green] (${priority_fee_sol * sol_price:.4f})\n"
                            f"├─ ⚡ Jito MEV Tip: [green]{jito_tip_sol:.6f} SOL[/green] (${jito_tip_sol * sol_price:.4f})\n"
                            f"└─ 🔄 Route Fees: [green]{total_route_fees/1e9:.6f} SOL[/green] (${total_route_fees/1e9 * sol_price:.4f})\n\n"
                            f"[bold cyan]Route Details:[/bold cyan]"
                        )
                        
                        for idx, route in enumerate(quote['routePlan'], 1):
                            swap = route['swapInfo']
                            fee_amount = float(swap['feeAmount'])
                            fee_token = all_tokens_info.get(swap['feeMint'], {'symbol': 'Unknown'})
                            
                            # Get correct decimals for fee token
                            if swap['feeMint'] == "11111111111111111111111111111111" or \
                               swap['feeMint'] == "So11111111111111111111111111111111111111112":
                                fee_decimals = 9  # SOL decimals
                            else:
                                fee_decimals = fee_token.get('decimals', 6)
                            
                            # Calculate actual fee amount with correct decimals
                            actual_fee = fee_amount / (10 ** fee_decimals)
                            
                            fee_breakdown += (
                                f"\n{'└─' if idx == len(quote['routePlan']) else '├─'} [yellow]Step {idx}[/yellow] "
                                f"via [blue]{swap['label']}[/blue]\n"
                                f"   {'  ' if idx == len(quote['routePlan']) else '│ '} Fee: [green]{actual_fee:.6f}[/green] "
                                f"{fee_token['symbol']}"
                            )
                        
                        fee_breakdown += (
                            f"\n\n[bold white]Total Cost:[/bold white]\n"
                            f"[bold green]{total_fees_sol:.6f} SOL[/bold green] "
                            f"([bold cyan]${total_fees_sol * sol_price:.4f}[/bold cyan])\n\n"
                            f"[dim]• Priority Fee (70%) helps transaction land faster[/dim]\n"
                            f"[dim]• Jito Tip (30%) provides MEV protection[/dim]\n"
                            f"[dim]• Route fees are paid to liquidity providers[/dim]"
                        )
                        
                        console.print(Panel(
                            fee_breakdown,
                            title="[bold yellow]🔥 Transaction Costs & Fees[/bold yellow]",
                            border_style="yellow",
                            padding=(1, 2),
                            highlight=True
                        ))

                        if console.input("\n[bold yellow]Execute swap? (y/n):[/bold yellow] ").lower() == 'y':
                            console.print("\n[bold cyan]Executing swap...[/bold cyan]")
                            tx_result = await bot.execute_swap(quote)
                            if tx_result:
                                tx_sig = tx_result.get("tx_sig")
                                dynamic_report = tx_result.get("dynamic_slippage_report")
                                jito_bundle_id = tx_result.get("jito_bundle_id")
                                rpc_used = tx_result.get("rpc_used", "Unknown")
                                
                                # Create RPC indicator based on which one was used
                                rpc_indicator = (
                                    "[bold yellow]⚡ JITO RPC[/bold yellow] (MEV Protected)" if rpc_used == "jito" else
                                    "[bold cyan]🚀 HELIUS RPC[/bold cyan] (High Performance)" if rpc_used == "helius" else
                                    "[bold red]❓ UNKNOWN RPC[/bold red]"
                                )
                                
                                # Create transaction details panel with RPC info
                                tx_details = [
                                    "[green]Transaction sent successfully![/green]\n",
                                    f"{rpc_indicator}",
                                    f"[dim]Transaction routed through fastest available RPC[/dim]\n",
                                    f"[white]Signature:[/white] {tx_sig}",
                                    f"[white]View on Solscan:[/white] https://solscan.io/tx/{tx_sig}"
                                ]
                                
                                # Add Jito-specific info if available
                                if jito_bundle_id:
                                    tx_details.extend([
                                        "",
                                        "[cyan]Jito MEV Protection Details:[/cyan]",
                                        f"[white]Bundle ID:[/white] {jito_bundle_id}",
                                        "[dim]Transaction bundled for frontrunning protection[/dim]"
                                    ])
                                
                                # Add performance metrics if available
                                if "rpc_latency" in tx_result:
                                    latency = tx_result.get("rpc_latency")
                                    tx_details.extend([
                                        "",
                                        "[magenta]Performance Metrics:[/magenta]",
                                        f"[white]RPC Latency:[/white] {latency:.2f}ms"
                                    ])
                                
                                console.print(Panel(
                                    "\n".join(tx_details),
                                    title="[bold green]Transaction Details[/bold green]",
                                    border_style="green",
                                    padding=(1, 2)
                                ))
                                if dynamic_report:
                                    final_slippage = float(dynamic_report.get("slippageBps", 0)) / 100.0
                                    simulated_slippage = float(dynamic_report.get("simulatedIncurredSlippageBps", 0)) / 100.0
                                    max_expected_slippage = float(dynamic_report.get("heuristicMaxSlippageBps", 0)) / 100.0
                                    amplification_ratio = dynamic_report.get("amplificationRatio", "N/A")
                                    token_category = dynamic_report.get("categoryName", "").capitalize()
                                    if token_category.lower() == "bluechip":
                                        token_category += " (High Trust)"
                                    slippage_panel = (
                                        f"🔄 Slippage Analysis:\n"
                                        f"   - 📉 Final Slippage: {final_slippage:.2f}%\n"
                                        f"   - ✅ Actual Slippage Lower Than Expected (Saved {abs(simulated_slippage):.2f}%)\n" 
                                        f"   - 🔸 Max Expected Slippage: {max_expected_slippage:.2f}%\n"
                                        f"   - 🔵 Token Category: {token_category}\n"
                                        f"   - 📊 Amplification Ratio: {amplification_ratio}x"
                                    )
                                    console.print(Panel(slippage_panel, title="[bold blue]Dynamic Slippage Analysis[/bold blue]", border_style="blue"))
                                console.print("\n[bold yellow]Checking initial transaction status...[/bold yellow]")
                                await bot.wait_for_confirmation(Signature.from_string(tx_sig))
                                console.print("\n[cyan]Waiting for transaction to settle...[/cyan]")
                                await asyncio.sleep(2)  # Short delay for transaction to settle
                                console.print("[cyan]Fetching updated wallet information...[/cyan]")
                                new_balance = await bot.get_wallet_balance()
                                new_holdings = await bot.get_token_holdings()
                                sol_price = await coingecko.get_sol_price()
                                console.print(Panel(
                                    format_wallet_info(new_balance, new_holdings, sol_price).renderable,
                                    title="[bold green]Updated Wallet Status[/bold green]",
                                    border_style="green"
                                ))

                                # Get the swap amounts from the quote
                                input_amount = float(quote['inAmount']) / 1e9  # SOL is always 9 decimals
                                output_decimals = price_data['token_info']['decimals']
                                output_amount = float(quote['outAmount']) / (10 ** output_decimals)
                                input_token = "SOL"
                                output_token = price_data['token_info']['symbol']
                                usd_value = float(quote['swapUsdValue'])

                                # Dynamic number formatting based on value
                                def format_token_amount(amount, token_symbol):
                                    if token_symbol == "SOL":
                                        if amount >= 1:
                                            return f"{amount:.4f}"   # 1.2345 SOL
                                        else:
                                            return f"{amount:.4f}"   # 0.0059 SOL
                                    else:
                                        if amount >= 1000:
                                            return f"{amount:,.2f}"  # 1,234.56
                                        elif amount >= 1:
                                            return f"{amount:.4f}"   # 12.3456
                                        elif amount >= 0.000001:     # Show up to 9 decimals for small values
                                            decimal_places = min(9, abs(len(str(amount).split('.')[-1])))
                                            return f"{amount:.{decimal_places}f}"
                                        else:
                                            # For extremely small values (< 0.000001), use scientific notation
                                            return f"{amount:.2e}"

                                # Format amounts
                                formatted_input = format_token_amount(input_amount, input_token)
                                formatted_output = format_token_amount(output_amount, output_token)

                                # Format USD with full decimals for small values
                                def format_usd_value(value):
                                    if value >= 0.01:
                                        return f"{value:,.2f}"
                                    elif value >= 0.000001:
                                        decimal_places = min(9, abs(len(str(value).split('.')[-1])))
                                        return f"{value:.{decimal_places}f}"
                                    else:
                                        return f"{value:.2e}"

                                formatted_usd = format_usd_value(usd_value)

                                swap_complete_panel = (
                                    f"[green]Swap transaction submitted successfully![/green]\n\n"
                                    f"[bold white]🔄 Swap Summary:[/bold white]\n"
                                    f"[cyan]FROM:[/cyan] {formatted_input} {input_token} 💫\n"
                                    f"[cyan]TO:[/cyan] {formatted_output} {output_token} ✨\n"
                                    f"[dim white]Value:[/dim white] ${formatted_usd} USD\n\n"
                                    f"[dim]Check Solscan for final confirmation status.[/dim]"
                                )
                                console.print(Panel(
                                    swap_complete_panel,
                                    title="[bold green]🎉 Swap Complete[/bold green]",
                                    border_style="green",
                                    padding=(1, 2)
                                ))
                            else:
                                console.print(Panel(
                                    "[red]Failed to execute swap. Please try again.[/red]",
                                    title="[bold red]❌ Swap Failed[/bold red]",
                                    border_style="red"
                                ))
                    else:
                        console.print("[red]Failed to get swap quote[/red]")
                else:
                    console.print("[red]Token does not have sufficient liquidity[/red]")
            
            console.print("\n[green]Test complete![/green]\n")
            
    except Exception as e:
        console.print(f"[red]Error: {str(e)}[/red]")
        logger.error(f"Error during swap test: {str(e)}")

if __name__ == "__main__":
    asyncio.run(interactive_swap_test())