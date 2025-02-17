#!/usr/bin/env python3
import os
import sys
import asyncio
from datetime import datetime
import json
from rich.console import Console, Group
from rich.panel import Panel
from rich.table import Table
from rich.box import DOUBLE_EDGE
from typing import Optional
from solders.pubkey import Pubkey

# Add project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.trading import TradingBot
from src.utils.helpers import setup_logging, get_solana_client, is_valid_token_address

console = Console()

def format_token_info(token_address: str, price_data: dict, liquidity: dict) -> Panel:
    """Format token information into a rich panel"""
    if not price_data:
        details = [
            f"[cyan]Token Address:[/cyan] {token_address}",
            "[red]Price: Unable to fetch[/red]",
            f"[red]Liquidity: {liquidity.get('reason', 'Insufficient')}[/red]"
        ]
    else:
        # Get token info
        token_info = price_data.get('token_info', {})
        symbol = token_info.get('symbol', 'Unknown')
        name = token_info.get('name', 'Unknown Token')
        verified = token_info.get('verified', False)
        daily_volume = token_info.get('daily_volume', 0)
        
        # Format prices with appropriate precision
        tokens_per_sol = price_data['tokens_per_sol']
        sol_per_token = price_data['sol_per_token']
        
        # Dynamic precision based on value magnitude
        def get_precision(value: float) -> int:
            if value == 0:
                return 6
            abs_value = abs(value)
            if abs_value >= 1000:
                return 2
            elif abs_value >= 1:
                return 4
            elif abs_value >= 0.0001:
                return 6
            else:
                return 9
        
        token_precision = get_precision(tokens_per_sol)
        sol_precision = get_precision(sol_per_token)
        
        verification_status = "[green]✓ Verified[/green]" if verified else "[yellow]⚠ Unverified[/yellow]"
        
        details = [
            f"[cyan]Token:[/cyan] {name} ({symbol}) {verification_status}",
            f"[cyan]Address:[/cyan] {token_address}",
            f"[magenta]Daily Volume:[/magenta] {daily_volume:,.2f} USD",
            "",
            f"[green]1 SOL = {tokens_per_sol:.{token_precision}f} {symbol}[/green]",
            f"[green]1 {symbol} = {sol_per_token:.{sol_precision}f} SOL[/green]",
            f"[magenta]Price Impact:[/magenta] {price_data['price_impact']}%",
            f"[{'green' if liquidity['has_liquidity'] else 'red'}]Liquidity Analysis:[/]",
            f"  • Impact: {liquidity['price_impact']:.2f}%",
            f"  • Routes: {liquidity['routes_available']}",
            f"  • Fees: {liquidity['total_fees']/1e9:.6f} SOL"
        ]
    return Panel(
        "\n".join(details),
        title="[bold cyan]Token Information[/bold cyan]",
        border_style="cyan"
    )

def format_swap_quote(quote: dict) -> Panel:
    """Format swap quote into a rich panel"""
    if not quote:
        return Panel("No quote available", title="Swap Quote", border_style="red")

    quote_details = [
        f"💰 [cyan]Input Amount:[/cyan] {float(quote['inAmount'])/1e9:.4f} SOL",
        f"💎 [cyan]Output Amount:[/cyan] {float(quote['outAmount'])/1e9:.9f} tokens",
        f"🛡️ [yellow]Slippage:[/yellow] {quote['slippageBps']/100}%",
        f"📊 [magenta]Price Impact:[/magenta] {quote.get('priceImpactPct', '0')}%"
    ]

    route_table = Table(
        title="🔄 Route Details",
        show_header=True,
        header_style="bold cyan",
        box=DOUBLE_EDGE
    )
    
    route_table.add_column("Step", justify="center", style="cyan")
    route_table.add_column("DEX", style="magenta")
    route_table.add_column("Fee", justify="right", style="red")

    for idx, route in enumerate(quote['routePlan'], 1):
        swap = route['swapInfo']
        route_table.add_row(
            f"#{idx}",
            f"🏦 {swap['label']}",
            f"🔸 {float(swap['feeAmount'])/1e9:.6f} SOL"
        )

    return Panel(
        Group(
            "\n".join(quote_details),
            route_table
        ),
        title="[bold yellow]Swap Quote Details[/bold yellow]",
        border_style="yellow"
    )

async def interactive_token_test():
    timestamp = datetime.now().strftime("%Y-%m-%d_%I-%M-%S_%p")
    logger = setup_logging(f"logs/test_token_{timestamp}.log")
    
    console.print("\n[bold cyan]🚀 Starting Solana Token Analyzer[/bold cyan]\n")
    
    while True:
        try:
            token_address = console.input("[bold green]Enter Token Address (or 'q' to quit):[/bold green] ").strip()
            
            if token_address.lower() == 'q':
                break

            # Validate token address format
            try:
                Pubkey.from_string(token_address)
            except Exception as e:
                console.print(f"[red]Invalid address format: {str(e)}[/red]")
                continue

            bot = TradingBot()
            
            # Get token price with decimals
            price_data = await bot.get_token_price(token_address)
            
            # Check liquidity
            liquidity = bot.check_liquidity(token_address)
            
            # Display token info
            console.print(format_token_info(token_address, price_data, liquidity))
            
            # Get and display swap quote if there's liquidity
            if liquidity['has_liquidity'] and price_data:
                quote = bot.get_swap_quote(
                    input_mint="So11111111111111111111111111111111111111112",
                    output_mint=token_address,
                    amount=1000000000  # 1 SOL
                )
                console.print(format_swap_quote(quote))
            
            console.print("\n[green]Analysis complete![/green]\n")
            
        except Exception as e:
            console.print(f"[red]Error: {str(e)}[/red]")
            logger.error(f"Error during analysis: {str(e)}")

if __name__ == "__main__":
    asyncio.run(interactive_token_test())
