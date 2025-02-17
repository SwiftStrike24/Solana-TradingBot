import os
import sys
import asyncio
from datetime import datetime
import json
from rich.console import Console, Group
from rich.panel import Panel
from rich.table import Table
from rich.layout import Layout
from rich.text import Text
from rich.box import DOUBLE_EDGE

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.trading import TradingBot
from src.utils.helpers import setup_logging

console = Console()

def format_swap_quote(quote):
    """Format swap quote into a rich formatted output"""
    # Create main quote panel
    quote_details = []
    
    # Add basic quote info
    quote_details.extend([
        f"💎 [cyan]Input Token:[/cyan] {quote['inputMint'][:4]}...{quote['inputMint'][-4:]}",
        f"🎯 [cyan]Output Token:[/cyan] {quote['outputMint'][:4]}...{quote['outputMint'][-4:]}",
        f"💰 [green]Amount In:[/green] {float(quote['inAmount'])/1e9:.4f} SOL",
        f"💸 [green]Amount Out:[/green] {float(quote['outAmount'])/1e9:.4f} BONK",
        f"🛡️ [yellow]Slippage:[/yellow] {quote['slippageBps']/100}%",
        f"📊 [magenta]Price Impact:[/magenta] {quote['priceImpactPct'] or '0'}%",
        f"💵 [blue]USD Value:[/blue] ${float(quote['swapUsdValue']):.2f}",
        f"⚡ [grey]Time Taken:[/grey] {float(quote['timeTaken'])*1000:.2f}ms\n"
    ])
    
    # Create route table
    route_table = Table(
        title="🔄 [bold yellow]Swap Route Details[/bold yellow]",
        show_header=True,
        header_style="bold cyan",
        box=DOUBLE_EDGE
    )
    
    # Add columns
    route_table.add_column("Step", justify="center", style="cyan")
    route_table.add_column("DEX", style="magenta")
    route_table.add_column("Input → Output", style="green")
    route_table.add_column("Amount", justify="right", style="yellow")
    route_table.add_column("Fee", justify="right", style="red")

    # Add route steps
    for idx, route in enumerate(quote['routePlan'], 1):
        swap = route['swapInfo']
        in_token = swap['inputMint'][:4] + "..." + swap['inputMint'][-4:]
        out_token = swap['outputMint'][:4] + "..." + swap['outputMint'][-4:]
        
        # Handle fee amount and symbol
        if (swap['feeMint'] == "11111111111111111111111111111111" or 
            swap['feeMint'] == "So11111111111111111111111111111111111111112"):
            fee_amount = float(swap['feeAmount'])/1e9
            fee_symbol = "SOL"
        else:
            fee_amount = float(swap['feeAmount'])/1e9  # Default to SOL decimals
            fee_symbol = "..."

        route_table.add_row(
            f"#{idx}",
            f"🏦 {swap['label']}",
            f"{in_token} → {out_token}",
            f"💰 {float(swap['outAmount'])/1e9:.4f}",
            f"🔸 {fee_amount:.6f} {fee_symbol}"
        )

    # Combine everything into a panel
    quote_panel = Panel(
        Group(
            "\n".join(quote_details),
            route_table
        ),
        title="📊 [bold yellow]Swap Quote Summary[/bold yellow]",
        border_style="yellow"
    )
    
    return quote_panel

async def test_basic_functions():
    # Setup test-specific logging
    timestamp = datetime.now().strftime("%Y-%m-%d_%I-%M-%S_%p")
    logger = setup_logging(f"logs/test_connection_{timestamp}.log")
    
    console.print("\n[bold cyan]🚀 Starting Solana Trading Bot Test[/bold cyan]\n")
    logger.info("🚀 Starting connection test...")
    
    bot = TradingBot()
    bonk_address = "DezXAZ8z7PnrnRJjz3wXBoRgixCa6xjnB7YaB1pPB263"
    
    # Test price check
    price = bot.get_token_price(bonk_address)
    price_panel = Panel(
        f"[green]💰 BONK Price: {price:.6f} SOL[/green]",
        title="Price Check",
        border_style="green"
    )
    console.print(price_panel)
    logger.info(f"💰 BONK Price check completed: {price:.6f} SOL")
    
    # Test liquidity check
    has_liquidity = bot.check_liquidity(bonk_address)
    liquidity_status = "✅ Sufficient" if has_liquidity else "❌ Insufficient"
    liquidity_panel = Panel(
        f"[{'green' if has_liquidity else 'red'}]🌊 Liquidity Status: {liquidity_status}[/]",
        title="Liquidity Check",
        border_style="blue"
    )
    console.print(liquidity_panel)
    logger.info(f"🌊 Liquidity check completed: {liquidity_status}")
    
    # Test swap quote
    quote = bot.get_swap_quote(
        input_mint="So11111111111111111111111111111111111111112",
        output_mint=bonk_address,
        amount=1000000000
    )
    
    console.print("\n[bold yellow]📊 Swap Quote Details:[/bold yellow]")
    console.print(format_swap_quote(quote))
    
    # Update the swap quote logging section
    formatted_quote = json.dumps(quote, indent=2)
    logger.info("📊 Detailed Swap Quote:\n" + "="*50 + "\n" + 
                "🔄 Route Summary:\n" + 
                "\n".join([f"   Step {i+1}: {r['swapInfo']['label']}" 
                          for i, r in enumerate(quote['routePlan'])]) +
                "\n" + "="*50 + "\n" + formatted_quote)
    
    console.print("\n[bold green]✨ Test Completed Successfully! ✨[/bold green]\n")
    logger.info("✨ Test completed successfully!")

if __name__ == "__main__":
    asyncio.run(test_basic_functions())
