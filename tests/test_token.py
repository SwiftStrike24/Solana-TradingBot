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
            "[red]Error: Unable to fetch token information[/red]"
        ]
        return Panel("\n".join(details), title="[bold red]Token Information[/bold red]", border_style="red")

    # Get token info
    token_info = price_data.get('token_info', {})
    symbol = token_info.get('symbol', 'Unknown')
    name = token_info.get('name', 'Unknown Token')
    verified = token_info.get('verified', False)
    daily_volume = token_info.get('daily_volume')
    created_at = token_info.get('created_at', 'Unknown')
    
    # Format creation date if available
    if created_at and created_at != 'Unknown':
        try:
            created_at = datetime.fromisoformat(created_at.replace('Z', '+00:00')).strftime('%Y-%m-%d %I:%M %p')
        except:
            pass
    
    # Format verification status
    verification_status = "[green]✓ Verified[/green]" if verified else "[yellow]⚠ Unverified[/yellow]"
    
    # Format prices with appropriate precision
    tokens_per_sol = price_data.get('tokens_per_sol', 0)
    sol_per_token = price_data.get('sol_per_token', 0)
    
    # Dynamic precision based on value magnitude
    def get_precision(value: float) -> int:
        if value == 0 or value is None:
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
    
    # Format daily volume with fallback
    volume_display = f"{daily_volume:,.2f} USD" if daily_volume is not None else "No data"
    
    details = [
        f"[cyan]Token:[/cyan] {name} ({symbol}) {verification_status}",
        f"[cyan]Address:[/cyan] {token_address}",
        f"[yellow]Created:[/yellow] {created_at}",
        f"[magenta]Daily Volume:[/magenta] {volume_display}",
        "",
        f"[green]1 SOL = {tokens_per_sol:.{token_precision}f} {symbol}[/green]",
        f"[green]1 {symbol} = {sol_per_token:.{sol_precision}f} SOL[/green]",
        f"[magenta]Price Impact:[/magenta] {price_data.get('price_impact', 0)}%",
        "",
        f"[{'green' if liquidity['has_liquidity'] else 'red'}]Liquidity Analysis:[/]",
        f"  • Impact: {liquidity.get('price_impact', 0):.2f}%",
        f"  • Routes: {liquidity.get('routes_available', 0)}",
        f"  • Fees: {liquidity.get('total_fees', 0)/1e9:.6f} SOL"
    ]
    
    return Panel(
        "\n".join(details),
        title="[bold cyan]Token Information[/bold cyan]",
        border_style="cyan"
    )

def format_swap_quote(quote: dict, token_info: dict = None, all_tokens_info: dict = None) -> Panel:
    """Format swap quote into a rich panel"""
    if not quote:
        return Panel("No quote available", title="Swap Quote", border_style="red")

    # Get token symbol and decimals if available
    token_symbol = token_info.get('symbol', 'tokens') if token_info else 'tokens'
    token_decimals = token_info.get('decimals', 6) if token_info else 6
    
    usd_value = float(quote.get('swapUsdValue', 0))
    time_taken_ms = float(quote.get('timeTaken', 0)) * 1000  # Convert to milliseconds
    
    # Format output amount using correct decimals
    input_amount = float(quote['inAmount']) / 1e9  # SOL is always 9 decimals
    output_amount = float(quote['outAmount']) / (10 ** token_decimals)

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

    # Format USD with full decimals for small values
    def format_usd_value(value):
        if value >= 0.01:
            return f"{value:,.2f}"
        elif value >= 0.000001:
            decimal_places = min(9, abs(len(str(value).split('.')[-1])))
            return f"{value:.{decimal_places}f}"
        else:
            return f"{value:.2e}"

    formatted_input = format_token_amount(input_amount, "SOL")
    formatted_output = format_token_amount(output_amount, token_symbol)
    formatted_usd = format_usd_value(usd_value)
    
    quote_details = [
        f"💰 [cyan]Input Amount:[/cyan] {formatted_input} SOL (${formatted_usd})",
        f"💎 [cyan]Output Amount:[/cyan] {formatted_output} {token_symbol}",
        f"🛡️ [yellow]Slippage:[/yellow] {quote['slippageBps']/100}%",
        f"📊 [magenta]Price Impact:[/magenta] {quote.get('priceImpactPct', '0')}%",
        f"💵 [green]USD Value:[/green] ${formatted_usd}",
        f"⚡ [bright_yellow]Time:[/bright_yellow] {time_taken_ms:.2f}ms"
    ]

    route_table = Table(
        title="🔄 Route Details",
        show_header=True,
        header_style="bold cyan",
        box=DOUBLE_EDGE
    )
    
    route_table.add_column("Step", justify="center", style="cyan")
    route_table.add_column("DEX", style="magenta")
    route_table.add_column("From", style="yellow")
    route_table.add_column("To", style="yellow") 
    route_table.add_column("Fee", justify="right", style="red")

    # Use dynamic token info
    all_tokens_info = all_tokens_info or {}
    
    for idx, route in enumerate(quote['routePlan'], 1):
        swap = route['swapInfo']
        
        # Get token symbols from the token info
        from_token = all_tokens_info.get(swap['inputMint'], {})
        to_token = all_tokens_info.get(swap['outputMint'], {})
        fee_token = all_tokens_info.get(swap['feeMint'], {})
        
        from_symbol = from_token.get('symbol', '...')
        to_symbol = to_token.get('symbol', '...')
        
        # Handle fee amount and symbol
        if (swap['feeMint'] == "11111111111111111111111111111111" or 
            swap['feeMint'] == "So11111111111111111111111111111111111111112"):
            fee_amount = float(swap['feeAmount'])/1e9
            fee_symbol = "SOL"
        else:
            decimals = fee_token.get('decimals', 6)
            fee_amount = float(swap['feeAmount'])/(10 ** decimals)
            fee_symbol = fee_token.get('symbol', '...')
        
        route_table.add_row(
            f"#{idx}",
            f"🏦 {swap['label']}",
            from_symbol,
            to_symbol,
            f"🔸 {fee_amount:.6f} {fee_symbol}"
        )

    return Panel(
        Group(
            "\n".join(quote_details),
            route_table
        ),
        title="[bold yellow]Swap Quote Details[/bold yellow]",
        border_style="yellow"
    )

def format_new_tokens(tokens: list) -> Panel:
    """Format new tokens information into a rich panel"""
    if not tokens:
        return Panel("[red]No new tokens found[/red]", title="New Tokens", border_style="red")

    table = Table(
        title="🆕 New Tokens",
        show_header=True,
        header_style="bold cyan",
        box=DOUBLE_EDGE
    )
    
    table.add_column("Symbol", style="cyan")
    table.add_column("Name", style="green")
    table.add_column("Address", style="magenta")
    table.add_column("Created", style="yellow")
    table.add_column("Markets", style="blue")
    
    for token in tokens:
        created_at = datetime.fromtimestamp(int(token['created_at'])).strftime('%Y-%m-%d %I:%M %p')
        markets = ", ".join(token.get('known_markets', []))[:30] + "..." if len(token.get('known_markets', [])) > 0 else "None"
        
        table.add_row(
            token.get('symbol', 'Unknown'),
            token.get('name', 'Unknown Token'),
            token.get('mint', 'N/A'),
            created_at,
            markets
        )
    
    return Panel(table, title="[bold cyan]New Tokens[/bold cyan]", border_style="cyan")

async def interactive_token_test():
    timestamp = datetime.now().strftime("%Y-%m-%d_%I-%M-%S_%p")
    logger = setup_logging(f"logs/test_token_{timestamp}.log")
    
    console.print("\n[bold cyan]🚀 Starting Solana Token Analyzer[/bold cyan]\n")
    bot = TradingBot()
    
    while True:
        try:
            choice = console.input("[bold green]Choose action ([cyan]1[/cyan]: Analyze Token, [cyan]2[/cyan]: View New Tokens, [cyan]q[/cyan]: Quit):[/bold green] ").strip()
            
            if choice.lower() == 'q':
                break
                
            if choice == '2':
                new_tokens = await bot.get_new_tokens(limit=10)  # Get 10 newest tokens
                if new_tokens:
                    console.print(format_new_tokens(new_tokens))
                continue

            if choice == '1':
                token_address = console.input("[bold green]Enter Token Address:[/bold green] ").strip()
                
                # Validate token address format
                try:
                    Pubkey.from_string(token_address)
                except Exception as e:
                    console.print(f"[red]Invalid address format: {str(e)}[/red]")
                    continue
                
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
            
            console.print("\n[green]Analysis complete![/green]\n")
            
        except Exception as e:
            console.print(f"[red]Error: {str(e)}[/red]")
            logger.error(f"Error during analysis: {str(e)}")

if __name__ == "__main__":
    asyncio.run(interactive_token_test())
