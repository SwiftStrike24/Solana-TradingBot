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

# Add project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.trading import TradingBot
from src.utils.helpers import setup_logging
from tests.test_token import format_token_info, format_swap_quote
from src.services.coingecko import CoinGeckoAPI

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
                    sol_price_usd = await bot.get_sol_price()
                    sol_amount = int((1.0 / sol_price_usd) * 1e9)  # Convert to lamports
                    
                    # Get swap quote
                    quote = bot.get_swap_quote(
                        input_mint="So11111111111111111111111111111111111111112",
                        output_mint=token_address,
                        amount=sol_amount
                    )
                    
                    if quote:
                        console.print(format_swap_quote(quote, price_data.get('token_info')))
                        
                        if console.input("\n[bold yellow]Execute swap? (y/n):[/bold yellow] ").lower() == 'y':
                            tx_sig = await bot.execute_swap(quote)
                            if tx_sig:
                                console.print(f"[green]Swap successful! Transaction: {tx_sig}[/green]")
                                console.print(f"View transaction: https://solscan.io/tx/{tx_sig}")
                            else:
                                console.print("[red]Swap failed![/red]")
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