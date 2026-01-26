"""
CLI interface for exocortex memory system.

Commands:
    exo recall <query>  - Search memories semantically
    exo stats          - Show memory statistics
"""

import click
from pathlib import Path
from rich.console import Console
from rich.table import Table
from rich import print as rprint
from src.exocortex.indexer import MemoryIndexer

# Default storage location
DEFAULT_STORAGE_PATH = Path.home() / ".exocortex" / "memories"

console = Console()


@click.group()
@click.option(
    '--storage',
    type=click.Path(path_type=Path),
    default=DEFAULT_STORAGE_PATH,
    help='Storage directory for memories'
)
@click.pass_context
def cli(ctx, storage):
    """Exocortex - Your Personal Memory System"""
    ctx.ensure_object(dict)
    ctx.obj['storage_path'] = storage


@cli.command()
@click.argument('query')
@click.option(
    '--limit',
    default=10,
    type=int,
    help='Maximum number of results'
)
@click.option(
    '--min-score',
    default=0.0,
    type=float,
    help='Minimum similarity score (0.0-1.0)'
)
@click.pass_context
def recall(ctx, query, limit, min_score):
    """Search memories semantically.

    Example:
        exo recall "Python programming"
        exo recall "conversation with Alice" --limit 5
    """
    storage_path = ctx.obj['storage_path']

    # Check if storage exists
    if not storage_path.exists():
        console.print(
            f"[yellow]No memories found at {storage_path}[/yellow]",
            style="bold"
        )
        console.print("Memories will be created when you integrate with the translation pipeline.")
        return

    try:
        with console.status(f"[bold green]Searching for: {query}...", spinner="dots"):
            with MemoryIndexer(storage_path=storage_path) as indexer:
                results = indexer.search_memories(
                    query=query,
                    limit=limit,
                    min_score=min_score
                )

        if not results:
            console.print(f"[yellow]No memories found matching: {query}[/yellow]")
            return

        # Display results in a table
        table = Table(title=f"Search Results for: {query}")
        table.add_column("Score", style="cyan", width=8)
        table.add_column("Text", style="white")
        table.add_column("Speaker", style="green", width=15)
        table.add_column("Time", style="dim", width=19)

        for result in results:
            memory = result.memory
            score_str = f"{result.score:.3f}"
            text = memory.text[:100] + "..." if len(memory.text) > 100 else memory.text
            speaker = memory.speaker_name or memory.speaker_label or "Unknown"
            timestamp = memory.metadata.timestamp.strftime("%Y-%m-%d %H:%M:%S")

            table.add_row(score_str, text, speaker, timestamp)

        console.print(table)
        console.print(f"\n[dim]Found {len(results)} results[/dim]")

    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise click.Abort()


@cli.command()
@click.pass_context
def stats(ctx):
    """Show memory statistics.

    Example:
        exo stats
    """
    storage_path = ctx.obj['storage_path']

    # Check if storage exists
    if not storage_path.exists():
        console.print(
            f"[yellow]No memories found at {storage_path}[/yellow]",
            style="bold"
        )
        console.print("Memories will be created when you integrate with the translation pipeline.")
        return

    try:
        # This is a placeholder - you'll improve it when you add count methods
        console.print(f"[bold]Exocortex Statistics[/bold]\n")
        console.print(f"Storage Location: {storage_path}")

        # Check if Qdrant collection exists
        qdrant_path = storage_path / "qdrant"
        sqlite_path = storage_path / "metadata.db"

        if qdrant_path.exists():
            console.print(f"[green]✓[/green] Vector database: {qdrant_path}")
        else:
            console.print(f"[red]✗[/red] Vector database: Not found")

        if sqlite_path.exists():
            import sqlite3
            conn = sqlite3.connect(str(sqlite_path))
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM memories")
            count = cursor.fetchone()[0]
            conn.close()
            console.print(f"[green]✓[/green] Metadata database: {sqlite_path}")
            console.print(f"\n[bold cyan]{count}[/bold cyan] memories indexed")
        else:
            console.print(f"[red]✗[/red] Metadata database: Not found")

    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise click.Abort()


if __name__ == '__main__':
    cli()
