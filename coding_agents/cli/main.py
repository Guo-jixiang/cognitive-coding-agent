"""CLI entry point for the Cognitive Coding Agent REPL.

This module provides the click-based command-line interface that starts
the interactive REPL session with configurable paradigm selection.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import click

# Load .env as early as possible
try:
    from dotenv import load_dotenv

    _env_path = Path(__file__).resolve().parent.parent.parent / ".env"
    if _env_path.exists():
        load_dotenv(dotenv_path=_env_path, override=True)
    else:
        load_dotenv(override=True)
except ImportError:
    pass


@click.command()
@click.option(
    "--paradigm",
    default="reflection",
    type=click.Choice(["react", "plan_and_solve", "reflection"]),
    help="Reasoning paradigm to use.",
)
def main(paradigm: str) -> None:
    """Start the Cognitive Coding Agent interactive REPL.

    Launches an interactive chat session where you can converse with the
    agent, switch paradigms, and manage memory.
    """
    from coding_agents.cli.repl import start_repl

    asyncio.run(start_repl(paradigm))


if __name__ == "__main__":
    main()
