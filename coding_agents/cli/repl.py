"""Interactive REPL implementation for the Cognitive Coding Agent.

This module provides the :class:`AgentREPL` class that implements the
read-eval-print loop with special command support, colored output, and
real-time reasoning step display.
"""

from __future__ import annotations

import logging
from typing import Any

import click

from coding_agents.context.builder import ContextBuilder
from coding_agents.core.actions.registry import ActionRegistry
from coding_agents.core.engine import AgentEngine
from coding_agents.llm.client import ChatMessage, LLMClient
from coding_agents.memory.factory import create_memory_manager
from coding_agents.memory.manager import MemoryManager

logger = logging.getLogger(__name__)

# Valid paradigm names for runtime switching
_VALID_PARADIGMS: frozenset[str] = frozenset({"react", "plan_and_solve", "reflection"})


class AgentREPL:
    """Interactive REPL for conversing with the Cognitive Coding Agent.

    Supports special commands (/clear, /paradigm, /memory, /quit) and
    displays reasoning steps with colored output.

    Args:
        engine: The AgentEngine instance for processing messages.
        memory_manager: The MemoryManager for session and stats.
        paradigm: Initial reasoning paradigm name.
    """

    def __init__(
        self,
        engine: AgentEngine,
        memory_manager: MemoryManager,
        paradigm: str = "reflection",
    ) -> None:
        """Initialize the REPL.

        Args:
            engine: The AgentEngine instance.
            memory_manager: The MemoryManager instance.
            paradigm: Initial reasoning paradigm.
        """
        self._engine = engine
        self._memory_manager = memory_manager
        self._paradigm = paradigm
        self._running = False
        self._conversation_history: list[ChatMessage] = []

    @property
    def paradigm(self) -> str:
        """The currently active reasoning paradigm."""
        return self._paradigm

    async def start(self) -> None:
        """Start the interactive REPL loop.

        Displays a welcome message and enters the main input loop.
        Handles special commands and sends regular messages to the agent.
        Exits on /quit or EOF (Ctrl+D).
        """
        self._running = True
        click.echo(
            click.style(
                "🧠 Cognitive Coding Agent REPL", fg="cyan", bold=True
            )
        )
        click.echo(
            click.style(
                f"   Paradigm: {self._paradigm} | Type /quit to exit",
                fg="white",
                dim=True,
            )
        )
        click.echo("")

        while self._running:
            try:
                user_input = click.prompt(
                    click.style("You", fg="green", bold=True),
                    prompt_suffix="> ",
                )
            except (EOFError, KeyboardInterrupt):
                click.echo("")
                await self._handle_quit()
                break

            user_input = user_input.strip()
            if not user_input:
                continue

            await self.handle_input(user_input)

    async def handle_input(self, user_input: str) -> None:
        """Process user input: special commands or agent message.

        Args:
            user_input: The raw user input string.
        """
        if user_input.startswith("/"):
            handled = await self.handle_command(user_input)
            if handled:
                return

        # Store user message in working memory
        working = self._memory_manager.subsystems.get("working")
        if working is not None:
            from coding_agents.memory.base import create_memory_item

            user_item = create_memory_item(
                content=user_input,
                memory_type="working",
                metadata={"role": "user"},
                importance=0.5,
            )
            await working.store(user_item)

        # Append to conversation history
        self._conversation_history.append(
            ChatMessage(role="user", content=user_input)
        )

        # Build history dicts for the engine (exclude the current message)
        history_for_engine: list[dict[str, str]] = [
            {"role": msg.role, "content": msg.content}
            for msg in self._conversation_history[:-1]
        ]

        # Send to agent engine
        click.echo(
            click.style("  Thinking...", fg="yellow", dim=True)
        )

        try:
            response = await self._engine.run(
                user_message=user_input,
                paradigm=self._paradigm,
                conversation_history=history_for_engine if history_for_engine else None,
            )
        except Exception as exc:
            click.echo(
                click.style(f"  Error: {exc}", fg="red")
            )
            return

        # Store assistant response in working memory
        if working is not None:
            assistant_item = create_memory_item(
                content=response.answer,
                memory_type="working",
                metadata={"role": "assistant"},
                importance=0.5,
            )
            await working.store(assistant_item)

        # Append assistant response to conversation history
        self._conversation_history.append(
            ChatMessage(role="assistant", content=response.answer)
        )

        # Display reasoning trace
        for step in response.reasoning_trace:
            self.display_reasoning_step(step)

        # Display final answer
        click.echo("")
        click.echo(
            click.style("Agent", fg="blue", bold=True) + "> " + response.answer
        )
        click.echo("")

    async def handle_command(self, command: str) -> bool:
        """Handle special commands.

        Supported commands:
            /clear — Clear conversation history (working memory).
            /paradigm <name> — Switch the active reasoning paradigm.
            /memory — Display memory usage statistics.
            /quit — Exit the REPL session.

        Args:
            command: The command string (including the leading /).

        Returns:
            True if the command was recognized and handled, False otherwise.
        """
        parts = command.strip().split(maxsplit=1)
        cmd = parts[0].lower()
        arg = parts[1] if len(parts) > 1 else ""

        if cmd == "/clear":
            await self._handle_clear()
            return True
        elif cmd == "/paradigm":
            self._handle_paradigm(arg)
            return True
        elif cmd == "/memory":
            await self._handle_memory()
            return True
        elif cmd == "/quit":
            await self._handle_quit()
            return True
        else:
            click.echo(
                click.style(
                    f"  Unknown command: {cmd}. "
                    "Available: /clear, /paradigm, /memory, /quit",
                    fg="red",
                )
            )
            return True

    def display_reasoning_step(self, step: dict[str, Any]) -> None:
        """Display a reasoning step with colored output.

        Formats different step types (thought, action, observation,
        reflection, error) with distinct colors for readability.

        Args:
            step: A dictionary with 'type' and 'content' keys.
        """
        step_type = step.get("type", "unknown")
        content = step.get("content", "")

        color_map: dict[str, str] = {
            "thought": "yellow",
            "action": "magenta",
            "observation": "cyan",
            "reflection": "blue",
            "plan": "green",
            "error": "red",
        }

        color = color_map.get(step_type, "white")
        prefix = f"  [{step_type.capitalize()}]"

        click.echo(
            click.style(prefix, fg=color, bold=True)
            + " "
            + click.style(str(content)[:500], fg=color)
        )

    async def _handle_clear(self) -> None:
        """Clear working memory (conversation history)."""
        working = self._memory_manager.subsystems.get("working")
        if working is not None:
            await working.clear()
        self._conversation_history.clear()
        click.echo(
            click.style("  ✓ Conversation history cleared.", fg="green")
        )

    def _handle_paradigm(self, arg: str) -> None:
        """Switch the active reasoning paradigm.

        Args:
            arg: The paradigm name to switch to.
        """
        name = arg.strip().lower()
        if not name:
            click.echo(
                click.style(
                    f"  Current paradigm: {self._paradigm}. "
                    f"Available: {', '.join(sorted(_VALID_PARADIGMS))}",
                    fg="cyan",
                )
            )
            return

        if name not in _VALID_PARADIGMS:
            click.echo(
                click.style(
                    f"  Unknown paradigm '{name}'. "
                    f"Available: {', '.join(sorted(_VALID_PARADIGMS))}",
                    fg="red",
                )
            )
            return

        self._paradigm = name
        click.echo(
            click.style(f"  ✓ Paradigm switched to: {name}", fg="green")
        )

    async def _handle_memory(self) -> None:
        """Display memory usage statistics."""
        click.echo(click.style("  Memory Statistics:", fg="cyan", bold=True))

        for name, subsystem in self._memory_manager.subsystems.items():
            degraded = name in self._memory_manager.degraded_subsystems
            status = click.style("degraded", fg="red") if degraded else click.style("active", fg="green")
            click.echo(f"    {name}: {status}")

    async def _handle_quit(self) -> None:
        """Exit the REPL session gracefully."""
        self._running = False
        click.echo(
            click.style("  Goodbye! 👋", fg="cyan")
        )


async def start_repl(paradigm: str = "reflection") -> None:
    """Create components and start the REPL.

    This is the main entry point called from the CLI main module.
    It creates the MemoryManager, Orchestrator, AgentEngine, and starts
    the REPL loop.

    Args:
        paradigm: The initial reasoning paradigm to use.
    """
    # Load environment variables
    try:
        from pathlib import Path

        from dotenv import load_dotenv

        # Search for .env from the current directory upward
        env_path = Path.cwd() / ".env"
        if not env_path.exists():
            # Try project root (where pyproject.toml lives)
            for parent in Path.cwd().parents:
                if (parent / "pyproject.toml").exists():
                    env_path = parent / ".env"
                    break
        load_dotenv(dotenv_path=env_path, override=True)
    except ImportError:
        pass

    # Create components
    memory_manager = create_memory_manager()
    llm_client = LLMClient()
    context_builder = ContextBuilder(memory_manager)
    action_registry = ActionRegistry()

    # Create Orchestrator for SubAgent-based execution
    from coding_agents.core.agents.orchestrator import Orchestrator

    orchestrator = Orchestrator(
        llm_client=llm_client,
        memory_manager=memory_manager,
        context_builder=context_builder,
    )

    engine = AgentEngine(
        llm_client=llm_client,
        memory_manager=memory_manager,
        context_builder=context_builder,
        action_registry=action_registry,
        orchestrator=orchestrator,
    )

    # Initialize
    await engine.initialize()

    try:
        repl = AgentREPL(
            engine=engine,
            memory_manager=memory_manager,
            paradigm=paradigm,
        )
        await repl.start()
    finally:
        await engine.shutdown()
        await llm_client.close()
