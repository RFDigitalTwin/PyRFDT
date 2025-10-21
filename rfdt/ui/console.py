"""
Console system for logging and command execution in the RFDT framework.
Provides a unified interface for server-side logging with real-time frontend display.
"""
from typing import Optional, Dict, Callable, List, Any, Tuple
from datetime import datetime
import re
import asyncio
import inspect
import shlex


class Console:
    """
    Console class for server-side logging and command registration.

    Example:
        server.console.log("Processing started")
        server.console.warn("Memory usage high")
        server.console.error("Failed to load file")

        # Register commands
        server.console.register_command("test", test_function)
        # User types: /test arg1 arg2
    """

    def __init__(self, server=None):
        """
        Initialize the Console.

        Args:
            server: Reference to the Server instance for WebSocket broadcasting
        """
        self.server = server
        self.messages: List[Dict[str, Any]] = []
        self.commands: Dict[str, Callable] = {}
        self.max_history = 1000  # Maximum messages to keep in history
        self._message_id_counter = 0  # Auto-incrementing ID counter

        # Register default commands
        self._register_default_commands()

    def _register_default_commands(self):
        """Register built-in console commands."""
        self.register_command("help", self._cmd_help)
        self.register_command("clear", self._cmd_clear)
        self.register_command("list", self._cmd_list)
        self.register_command("echo", self._cmd_echo)

    def log(self, message: str, source: Optional[str] = None) -> None:
        """
        Log an informational message.

        Args:
            message: The message to log
            source: Optional source identifier (e.g., module name)
        """
        self._add_message("log", message, source)

    def warn(self, message: str, source: Optional[str] = None) -> None:
        """
        Log a warning message.

        Args:
            message: The warning message
            source: Optional source identifier
        """
        self._add_message("warning", message, source)

    def error(self, message: str, source: Optional[str] = None) -> None:
        """
        Log an error message.

        Args:
            message: The error message
            source: Optional source identifier
        """
        self._add_message("error", message, source)

    def _add_message(self, level: str, message: str, source: Optional[str] = None) -> None:
        """
        Internal method to add a message to the console.

        Args:
            level: Message level (log, warning, error)
            message: The message content
            source: Optional source identifier
        """
        msg_data = {
            "id": self._message_id_counter,
            "level": level,
            "message": message,
            "source": source or "System",
            "timestamp": datetime.now().isoformat()
        }
        self._message_id_counter += 1

        # Add to history
        self.messages.append(msg_data)

        # Trim history if needed
        if len(self.messages) > self.max_history:
            self.messages = self.messages[-self.max_history:]

        # Broadcast to connected clients if server is available
        if self.server:
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    asyncio.create_task(self._broadcast_message(msg_data))
                else:
                    # If no event loop is running (e.g., in Jupyter), just skip broadcasting
                    pass
            except RuntimeError:
                # No event loop, skip broadcasting
                pass

        # Also print to terminal for debugging
        prefix = {
            "log": "[LOG]",
            "warning": "[WARN]",
            "error": "[ERROR]"
        }.get(level, "[INFO]")

        if source:
            print(f"{prefix} [{source}] {message}")
        else:
            print(f"{prefix} {message}")

    async def _broadcast_message(self, message_data: Dict[str, Any]) -> None:
        """
        Broadcast a console message to all connected clients.

        Args:
            message_data: The message data to broadcast
        """
        await self.server.broadcast({
            "type": "console_message",
            "data": message_data
        })

    def register_command(self, name: str, handler: Callable, description: str = "") -> None:
        """
        Register a command handler.

        Args:
            name: Command name (without the / prefix)
            handler: Function to call when command is executed
            description: Optional description for help text

        Example:
            def my_command(arg1: str, arg2: int = 0):
                return f"Executed with {arg1} and {arg2}"

            console.register_command("mycommand", my_command, "My custom command")
        """
        self.commands[name] = {
            "handler": handler,
            "description": description or f"Execute {name} command"
        }

    async def execute_command(self, command_str: str) -> Dict[str, Any]:
        """
        Execute a console command.

        Args:
            command_str: The full command string (e.g., "/test arg1 arg2")

        Returns:
            Dict with result or error information
        """
        # Check if it's a command (starts with /)
        if not command_str.startswith("/"):
            return {
                "success": False,
                "error": "Commands must start with /"
            }

        # Parse command and arguments
        try:
            parts = shlex.split(command_str[1:])  # Remove the / and split
        except ValueError as e:
            return {
                "success": False,
                "error": f"Failed to parse command: {str(e)}"
            }

        if not parts:
            return {
                "success": False,
                "error": "Empty command"
            }

        cmd_name = parts[0]
        args = parts[1:]

        # Find command handler
        if cmd_name not in self.commands:
            self.error(f"Unknown command: /{cmd_name}", source="Console")
            return {
                "success": False,
                "error": f"Unknown command: /{cmd_name}. Type /help for available commands."
            }

        # Execute command
        try:
            handler = self.commands[cmd_name]["handler"]

            # Get function signature to handle arguments properly
            sig = inspect.signature(handler)
            params = sig.parameters

            # Build kwargs from positional arguments
            kwargs = {}
            param_list = list(params.items())

            for i, arg in enumerate(args):
                if i >= len(param_list):
                    break

                param_name, param = param_list[i]

                # Try to convert argument to proper type
                if param.annotation != inspect.Parameter.empty:
                    try:
                        if param.annotation == int:
                            kwargs[param_name] = int(arg)
                        elif param.annotation == float:
                            kwargs[param_name] = float(arg)
                        elif param.annotation == bool:
                            kwargs[param_name] = arg.lower() in ('true', '1', 'yes')
                        else:
                            kwargs[param_name] = arg
                    except ValueError:
                        kwargs[param_name] = arg
                else:
                    kwargs[param_name] = arg

            # Execute handler
            if asyncio.iscoroutinefunction(handler):
                result = await handler(**kwargs)
            else:
                result = handler(**kwargs)

            # Log success (but mark it as from command execution)
            if result is not None:
                self.log(str(result), source=f"/{cmd_name}")

            return {
                "success": True,
                "result": str(result) if result is not None else "Command executed successfully"
            }

        except Exception as e:
            error_msg = f"Error executing /{cmd_name}: {str(e)}"
            self.error(error_msg, source="Console")
            return {
                "success": False,
                "error": error_msg
            }

    def get_history(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Get console message history.

        Args:
            limit: Maximum number of messages to return (newest first)

        Returns:
            List of message dictionaries
        """
        if limit:
            return self.messages[-limit:]
        return self.messages

    def clear(self, broadcast_message: bool = True) -> None:
        """Clear all console messages."""
        self.messages = []
        if broadcast_message:
            self.log("Console cleared", source="Console")

    # Built-in command handlers
    def _cmd_help(self) -> str:
        """Display available commands."""
        lines = ["Available commands:"]
        for name, info in sorted(self.commands.items()):
            lines.append(f"  /{name} - {info['description']}")
        return "\n".join(lines)

    def _cmd_clear(self) -> None:
        """Clear the console."""
        self.clear(broadcast_message=False)
        # Don't return anything to avoid duplicate message

    def _cmd_list(self) -> str:
        """List recent messages."""
        recent = self.get_history(limit=10)
        if not recent:
            return "No messages in history"

        lines = ["Recent messages:"]
        for msg in recent:
            time_str = msg['timestamp'].split('T')[1][:8]
            lines.append(f"[{time_str}] [{msg['level'].upper()}] {msg['message']}")
        return "\n".join(lines)

    def _cmd_echo(self, *args) -> str:
        """Echo back the arguments."""
        return " ".join(str(arg) for arg in args)