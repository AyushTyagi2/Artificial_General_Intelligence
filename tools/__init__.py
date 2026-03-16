"""
Digital Brain — Tools Package

Exposes ToolRouter. Sub-packages (search, wikipedia, notes, context, chatbot)
are loaded on demand by the router using file-based imports.
"""
from tools.router import (
    ToolRouter,
    get_tool_events,
    subscribe_tool_events,
    unsubscribe_tool_events,
)

__all__ = [
    "ToolRouter",
    "get_tool_events",
    "subscribe_tool_events",
    "unsubscribe_tool_events",
]