# Copyright (c) 2025 TanukiMCP Orchestra
# Licensed under Non-Commercial License - Commercial use requires approval from TanukiMCP
# Contact tanukimcp@gmail.com for commercial licensing inquiries

"""
Actual Large Language Model (LLM) client implementation for MCP.

This module provides a real client that interacts with a language model through the MCP Context.
It integrates with the Model Context Protocol to enable proper agentic orchestration.
"""

from typing import Any, Dict, Optional
from mcp.server.fastmcp import Context

class LLMResponse:
    """Response from an LLM, containing the generated text."""
    def __init__(self, text: str):
        self.text = text

class LLMClient:
    """
    A real client that interacts with a large language model through the MCP Context.
    Instead of returning mock data, it calls the actual LLM to generate responses.
    """
    def __init__(self):
        """Initialize the LLM client."""
        pass

    async def generate(self, ctx: Context, prompt: str, response_format: Optional[Dict[str, Any]] = None) -> LLMResponse:
        """
        Makes a real call to an LLM through the provided MCP Context.
        
        Args:
            ctx: The MCP Context object used to access the LLM
            prompt: The prompt to send to the LLM
            response_format: Optional format specification for the response (e.g., JSON)
            
        Returns:
            LLMResponse containing the LLM's response
        """
        # Use the context to sample from the LLM with the given prompt
        response = await ctx.sample(prompt, response_format)
        
        # Return the text response wrapped in our LLMResponse class
        return LLMResponse(text=response.text) 