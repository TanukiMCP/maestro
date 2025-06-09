# Copyright (c) 2025 TanukiMCP Orchestra
# Licensed under Non-Commercial License - Commercial use requires approval from TanukiMCP
# Contact tanukimcp@gmail.com for commercial licensing inquiries

"""
Basic Arithmetic Engine for MAESTRO IAE

A simple, direct engine for handling basic mathematical calculations.
"""

import re
import operator

class BasicArithmeticEngine:
    """A computational engine for performing basic arithmetic."""

    def __init__(self):
        """Initializes the BasicArithmeticEngine."""
        self.name = "basic_arithmetic_engine"
        self.version = "1.0"
        self.capabilities = [
            "Handles simple arithmetic operations: +, -, *, /",
            "Parses natural language math queries.",
            "Returns direct numerical answers."
        ]
        self.ops = {
            '+': operator.add,
            '-': operator.sub,
            '*': operator.mul,
            '/': operator.truediv
        }

    def execute(self, analysis_request: str, **kwargs) -> dict:
        """
        Executes the arithmetic calculation from a natural language request.

        Args:
            analysis_request: A string containing the math problem.

        Returns:
            A dictionary containing the result of the calculation.
        """
        try:
            # Sanitize and parse the input string
            # This regex finds two numbers and the operator between them.
            match = re.search(r'([\d\.]+)\s*([+\-*/])\s*([\d\.]+)', analysis_request)
            if not match:
                return {"status": "error", "error": "Invalid arithmetic expression. Use a format like '10 + 5'."}

            num1 = float(match.group(1))
            op = match.group(2)
            num2 = float(match.group(3))

            if op == '/' and num2 == 0:
                return {"status": "error", "error": "Division by zero is not allowed."}

            result = self.ops[op](num1, num2)
            
            return {
                "status": "success",
                "result": result,
                "calculation": f"{num1} {op} {num2} = {result}"
            }
        except Exception as e:
            return {"status": "error", "error": str(e)} 