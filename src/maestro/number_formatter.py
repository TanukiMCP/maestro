"""
Maestro Number Formatter

Intelligent number formatting utility to handle floating-point precision issues
and provide clean, readable numeric outputs across all Maestro tools.
"""

import re
import json
from typing import Any, Union, Dict, List
from decimal import Decimal, getcontext
import math

# Set high precision for decimal calculations
getcontext().prec = 50

class MaestroNumberFormatter:
    """
    Intelligent number formatter that handles floating-point precision issues
    and provides clean, readable numeric outputs.
    """
    
    def __init__(self, 
                 default_precision: int = 6,
                 scientific_threshold: float = 1e6,
                 small_threshold: float = 1e-4,
                 auto_round: bool = True):
        """
        Initialize the number formatter.
        
        Args:
            default_precision: Default number of decimal places to show
            scientific_threshold: Use scientific notation for numbers >= this value
            small_threshold: Use scientific notation for numbers <= this value
            auto_round: Automatically round to remove floating-point artifacts
        """
        self.default_precision = default_precision
        self.scientific_threshold = scientific_threshold
        self.small_threshold = small_threshold
        self.auto_round = auto_round
    
    def format_number(self, value: Union[int, float, str], precision: int = None) -> str:
        """
        Format a single number with intelligent precision handling.
        
        Args:
            value: The number to format
            precision: Override default precision
            
        Returns:
            Formatted number string
        """
        if precision is None:
            precision = self.default_precision
            
        try:
            # Convert to float if string
            if isinstance(value, str):
                value = float(value)
            
            # Handle special cases
            if math.isnan(value):
                return "NaN"
            if math.isinf(value):
                return "∞" if value > 0 else "-∞"
            
            # Handle very large or very small numbers
            abs_value = abs(value)
            if abs_value >= self.scientific_threshold or (abs_value <= self.small_threshold and abs_value != 0):
                return f"{value:.{precision}e}"
            
            # Handle floating-point precision issues
            if self.auto_round:
                # Check if this looks like a floating-point precision error
                rounded = round(value, precision)
                if abs(value - rounded) < 1e-14:  # Very small difference, likely precision error
                    value = rounded
            
            # Format with appropriate precision
            formatted = f"{value:.{precision}f}"
            
            # Remove trailing zeros and unnecessary decimal point
            if '.' in formatted:
                formatted = formatted.rstrip('0').rstrip('.')
            
            return formatted
            
        except (ValueError, TypeError):
            return str(value)
    
    def format_text(self, text: str, precision: int = None) -> str:
        """
        Format all numbers found in a text string.
        
        Args:
            text: Text containing numbers to format
            precision: Override default precision
            
        Returns:
            Text with formatted numbers
        """
        if precision is None:
            precision = self.default_precision
        
        # Pattern to match floating-point numbers (including scientific notation)
        number_pattern = r'-?\d+\.?\d*(?:[eE][+-]?\d+)?'
        
        def replace_number(match):
            try:
                number_str = match.group()
                number = float(number_str)
                return self.format_number(number, precision)
            except (ValueError, TypeError):
                return match.group()  # Return original if can't parse
        
        return re.sub(number_pattern, replace_number, text)
    
    def format_json_output(self, data: Any, precision: int = None) -> str:
        """
        Format JSON output with intelligent number formatting.
        
        Args:
            data: Data structure to format
            precision: Override default precision
            
        Returns:
            JSON string with formatted numbers
        """
        if precision is None:
            precision = self.default_precision
        
        def format_value(obj):
            if isinstance(obj, float):
                # Convert to formatted string, then back to appropriate type
                formatted = self.format_number(obj, precision)
                try:
                    # Try to convert back to int if it's a whole number
                    if '.' not in formatted and 'e' not in formatted.lower():
                        return int(formatted)
                    else:
                        return float(formatted)
                except ValueError:
                    return formatted
            elif isinstance(obj, dict):
                return {k: format_value(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [format_value(item) for item in obj]
            else:
                return obj
        
        formatted_data = format_value(data)
        return json.dumps(formatted_data, indent=2, ensure_ascii=False)
    
    def format_calculation_result(self, expression: str, result: Union[int, float], 
                                show_exact: bool = False, precision: int = None) -> str:
        """
        Format a calculation result with optional exact decimal representation.
        
        Args:
            expression: The mathematical expression
            result: The calculated result
            show_exact: Whether to show exact decimal representation
            precision: Override default precision
            
        Returns:
            Formatted calculation result
        """
        if precision is None:
            precision = self.default_precision
        
        formatted_result = self.format_number(result, precision)
        
        output = f"{expression} = {formatted_result}"
        
        if show_exact and isinstance(result, float):
            # Show exact decimal representation if different
            exact_decimal = Decimal(str(result))
            if str(exact_decimal) != formatted_result:
                output += f"\n(Exact: {exact_decimal})"
        
        return output
    
    def detect_precision_issues(self, value: float) -> Dict[str, Any]:
        """
        Detect potential floating-point precision issues.
        
        Args:
            value: The floating-point value to analyze
            
        Returns:
            Dictionary with precision analysis
        """
        analysis = {
            "has_precision_issue": False,
            "recommended_precision": self.default_precision,
            "exact_decimal": str(Decimal(str(value))),
            "binary_representation": value.hex(),
            "suggestions": []
        }
        
        # Check for common precision issues
        rounded_values = [round(value, i) for i in range(1, 16)]
        for i, rounded in enumerate(rounded_values, 1):
            if abs(value - rounded) < 1e-14:
                analysis["has_precision_issue"] = True
                analysis["recommended_precision"] = i
                analysis["suggestions"].append(f"Consider rounding to {i} decimal places")
                break
        
        # Check for very long decimal representations
        decimal_str = f"{value:.15f}".rstrip('0')
        if len(decimal_str.split('.')[-1]) > 10:
            analysis["suggestions"].append("Consider using decimal arithmetic for exact calculations")
        
        return analysis

# Global formatter instance
_global_formatter = MaestroNumberFormatter()

def format_number(value: Union[int, float, str], precision: int = 6) -> str:
    """Convenience function for formatting a single number."""
    return _global_formatter.format_number(value, precision)

def format_text(text: str, precision: int = 6) -> str:
    """Convenience function for formatting numbers in text."""
    return _global_formatter.format_text(text, precision)

def format_json_output(data: Any, precision: int = 6) -> str:
    """Convenience function for formatting JSON with clean numbers."""
    return _global_formatter.format_json_output(data, precision)

def format_calculation(expression: str, result: Union[int, float], 
                      show_exact: bool = False, precision: int = 6) -> str:
    """Convenience function for formatting calculation results."""
    return _global_formatter.format_calculation_result(expression, result, show_exact, precision)

def clean_output(text: str, precision: int = 6) -> str:
    """
    Clean up any text output by formatting numbers appropriately.
    This is the main function to use for cleaning Maestro tool outputs.
    """
    return _global_formatter.format_text(text, precision) 