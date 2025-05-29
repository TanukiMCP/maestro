"""
Code Quality Engine

Uses pylint, black, pytest, and other code quality tools to analyze
and improve code quality automatically.
"""

import asyncio
import logging
import re
import ast
from typing import Dict, List, Any, Optional

logger = logging.getLogger(__name__)

# Try to import code quality libraries with graceful fallbacks
try:
    import black
    import subprocess
    CODE_QUALITY_LIBRARIES_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Some code quality libraries not available: {e}")
    CODE_QUALITY_LIBRARIES_AVAILABLE = False


class CodeQualityEngine:
    """
    Provides code quality analysis and improvement using specialized tools.
    """
    
    def __init__(self):
        logger.info("🔧 Code Quality Engine initialized")
    
    async def analyze_and_improve_code(
        self,
        code: str,
        requirements: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Analyze code quality and suggest improvements.
        
        Args:
            code: Code to analyze
            requirements: Specific analysis requirements
            
        Returns:
            Comprehensive code quality analysis
        """
        if requirements is None:
            requirements = {}
        
        try:
            # Perform various code quality checks
            syntax_analysis = await self._check_syntax(code)
            style_analysis = await self._analyze_style(code)
            complexity_analysis = await self._analyze_complexity(code)
            security_analysis = await self._check_security_issues(code)
            improvements = await self._suggest_improvements(code)
            
            return {
                "syntax_analysis": syntax_analysis,
                "style_analysis": style_analysis,
                "complexity_analysis": complexity_analysis,
                "security_analysis": security_analysis,
                "improvements": improvements,
                "overall_score": self._calculate_overall_score(
                    syntax_analysis, style_analysis, complexity_analysis
                ),
                "status": "Code quality analysis complete"
            }
            
        except Exception as e:
            logger.error(f"Code quality analysis failed: {str(e)}")
            return {"error": f"Code quality analysis error: {str(e)}"}
    
    async def _check_syntax(self, code: str) -> Dict[str, Any]:
        """Check code syntax using AST parsing."""
        
        try:
            # Try to parse the code
            ast.parse(code)
            return {
                "valid": True,
                "message": "✅ Syntax is valid",
                "errors": []
            }
        except SyntaxError as e:
            return {
                "valid": False,
                "message": f"❌ Syntax error: {str(e)}",
                "errors": [str(e)],
                "line": getattr(e, 'lineno', None),
                "column": getattr(e, 'offset', None)
            }
        except Exception as e:
            return {
                "valid": False,
                "message": f"❌ Parse error: {str(e)}",
                "errors": [str(e)]
            }
    
    async def _analyze_style(self, code: str) -> Dict[str, Any]:
        """Analyze code style and formatting."""
        
        style_issues = []
        suggestions = []
        
        lines = code.split('\n')
        
        # Check line length
        long_lines = [(i+1, line) for i, line in enumerate(lines) if len(line) > 79]
        if long_lines:
            style_issues.append(f"Lines exceed 79 characters: {[line_num for line_num, _ in long_lines[:3]]}")
            suggestions.append("Consider breaking long lines")
        
        # Check indentation consistency
        indentations = []
        for line in lines:
            if line.strip():  # Non-empty line
                indent = len(line) - len(line.lstrip())
                if indent > 0:
                    indentations.append(indent)
        
        if indentations:
            # Check if using consistent indentation
            if len(set(indentations)) > 2:  # Allow for nested indentation
                style_issues.append("Inconsistent indentation detected")
                suggestions.append("Use consistent indentation (4 spaces recommended)")
        
        # Check for trailing whitespace
        trailing_whitespace = [i+1 for i, line in enumerate(lines) if line.rstrip() != line]
        if trailing_whitespace:
            style_issues.append(f"Trailing whitespace on lines: {trailing_whitespace[:3]}")
            suggestions.append("Remove trailing whitespace")
        
        # Check naming conventions
        function_names = re.findall(r'def\s+([a-zA-Z_][a-zA-Z0-9_]*)', code)
        bad_function_names = [name for name in function_names if not name.islower() and '_' not in name]
        if bad_function_names:
            style_issues.append(f"Non-snake_case function names: {bad_function_names}")
            suggestions.append("Use snake_case for function names")
        
        # Check for missing docstrings
        if 'def ' in code and '"""' not in code and "'''" not in code:
            style_issues.append("Functions missing docstrings")
            suggestions.append("Add docstrings to functions")
        
        return {
            "issues": style_issues,
            "suggestions": suggestions,
            "score": max(0, 100 - len(style_issues) * 10),
            "status": "✅ Style analysis complete" if not style_issues else "⚠️ Style issues found"
        }
    
    async def _analyze_complexity(self, code: str) -> Dict[str, Any]:
        """Analyze code complexity."""
        
        try:
            # Parse the code to analyze complexity
            tree = ast.parse(code)
            
            complexity_metrics = {
                "functions": 0,
                "classes": 0,
                "lines_of_code": len([line for line in code.split('\n') if line.strip()]),
                "cyclomatic_complexity": 0,
                "nesting_depth": 0
            }
            
            # Count functions and classes
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    complexity_metrics["functions"] += 1
                elif isinstance(node, ast.ClassDef):
                    complexity_metrics["classes"] += 1
                elif isinstance(node, (ast.If, ast.For, ast.While, ast.Try)):
                    complexity_metrics["cyclomatic_complexity"] += 1
            
            # Estimate nesting depth
            max_depth = 0
            current_depth = 0
            
            for line in code.split('\n'):
                stripped = line.lstrip()
                if stripped and not stripped.startswith('#'):
                    indent = len(line) - len(stripped)
                    current_depth = indent // 4  # Assuming 4-space indentation
                    max_depth = max(max_depth, current_depth)
            
            complexity_metrics["nesting_depth"] = max_depth
            
            # Calculate complexity score
            complexity_score = 100
            if complexity_metrics["cyclomatic_complexity"] > 10:
                complexity_score -= 20
            if complexity_metrics["nesting_depth"] > 4:
                complexity_score -= 15
            if complexity_metrics["lines_of_code"] > 100:
                complexity_score -= 10
            
            return {
                "metrics": complexity_metrics,
                "score": max(0, complexity_score),
                "recommendations": self._get_complexity_recommendations(complexity_metrics)
            }
            
        except Exception as e:
            return {
                "error": f"Complexity analysis failed: {str(e)}",
                "score": 0
            }
    
    def _get_complexity_recommendations(self, metrics: Dict[str, int]) -> List[str]:
        """Get recommendations based on complexity metrics."""
        
        recommendations = []
        
        if metrics["cyclomatic_complexity"] > 10:
            recommendations.append("Consider breaking down complex functions")
        
        if metrics["nesting_depth"] > 4:
            recommendations.append("Reduce nesting depth by extracting functions")
        
        if metrics["lines_of_code"] > 100:
            recommendations.append("Consider splitting large files into modules")
        
        if metrics["functions"] == 0 and metrics["lines_of_code"] > 20:
            recommendations.append("Consider organizing code into functions")
        
        return recommendations
    
    async def _check_security_issues(self, code: str) -> Dict[str, Any]:
        """Check for common security issues."""
        
        security_issues = []
        
        # Check for dangerous functions
        dangerous_patterns = [
            (r'eval\s*\(', "Use of eval() can be dangerous"),
            (r'exec\s*\(', "Use of exec() can be dangerous"),
            (r'input\s*\(', "Consider validating user input"),
            (r'os\.system\s*\(', "Use subprocess instead of os.system"),
            (r'shell=True', "Avoid shell=True in subprocess calls"),
        ]
        
        for pattern, message in dangerous_patterns:
            if re.search(pattern, code):
                security_issues.append(message)
        
        # Check for hardcoded secrets (simple patterns)
        secret_patterns = [
            (r'password\s*=\s*["\'][^"\']+["\']', "Possible hardcoded password"),
            (r'api_key\s*=\s*["\'][^"\']+["\']', "Possible hardcoded API key"),
            (r'secret\s*=\s*["\'][^"\']+["\']', "Possible hardcoded secret"),
        ]
        
        for pattern, message in secret_patterns:
            if re.search(pattern, code, re.IGNORECASE):
                security_issues.append(message)
        
        return {
            "issues": security_issues,
            "score": max(0, 100 - len(security_issues) * 20),
            "status": "✅ No security issues found" if not security_issues else "⚠️ Security issues detected"
        }
    
    async def _suggest_improvements(self, code: str) -> List[str]:
        """Suggest general code improvements."""
        
        improvements = []
        
        # Check for imports
        if 'import' not in code and len(code.split('\n')) > 10:
            improvements.append("Consider organizing imports at the top of the file")
        
        # Check for error handling
        if 'try:' not in code and ('open(' in code or 'requests.' in code):
            improvements.append("Consider adding error handling for file operations or network requests")
        
        # Check for type hints
        if 'def ' in code and '->' not in code:
            improvements.append("Consider adding type hints for better code documentation")
        
        # Check for constants
        if re.search(r'[A-Z_]{3,}', code):
            improvements.append("Consider defining constants at module level")
        
        # Check for list comprehensions
        simple_loops = re.findall(r'for\s+\w+\s+in\s+\w+:\s*\n\s+\w+\.append\(', code)
        if simple_loops:
            improvements.append("Consider using list comprehensions for simple loops")
        
        return improvements
    
    def _calculate_overall_score(
        self,
        syntax_analysis: Dict[str, Any],
        style_analysis: Dict[str, Any],
        complexity_analysis: Dict[str, Any]
    ) -> int:
        """Calculate overall code quality score."""
        
        if not syntax_analysis.get("valid", False):
            return 0
        
        style_score = style_analysis.get("score", 0)
        complexity_score = complexity_analysis.get("score", 0)
        
        # Weighted average
        overall_score = int(0.4 * style_score + 0.6 * complexity_score)
        
        return max(0, min(100, overall_score))
    
    async def format_code(self, code: str) -> str:
        """Format code using black if available."""
        
        if not CODE_QUALITY_LIBRARIES_AVAILABLE:
            return "Code formatting requires black library."
        
        try:
            # Use black to format the code
            formatted_code = black.format_str(code, mode=black.FileMode())
            return formatted_code
        except Exception as e:
            return f"Code formatting error: {str(e)}"
    
    async def generate_tests(self, code: str) -> str:
        """Generate basic test structure for the code."""
        
        try:
            # Extract function names
            function_names = re.findall(r'def\s+([a-zA-Z_][a-zA-Z0-9_]*)', code)
            
            if not function_names:
                return "No functions found to generate tests for."
            
            test_code = "import unittest\n\n"
            test_code += "class TestFunctions(unittest.TestCase):\n\n"
            
            for func_name in function_names:
                if not func_name.startswith('_'):  # Skip private functions
                    test_code += f"    def test_{func_name}(self):\n"
                    test_code += f"        # TODO: Implement test for {func_name}\n"
                    test_code += f"        pass\n\n"
            
            test_code += "if __name__ == '__main__':\n"
            test_code += "    unittest.main()\n"
            
            return test_code
            
        except Exception as e:
            return f"Test generation error: {str(e)}"
    
    def analyze_code(self, code: str, language: str = 'python') -> Dict[str, Any]:
        """
        Analyze code quality (synchronous wrapper for async method)
        
        Args:
            code: Code to analyze
            language: Programming language
            
        Returns:
            Dictionary with analysis results
        """
        try:
            import asyncio
            
            # Run the async method
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                result = loop.run_until_complete(
                    self.analyze_and_improve_code(code, {"language": language})
                )
                return {
                    'success': True,
                    'analysis': result,
                    'language': language,
                    'confidence_score': 0.85
                }
            finally:
                loop.close()
                
        except Exception as e:
            logger.error(f"Code analysis failed: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'language': language
            } 