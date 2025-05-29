"""
Code Development Workflow Template
"""

CODE_DEVELOPMENT_TEMPLATE = {
    "template_name": "code_development",
    "description": "Template for code development tasks including functions, classes, and scripts",
    
    "task_analysis_prompts": {
        "complexity_indicators": [
            "multiple functions or classes",
            "advanced algorithms", 
            "performance optimization",
            "testing requirements",
            "documentation needs"
        ],
        "capability_requirements": [
            "code_generation",
            "testing", 
            "quality_analysis",
            "documentation"
        ]
    },
    
    "system_prompt_guidance": {
        "role": "Expert Software Developer",
        "expertise": [
            "Clean code principles",
            "Test-driven development", 
            "Code documentation",
            "Error handling",
            "Performance optimization"
        ],
        "approach": [
            "Write production-ready code with comprehensive error handling",
            "Include type hints and docstrings for all functions",
            "Create unit tests with edge case coverage",
            "Follow PEP 8 style guidelines",
            "Provide clear examples and usage documentation"
        ],
        "quality_standards": [
            "Code must be executable and functional",
            "Include comprehensive error handling",
            "Achieve 90%+ test coverage",
            "Pass all linting checks",
            "Include clear documentation"
        ]
    },    
    "execution_phases": [
        {
            "phase": "Analysis",
            "description": "Analyze requirements and plan implementation",
            "success_criteria": [
                "Requirements clearly understood",
                "Function/class structure planned",
                "Test cases identified",
                "Edge cases considered"
            ],
            "suggested_tools": ["analyze_requirements", "plan_structure"]
        },
        {
            "phase": "Implementation", 
            "description": "Write the core functionality",
            "success_criteria": [
                "Core logic implemented",
                "Error handling included",
                "Type hints added", 
                "Docstrings written"
            ],
            "suggested_tools": ["write_code", "add_documentation"]
        },
        {
            "phase": "Testing",
            "description": "Create and run comprehensive tests",
            "success_criteria": [
                "Unit tests written",
                "Edge cases covered",
                "All tests passing",
                "Good test coverage"
            ],
            "suggested_tools": ["write_tests", "run_tests", "check_coverage"]
        },
        {
            "phase": "Quality_Assurance",
            "description": "Verify code quality and standards",
            "success_criteria": [
                "Code passes linting",
                "Follows style guidelines",
                "No security issues",
                "Performance acceptable"
            ],
            "suggested_tools": ["run_linter", "security_check", "performance_test"]
        },
        {
            "phase": "Documentation",
            "description": "Create comprehensive documentation",
            "success_criteria": [
                "README created",
                "Usage examples provided",
                "API documentation complete",
                "Installation instructions clear"
            ],
            "suggested_tools": ["write_readme", "create_examples"]
        }
    ],    
    "verification_methods": [
        "automated_testing",
        "code_quality_verification",
        "documentation_review"
    ],
    
    "common_patterns": {
        "function_template": {
            "structure": [
                "Type hints for parameters and return",
                "Comprehensive docstring with examples",
                "Input validation",
                "Core logic implementation", 
                "Error handling",
                "Logging where appropriate"
            ]
        },
        "test_template": {
            "structure": [
                "Test basic functionality",
                "Test edge cases",
                "Test error conditions",
                "Test performance if relevant",
                "Test with various input types"
            ]
        }
    }
}