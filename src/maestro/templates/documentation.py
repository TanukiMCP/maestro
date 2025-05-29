"""
Documentation Workflow Template
"""

DOCUMENTATION_TEMPLATE = {
    "template_name": "documentation",
    "description": "Template for creating comprehensive technical and user documentation",
    
    "system_prompt_guidance": {
        "role": "Expert Technical Writer",
        "expertise": ["Technical writing", "Documentation design", "User experience", "Information architecture"],
        "approach": [
            "Write clear, concise documentation", 
            "Structure information logically",
            "Include practical examples",
            "Consider different user personas"
        ]
    },
    
    "execution_phases": [
        {
            "phase": "Content_Planning",
            "description": "Plan documentation structure and content",
            "success_criteria": ["Structure outlined", "Content planned", "Audience identified"],
            "suggested_tools": ["plan_structure", "identify_audience", "outline_content"]
        },
        {
            "phase": "Content_Creation",
            "description": "Write comprehensive documentation",
            "success_criteria": ["Content written", "Examples included", "Formatting applied"],
            "suggested_tools": ["write_content", "add_examples", "format_documentation"]
        },
        {
            "phase": "Review_Polish",
            "description": "Review and polish documentation quality",
            "success_criteria": ["Grammar checked", "Clarity verified", "Completeness confirmed"],
            "suggested_tools": ["check_grammar", "verify_clarity", "validate_completeness"]
        }
    ],
    
    "verification_methods": ["language_quality_verification", "completeness_check"]
}