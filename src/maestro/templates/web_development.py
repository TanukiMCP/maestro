"""
Web Development Workflow Template
"""

WEB_DEVELOPMENT_TEMPLATE = {
    "template_name": "web_development",
    "description": "Template for web application development including frontend, backend, and full-stack",
    
    "task_analysis_prompts": {
        "complexity_indicators": [
            "frontend and backend components",
            "database integration",
            "user authentication",
            "responsive design",
            "API development"
        ],
        "capability_requirements": [
            "web_development",
            "frontend",
            "backend", 
            "testing",
            "visual_verification"
        ]
    },
    
    "system_prompt_guidance": {
        "role": "Expert Full-Stack Web Developer",
        "expertise": [
            "Modern web frameworks",
            "Responsive design",
            "API development",
            "Database design",
            "Security best practices"
        ],
        "approach": [
            "Create modern, responsive user interfaces",
            "Implement secure backend APIs",
            "Follow web accessibility standards",
            "Use proper error handling and validation",
            "Ensure cross-browser compatibility"
        ],
        "quality_standards": [
            "Code passes accessibility checks",
            "Responsive across device sizes",
            "Security vulnerabilities addressed",
            "Performance optimized",
            "User experience tested"
        ]
    },    
    "execution_phases": [
        {
            "phase": "Planning",
            "description": "Plan application architecture and design",
            "success_criteria": [
                "Requirements analyzed",
                "Architecture designed",
                "Technology stack selected",
                "Database schema planned"
            ],
            "suggested_tools": ["plan_architecture", "design_database"]
        },
        {
            "phase": "Backend_Development",
            "description": "Implement server-side functionality",
            "success_criteria": [
                "API endpoints created",
                "Database models implemented",
                "Authentication system built",
                "Error handling added"
            ],
            "suggested_tools": ["create_api", "implement_auth", "setup_database"]
        },
        {
            "phase": "Frontend_Development", 
            "description": "Create user interface",
            "success_criteria": [
                "UI components built",
                "Responsive design implemented",
                "User interactions working",
                "API integration complete"
            ],
            "suggested_tools": ["create_ui", "implement_responsive_design"]
        },
        {
            "phase": "Testing_Integration",
            "description": "Test and validate full application",
            "success_criteria": [
                "Unit tests passing",
                "Integration tests working",
                "Accessibility validated",
                "Performance acceptable"
            ],
            "suggested_tools": ["run_tests", "validate_accessibility", "test_performance"]
        }
    ],
    
    "verification_methods": [
        "visual_verification",
        "accessibility_verification", 
        "automated_testing"
    ]
}