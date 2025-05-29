"""
Data Analysis Workflow Template
"""

DATA_ANALYSIS_TEMPLATE = {
    "template_name": "data_analysis",
    "description": "Template for data processing, analysis, and visualization tasks",
    
    "system_prompt_guidance": {
        "role": "Expert Data Analyst",
        "expertise": ["Data processing", "Statistical analysis", "Visualization", "Machine learning"],
        "approach": [
            "Clean and validate data thoroughly",
            "Apply appropriate statistical methods",
            "Create clear, informative visualizations",
            "Document analysis methodology"
        ]
    },
    
    "execution_phases": [
        {
            "phase": "Data_Exploration",
            "description": "Load and explore the dataset",
            "success_criteria": ["Data loaded successfully", "Basic statistics computed", "Missing values identified"],
            "suggested_tools": ["load_data", "explore_data", "compute_statistics"]
        },
        {
            "phase": "Analysis",
            "description": "Perform statistical analysis",
            "success_criteria": ["Analysis complete", "Insights identified", "Results validated"],
            "suggested_tools": ["analyze_data", "statistical_tests", "validate_results"]
        },
        {
            "phase": "Visualization",
            "description": "Create visual representations",
            "success_criteria": ["Charts created", "Visualizations clear", "Insights highlighted"],
            "suggested_tools": ["create_charts", "plot_data", "generate_report"]
        }
    ],
    
    "verification_methods": ["automated_testing", "visual_verification"]
}