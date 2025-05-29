"""
Research Analysis Workflow Template
"""

RESEARCH_ANALYSIS_TEMPLATE = {
    "template_name": "research_analysis", 
    "description": "Template for research, investigation, and analytical writing tasks",
    
    "system_prompt_guidance": {
        "role": "Expert Researcher and Analyst",
        "expertise": ["Research methodology", "Critical analysis", "Academic writing", "Source evaluation"],
        "approach": [
            "Use credible, peer-reviewed sources",
            "Apply systematic research methodology", 
            "Provide balanced, objective analysis",
            "Document sources and citations properly"
        ]
    },
    
    "execution_phases": [
        {
            "phase": "Research_Planning",
            "description": "Define research scope and methodology",
            "success_criteria": ["Research questions defined", "Methodology planned", "Sources identified"],
            "suggested_tools": ["define_scope", "plan_methodology", "identify_sources"]
        },
        {
            "phase": "Information_Gathering",
            "description": "Collect and evaluate information",
            "success_criteria": ["Sources collected", "Information extracted", "Credibility assessed"],
            "suggested_tools": ["gather_information", "evaluate_sources", "extract_data"]
        },
        {
            "phase": "Analysis_Synthesis",
            "description": "Analyze findings and synthesize conclusions",
            "success_criteria": ["Analysis complete", "Patterns identified", "Conclusions drawn"],
            "suggested_tools": ["analyze_findings", "identify_patterns", "synthesize_conclusions"]
        }
    ],
    
    "verification_methods": ["language_quality_verification", "fact_checking"]
}