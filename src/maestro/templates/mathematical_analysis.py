"""
Mathematical Analysis Workflow Template
"""

MATHEMATICAL_ANALYSIS_TEMPLATE = {
    "template_name": "mathematical_analysis",
    "description": "Template for mathematical computation, analysis, and proof tasks",
    
    "task_analysis_prompts": {
        "complexity_indicators": [
            "symbolic computation required",
            "numerical analysis needed", 
            "statistical calculations",
            "proof verification",
            "visualization requirements"
        ],
        "capability_requirements": [
            "mathematics",
            "symbolic_computation",
            "numerical_analysis",
            "visualization"
        ]
    },
    
    "system_prompt_guidance": {
        "role": "Expert Mathematician and Computational Analyst",
        "expertise": [
            "Symbolic mathematics",
            "Numerical computation",
            "Statistical analysis",
            "Mathematical proofs",
            "Data visualization"
        ],
        "approach": [
            "Use appropriate mathematical libraries (SymPy, NumPy, SciPy)",
            "Verify calculations through multiple methods when possible",
            "Provide clear mathematical explanations",
            "Include error bounds and limitations",
            "Create visualizations to illustrate concepts"
        ],
        "quality_standards": [
            "Mathematical accuracy verified",
            "Calculations reproducible",
            "Proper error handling for edge cases",
            "Clear documentation of methods",
            "Appropriate precision maintained"
        ]
    },    
    "execution_phases": [
        {
            "phase": "Problem_Analysis",
            "description": "Understand mathematical requirements and approach",
            "success_criteria": [
                "Problem clearly defined",
                "Mathematical approach identified",
                "Required tools/libraries determined",
                "Expected output format specified"
            ],
            "suggested_tools": ["analyze_mathematical_problem", "select_approach"]
        },
        {
            "phase": "Implementation",
            "description": "Implement mathematical solution",
            "success_criteria": [
                "Mathematical code written",
                "Appropriate libraries used",
                "Edge cases handled",
                "Results computed accurately"
            ],
            "suggested_tools": ["implement_solution", "verify_accuracy"]
        },
        {
            "phase": "Verification",
            "description": "Verify mathematical accuracy and correctness",
            "success_criteria": [
                "Results verified through alternate methods",
                "Edge cases tested",
                "Precision validated",
                "Error bounds established"
            ],
            "suggested_tools": ["verify_results", "cross_check_calculations"]
        }
    ],
    
    "verification_methods": [
        "mathematical_verification",
        "numerical_validation"
    ],
    
    "common_patterns": {
        "symbolic_computation": ["SymPy for algebraic manipulation"],
        "numerical_analysis": ["NumPy/SciPy for computations"],
        "visualization": ["Matplotlib/Plotly for graphs"]
    }
}