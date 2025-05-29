"""
MAESTRO Protocol Workflow Templates

Modular templates for different workflow types that can be imported
and used by the orchestrator for consistent workflow planning.
"""

from .code_development import CODE_DEVELOPMENT_TEMPLATE
from .web_development import WEB_DEVELOPMENT_TEMPLATE
from .data_analysis import DATA_ANALYSIS_TEMPLATE
from .mathematical_analysis import MATHEMATICAL_ANALYSIS_TEMPLATE
from .research_analysis import RESEARCH_ANALYSIS_TEMPLATE
from .documentation import DOCUMENTATION_TEMPLATE

__all__ = [
    "CODE_DEVELOPMENT_TEMPLATE",
    "WEB_DEVELOPMENT_TEMPLATE", 
    "DATA_ANALYSIS_TEMPLATE",
    "MATHEMATICAL_ANALYSIS_TEMPLATE",
    "RESEARCH_ANALYSIS_TEMPLATE",
    "DOCUMENTATION_TEMPLATE"
]

# Template registry for dynamic loading
TEMPLATE_REGISTRY = {
    "code_development": CODE_DEVELOPMENT_TEMPLATE,
    "web_development": WEB_DEVELOPMENT_TEMPLATE,
    "data_analysis": DATA_ANALYSIS_TEMPLATE,
    "mathematical_analysis": MATHEMATICAL_ANALYSIS_TEMPLATE,
    "research_analysis": RESEARCH_ANALYSIS_TEMPLATE,
    "documentation": DOCUMENTATION_TEMPLATE,
}

def get_template(template_name: str) -> dict:
    """Get a workflow template by name."""
    return TEMPLATE_REGISTRY.get(template_name, {})

def list_available_templates() -> list:
    """List all available workflow templates."""
    return list(TEMPLATE_REGISTRY.keys())