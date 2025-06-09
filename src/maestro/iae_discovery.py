# Copyright (c) 2025 TanukiMCP Orchestra
# Licensed under Non-Commercial License - Commercial use requires approval from TanukiMCP
# Contact tanukimcp@gmail.com for commercial licensing inquiries

"""
MAESTRO IAE Discovery System

Provides dynamic discovery and registry of Intelligence Amplification Engines (IAEs).
This is a backend-only, headless system designed to be called by external agentic IDEs.
No UI, no LLM client logic, just pure engine discovery and registry management.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
import importlib
import inspect
import os
import sys
from pathlib import Path

from mcp.server.fastmcp import Context
from mcp.types import TextContent

logger = logging.getLogger(__name__)

@dataclass
class IAECapability:
    """Represents a single capability of an IAE."""
    name: str
    description: str
    input_types: List[str]
    output_types: List[str]
    performance_characteristics: Dict[str, Any]
    resource_requirements: Dict[str, Any]
    validation_method: Optional[str] = None
    validation_criteria: Optional[Dict[str, Any]] = None

@dataclass
class IAEMetadata:
    """Metadata for an Intelligence Amplification Engine."""
    engine_id: str
    name: str
    version: str
    description: str
    capabilities: List[IAECapability]
    supported_domains: List[str]
    enhancement_types: List[str]
    last_validated: datetime
    is_active: bool = True
    validation_status: Dict[str, Any] = field(default_factory=dict)
    error_history: List[Dict[str, Any]] = field(default_factory=list)

class IAERegistry:
    """
    Dynamic registry for Intelligence Amplification Engines.
    Supports runtime addition/removal and real-time discovery.
    """
    
    def __init__(self):
        self._engines: Dict[str, IAEMetadata] = {}
        self._last_discovery: Optional[datetime] = None
        self._discovery_cache_ttl = 300  # 5 minutes
        self._lock = asyncio.Lock()
        
    async def discover_engines(self, force: bool = False) -> List[IAEMetadata]:
        """
        Dynamically discover available IAEs in the environment.
        Args:
            force: If True, bypass cache and force rediscovery
        """
        async with self._lock:
            now = datetime.now(timezone.utc)
            
            # Check cache unless forced
            if not force and self._last_discovery and \
               (now - self._last_discovery).total_seconds() < self._discovery_cache_ttl:
                return list(self._engines.values())
                
            logger.info("🔍 Starting dynamic IAE discovery...")
            
            # Clear existing registry if forced
            if force:
                self._engines.clear()
            
            try:
                # Initialize the integration manager for built-in engines
                from .maestro_iae import IAEIntegrationManager
                integration_manager = IAEIntegrationManager(self)
                await integration_manager.initialize_engines()
                
                # Discover MCP-compatible engines
                mcp_engines = await self._discover_mcp_engines()
                self._engines.update({e.engine_id: e for e in mcp_engines})
                
                # Discover plugin engines (if any)
                plugin_engines = await self._discover_plugin_engines()
                self._engines.update({e.engine_id: e for e in plugin_engines})
                
                self._last_discovery = now
                logger.info(f"✅ Discovered {len(self._engines)} IAEs")
                
                return list(self._engines.values())
                
            except Exception as e:
                logger.error(f"Engine discovery failed: {e}")
                raise
    
    async def _discover_mcp_engines(self) -> List[IAEMetadata]:
        """Discover MCP-compatible engines."""
        engines = []
        
        try:
            # Get the engines directory path
            engines_dir = Path(__file__).parent.parent / 'engines'
            
            if not engines_dir.exists():
                logger.warning(f"Engines directory not found: {engines_dir}")
                return engines
            
            # Import all engine modules
            sys.path.insert(0, str(engines_dir.parent))
            
            for file in engines_dir.glob('*.py'):
                if file.stem.startswith('__'):
                    continue
                    
                try:
                    module = importlib.import_module(f"engines.{file.stem}")
                    
                    # Find engine classes
                    engine_classes = [
                        obj for name, obj in inspect.getmembers(module)
                        if inspect.isclass(obj) and 
                        name.endswith('Engine') and 
                        obj.__module__ == module.__name__
                    ]
                    
                    for engine_class in engine_classes:
                        try:
                            # Instantiate and validate engine
                            engine = engine_class()
                            metadata = await self._validate_engine(engine)
                            if metadata:
                                engines.append(metadata)
                        except Exception as e:
                            logger.error(f"Failed to validate engine {engine_class.__name__}: {e}")
                            
                except ImportError as e:
                    logger.error(f"Failed to import engine module {file.stem}: {e}")
                except Exception as e:
                    logger.error(f"Error processing engine module {file.stem}: {e}")
            
            sys.path.pop(0)
            
        except Exception as e:
            logger.error(f"MCP engine discovery failed: {e}")
            
        return engines
    
    async def _discover_plugin_engines(self) -> List[IAEMetadata]:
        """Discover plugin engines from external sources."""
        # TODO: Implement plugin discovery when needed
        return []
    
    async def _validate_engine(self, engine: Any) -> Optional[IAEMetadata]:
        """Validate an engine instance and create metadata."""
        try:
            # Basic validation
            if not hasattr(engine, 'name') or not hasattr(engine, 'version'):
                logger.warning(f"Engine {engine.__class__.__name__} missing required attributes")
                return None
            
            # Get capabilities
            capabilities = []
            for name, method in inspect.getmembers(engine, inspect.ismethod):
                if name.startswith('_'):
                    continue
                    
                try:
                    sig = inspect.signature(method)
                    doc = inspect.getdoc(method) or "No description available"
                    
                    capability = IAECapability(
                        name=name,
                        description=doc,
                        input_types=[
                            str(param.annotation) 
                            for param in sig.parameters.values()
                        ],
                        output_types=[str(sig.return_annotation)],
                        performance_characteristics=self._get_performance_characteristics(method),
                        resource_requirements=self._get_resource_requirements(method)
                    )
                    capabilities.append(capability)
                except Exception as e:
                    logger.warning(f"Failed to process capability {name}: {e}")
            
            # Create metadata
            metadata = IAEMetadata(
                engine_id=f"iae_{engine.__class__.__name__.lower()}",
                name=engine.name,
                version=engine.version,
                description=engine.__doc__ or "No description available",
                capabilities=capabilities,
                supported_domains=self._infer_domains(engine.__class__),
                enhancement_types=self._infer_enhancement_types(engine),
                last_validated=datetime.now(timezone.utc)
            )
            
            return metadata
            
        except Exception as e:
            logger.error(f"Engine validation failed: {e}")
            return None
    
    def _get_performance_characteristics(self, method: Any) -> Dict[str, Any]:
        """Extract performance characteristics from method."""
        # TODO: Implement actual performance profiling
        return {
            "accuracy": 0.95,
            "latency": "medium",
            "resource_usage": "medium"
        }
    
    def _get_resource_requirements(self, method: Any) -> Dict[str, Any]:
        """Extract resource requirements from method."""
        # TODO: Implement actual resource profiling
        return {
            "memory": "medium",
            "cpu": "medium",
            "gpu": "none"
        }
    
    def _infer_domains(self, engine_class: type) -> List[str]:
        """Infer the domains an engine supports from its name and docstring."""
        domains = []
        name = engine_class.__name__.lower()
        doc = engine_class.__doc__ or ""
        
        domain_keywords = {
            'math': 'mathematics',
            'physics': 'physics',
            'quantum': 'quantum_physics',
            'astro': 'astrophysics',
            'bio': 'biology',
            'medical': 'medicine',
            'language': 'linguistics',
            'grammar': 'linguistics',
            'intelligence': 'cognitive_science',
            'data': 'data_science',
            'code': 'software_engineering',
            'scientific': 'science'
        }
        
        for keyword, domain in domain_keywords.items():
            if keyword in name or keyword in doc.lower():
                domains.append(domain)
                
        if not domains:
            domains = ['general']
            
        return domains
    
    def _infer_enhancement_types(self, engine: Any) -> List[str]:
        """Infer enhancement types from engine capabilities."""
        types = set()
        
        method_keywords = {
            'analyze': 'analysis',
            'optimize': 'optimization',
            'validate': 'validation',
            'reason': 'reasoning',
            'compute': 'computation',
            'predict': 'prediction',
            'cluster': 'clustering',
            'classify': 'classification'
        }
        
        for name, _ in inspect.getmembers(engine, inspect.ismethod):
            if name.startswith('_'):
                continue
                
            for keyword, enhancement_type in method_keywords.items():
                if keyword in name:
                    types.add(enhancement_type)
        
        return list(types) if types else ['general']

class IAEDiscovery:
    """
    Main entry point for IAE discovery functionality.
    This is a backend-only, headless service designed to be called by external agentic IDEs.
    """
    
    def __init__(self):
        self.registry = IAERegistry()
        
    async def discover_engines_for_task(
        self,
        task_type: str,
        domain_context: Optional[str] = None,
        complexity_requirements: Optional[Dict[str, Any]] = None
    ) -> List[TextContent]:
        """
        Discover and rank engines suitable for a specific task.
        Args:
            task_type: Type of task to be performed
            domain_context: Optional domain context for filtering
            complexity_requirements: Optional dict of complexity requirements
        """
        try:
            # Discover all available engines
            all_engines = await self.registry.discover_engines()
            
            # Filter by domain if provided
            if domain_context:
                engines = [e for e in all_engines if domain_context in e.supported_domains]
            else:
                engines = all_engines
            
            # Sort engines by relevance to task
            ranked_engines = self._sort_engines_by_task_relevance(
                task_type=task_type,
                engines=engines,
                complexity_requirements=complexity_requirements
            )
            
            # Format results as MCP text content
            results = []
            
            # Overall summary
            summary = TextContent(
                type="text",
                text=f"# 🔍 IAE Discovery Results\n\nFound {len(ranked_engines)} relevant engines for task type: {task_type}\n"
            )
            results.append(summary)
            
            # Detailed engine information
            for engine, score in ranked_engines:
                details = [
                    f"## {engine.name} (Score: {score:.2f})",
                    f"**ID:** `{engine.id}`",
                    f"**Version:** {engine.version}",
                    f"**Description:** {engine.description}",
                    "\n**Capabilities:**"
                ]
                
                for cap in engine.capabilities:
                    details.extend([
                        f"- {cap.name}",
                        f"  - Description: {cap.description}",
                        f"  - Input Types: {', '.join(cap.input_types)}",
                        f"  - Output Types: {', '.join(cap.output_types)}"
                    ])
                
                details.extend([
                    f"\n**Domains:** {', '.join(engine.supported_domains)}",
                    f"**Enhancement Types:** {', '.join(engine.enhancement_types)}",
                    "---\n"
                ])
                
                results.append(TextContent(
                    type="text",
                    text="\n".join(details)
                ))
            
            return results
            
        except Exception as e:
            logger.error(f"Engine discovery failed: {e}")
            return [TextContent(
                type="text",
                text=f"# ❌ Discovery Error\n\nFailed to discover engines: {str(e)}"
            )]
    
    def _sort_engines_by_task_relevance(
        self,
        task_type: str,
        engines: List[IAEMetadata],
        complexity_requirements: Optional[Dict[str, Any]] = None
    ) -> List[tuple[IAEMetadata, float]]:
        """Sort engines by their relevance to the task."""
        scored_engines = [
            (engine, self._calculate_engine_relevance(engine, task_type, complexity_requirements))
            for engine in engines
        ]
        return sorted(scored_engines, key=lambda x: x[1], reverse=True)
    
    def _calculate_engine_relevance(
        self,
        engine: IAEMetadata,
        task_type: str,
        complexity_requirements: Optional[Dict[str, Any]] = None
    ) -> float:
        """Calculate how relevant an engine is for a specific task."""
        score = 0.0
        
        # Check if engine capabilities match task type
        for capability in engine.capabilities:
            if task_type.lower() in capability.name.lower():
                score += 1.0
                
        # Check complexity requirements if provided
        if complexity_requirements:
            for cap in engine.capabilities:
                if all(
                    cap.performance_characteristics.get(k, 0) >= v 
                    for k, v in complexity_requirements.items()
                ):
                    score += 0.5
                    
        # Boost score for active and recently validated engines
        if engine.is_active:
            score *= 1.2
            
        # Normalize to 0-1 range
        return min(score, 1.0) 