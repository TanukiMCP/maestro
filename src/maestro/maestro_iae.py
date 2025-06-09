"""
MAESTRO IAE Integration Manager

Provides integration and management of Intelligence Amplification Engines (IAEs).
This is a backend-only, headless system designed to be called by external agentic IDEs.
No UI, no LLM client logic, just pure engine integration and management.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import importlib
import inspect
from pathlib import Path

from mcp.server.fastmcp import Context
from mcp.types import TextContent

from .iae_discovery import IAERegistry, IAEMetadata, IAECapability

logger = logging.getLogger(__name__)

@dataclass
class IAEIntegrationConfig:
    """Configuration for IAE integration."""
    engine_id: str
    enabled: bool = True
    max_concurrent_tasks: int = 10
    timeout_seconds: float = 30.0
    retry_attempts: int = 3
    retry_delay_seconds: float = 1.0
    validation_interval_seconds: float = 3600.0  # 1 hour

class IAEIntegrationManager:
    """
    Manages integration of Intelligence Amplification Engines.
    Handles engine lifecycle, task execution, and error recovery.
    """
    
    def __init__(self, registry: IAERegistry):
        self.registry = registry
        self._configs: Dict[str, IAEIntegrationConfig] = {}
        self._engine_instances: Dict[str, Any] = {}
        self._task_semaphores: Dict[str, asyncio.Semaphore] = {}
        self._validation_tasks: Dict[str, asyncio.Task] = {}
        
    async def initialize_engines(self) -> None:
        """Initialize all registered engines."""
        try:
            logger.info("🚀 Initializing IAE engines...")
            
            # Load engine configurations
            await self._load_configurations()
            
            # Initialize engines with configurations
            for engine_id, config in self._configs.items():
                if config.enabled:
                    await self._initialize_engine(engine_id, config)
            
            logger.info("✅ Engine initialization complete")
            
        except Exception as e:
            logger.error(f"Engine initialization failed: {e}")
            raise
    
    async def execute_task(
        self,
        engine_id: str,
        task_name: str,
        parameters: Dict[str, Any],
        context: Optional[Context] = None
    ) -> TextContent:
        """
        Execute a task on a specific engine.
        Args:
            engine_id: ID of the engine to use
            task_name: Name of the task/capability to execute
            parameters: Task parameters
            context: Optional MCP context
        """
        try:
            # Get engine instance and config
            engine = self._engine_instances.get(engine_id)
            config = self._configs.get(engine_id)
            
            if not engine or not config:
                return TextContent(
                    type="text",
                    text=f"# ❌ Task Execution Error\n\nEngine {engine_id} not found or not initialized"
                )
            
            # Get task semaphore
            semaphore = self._task_semaphores[engine_id]
            
            # Execute with concurrency control and retries
            async with semaphore:
                for attempt in range(config.retry_attempts):
                    try:
                        # Set timeout for task execution
                        result = await asyncio.wait_for(
                            self._execute_task_internal(engine, task_name, parameters, context),
                            timeout=config.timeout_seconds
                        )
                        
                        # Format successful result
                        return TextContent(
                            type="text",
                            text=f"# ✅ Task Complete\n\n{json.dumps(result, indent=2)}"
                        )
                        
                    except asyncio.TimeoutError:
                        if attempt < config.retry_attempts - 1:
                            logger.warning(
                                f"Task {task_name} timed out on attempt {attempt + 1}, retrying..."
                            )
                            await asyncio.sleep(config.retry_delay_seconds)
                        else:
                            return TextContent(
                                type="text",
                                text=f"# ❌ Task Timeout\n\nTask {task_name} timed out after {config.retry_attempts} attempts"
                            )
                            
                    except Exception as e:
                        if attempt < config.retry_attempts - 1:
                            logger.warning(
                                f"Task {task_name} failed on attempt {attempt + 1}: {e}, retrying..."
                            )
                            await asyncio.sleep(config.retry_delay_seconds)
                        else:
                            return TextContent(
                                type="text",
                                text=f"# ❌ Task Error\n\nTask {task_name} failed: {str(e)}"
                            )
            
        except Exception as e:
            logger.error(f"Task execution failed: {e}")
            return TextContent(
                type="text",
                text=f"# ❌ Task Error\n\nTask execution failed: {str(e)}"
            )
    
    async def _load_configurations(self) -> None:
        """Load engine configurations."""
        try:
            # Get engines directory path
            engines_dir = Path(__file__).parent.parent / 'engines'
            
            if not engines_dir.exists():
                logger.warning(f"Engines directory not found: {engines_dir}")
                return
            
            # Load each engine's configuration
            for file in engines_dir.glob('*.py'):
                if file.stem.startswith('__'):
                    continue
                    
                try:
                    # Create default configuration
                    engine_id = f"iae_{file.stem.lower()}"
                    config = IAEIntegrationConfig(engine_id=engine_id)
                    
                    # TODO: Load custom configuration from config file if exists
                    
                    self._configs[engine_id] = config
                    self._task_semaphores[engine_id] = asyncio.Semaphore(
                        config.max_concurrent_tasks
                    )
                    
                except Exception as e:
                    logger.error(f"Failed to load configuration for {file.stem}: {e}")
                    
        except Exception as e:
            logger.error(f"Configuration loading failed: {e}")
            raise
    
    async def _initialize_engine(
        self,
        engine_id: str,
        config: IAEIntegrationConfig
    ) -> None:
        """Initialize a specific engine."""
        try:
            # Import engine module
            module_name = engine_id.replace('iae_', '')
            module = importlib.import_module(f"engines.{module_name}")
            
            # Find engine class
            engine_classes = [
                obj for name, obj in inspect.getmembers(module)
                if inspect.isclass(obj) and 
                name.endswith('Engine') and 
                obj.__module__ == module.__name__
            ]
            
            if not engine_classes:
                raise ValueError(f"No engine class found in module {module_name}")
            
            # Instantiate engine
            engine_class = engine_classes[0]
            engine = engine_class()
            
            # Store engine instance
            self._engine_instances[engine_id] = engine
            
            # Start validation task
            self._validation_tasks[engine_id] = asyncio.create_task(
                self._validate_engine_periodically(engine_id, config)
            )
            
            logger.info(f"✅ Initialized engine: {engine_id}")
            
        except Exception as e:
            logger.error(f"Engine initialization failed for {engine_id}: {e}")
            raise
    
    async def _execute_task_internal(
        self,
        engine: Any,
        task_name: str,
        parameters: Dict[str, Any],
        context: Optional[Context]
    ) -> Any:
        """Internal task execution with proper error handling."""
        try:
            # Get task method
            method = getattr(engine, task_name, None)
            if not method:
                raise ValueError(f"Task {task_name} not found in engine")
            
            # Execute task
            if context:
                parameters['context'] = context
            
            if asyncio.iscoroutinefunction(method):
                result = await method(**parameters)
            else:
                result = method(**parameters)
            
            return result
            
        except Exception as e:
            logger.error(f"Task execution failed: {e}")
            raise
    
    async def _validate_engine_periodically(
        self,
        engine_id: str,
        config: IAEIntegrationConfig
    ) -> None:
        """Periodically validate engine health."""
        while True:
            try:
                # Get engine instance
                engine = self._engine_instances.get(engine_id)
                if not engine:
                    logger.error(f"Engine {engine_id} not found")
                    return
                
                # Perform validation
                # TODO: Implement actual validation logic
                
                # Update engine metadata
                metadata = await self.registry._validate_engine(engine)
                if metadata:
                    self.registry._engines[engine_id] = metadata
                
                # Wait for next validation
                await asyncio.sleep(config.validation_interval_seconds)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Engine validation failed for {engine_id}: {e}")
                await asyncio.sleep(config.validation_interval_seconds) 