"""
Dependency Injection Providers for the Maestro MCP Server

This module contains dependency provider functions that are used by the
application to inject objects like configuration and clients into the request
handling lifecycle. This is key to supporting per-request configuration and
lazy loading of resources.
"""

import base64
import json
import logging
from functools import lru_cache
from typing import Optional

from starlette.requests import Request

from .config import MAESTROConfig

logger = logging.getLogger(__name__)

@lru_cache()
def get_config_from_env() -> MAESTROConfig:
    """
    Cached function to load configuration from environment variables.
    The lru_cache ensures this is only done once.
    """
    logger.info("Loading configuration from environment variables.")
    return MAESTROConfig.from_env()

def get_config(request: Optional[Request] = None) -> MAESTROConfig:
    """
    Dependency to provide MAESTROConfig.
    
    For Smithery deployments, it parses dot-notation query parameters
    (e.g., server.host=localhost&server.port=8080) into a nested config object.
    
    For the legacy base64 config format, it decodes the 'config' parameter.
    
    If no request config is found, it falls back to loading from environment variables.
    
    This approach provides flexibility for Smithery's deployment model,
    legacy configurations, and local environment-based development.
    
    Returns a minimal default config if called during tool scanning to prevent timeouts.
    """
    # Fast path for tool scanning - return minimal config to prevent timeout
    if request is None:
        try:
            return get_config_from_env()
        except Exception:
            # If environment config fails, return minimal default for tool scanning
            logger.warning("Using minimal default configuration for tool scanning compatibility")
            return get_minimal_config()
    
    # Rest of existing logic for actual requests...
    if request:
        # Check for Smithery's dot-notation query parameters first
        config_data = {}
        has_config_params = False
        
        for key, value in request.query_params.items():
            if '.' in key:
                # Parse dot-notation (e.g., "server.host" -> {"server": {"host": value}})
                parts = key.split('.')
                current = config_data
                for part in parts[:-1]:
                    if part not in current:
                        current[part] = {}
                    current = current[part]
                
                # Type conversion for known parameter types
                final_value = value
                if key.endswith('.port') or key.endswith('.workers') or key.endswith('.timeout') or \
                   key.endswith('.rate_limit_requests') or key.endswith('.rate_limit_window') or \
                   key.endswith('.max_concurrent_tasks') or key.endswith('.task_timeout') or \
                   key.endswith('.memory_limit') or key.endswith('.rotation_size') or key.endswith('.retention_days'):
                    try:
                        final_value = int(value)
                    except ValueError:
                        logger.warning(f"Could not convert {key}={value} to integer, using string")
                elif key.endswith('.api_key_required') or key.endswith('.rate_limit_enabled') or \
                     key.endswith('.file_enabled') or key.endswith('.enable_gpu'):
                    final_value = str(value).lower() in ('true', '1', 'yes', 'on')
                elif key.endswith('.cors_origins') or key.endswith('.allowed_origins'):
                    # Handle comma-separated lists
                    if isinstance(value, str) and ',' in value:
                        final_value = value.split(',')
                    else:
                        final_value = value
                
                current[parts[-1]] = final_value
                has_config_params = True
                
        if has_config_params:
            logger.info("Successfully parsed configuration from Smithery dot-notation query parameters.")
            return MAESTROConfig.from_dict(config_data)
            
        # Check for legacy base64-encoded config parameter
        if "config" in request.query_params:
            config_b64 = request.query_params["config"]
            logger.debug("Found legacy 'config' in query parameters.")
            try:
                config_json = base64.b64decode(config_b64).decode('utf-8')
                config_data = json.loads(config_json)
                logger.info("Successfully decoded legacy configuration from query parameters.")
                return MAESTROConfig.from_dict(config_data)
            except (json.JSONDecodeError, UnicodeDecodeError, ValueError) as e:
                logger.error(
                    f"Failed to decode/parse legacy config from query parameter: {e}. "
                    "Falling back to environment configuration."
                )
                return get_config_from_env()

    # Fallback for local development or non-request contexts
    return get_config_from_env()

def get_minimal_config() -> MAESTROConfig:
    """
    Provides a minimal default configuration for tool scanning compatibility.
    This prevents timeouts during Smithery tool discovery.
    """
    from .config import ServerConfig, SecurityConfig, EngineConfig, LoggingConfig, MAESTROConfig
    from .config import EngineMode, LogLevel
    
    return MAESTROConfig(
        server=ServerConfig(
            host="0.0.0.0",
            port=8000,
            workers=1,
            timeout=30,
            cors_origins=["*"]
        ),
        security=SecurityConfig(
            api_key_required=False,
            allowed_origins=["*"],
            rate_limit_enabled=False,
            rate_limit_requests=100,
            rate_limit_window=60
        ),
        engine=EngineConfig(
            mode=EngineMode.PRODUCTION,
            max_concurrent_tasks=1,
            task_timeout=60,
            memory_limit=512,
            enable_gpu=False
        ),
        logging=LoggingConfig(
            level=LogLevel.INFO,
            file_enabled=False,
            file_path=None,
            rotation_size=100,
            retention_days=30
        )
    ) 