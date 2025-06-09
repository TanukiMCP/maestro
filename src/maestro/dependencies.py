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
    
    It first checks for a base64-encoded config in the request's query
    parameters ('config'). If found, it decodes and uses it.
    
    If not found or if the request object is not available (e.g., in a
    non-request context), it falls back to loading the config from
    environment variables.
    
    This approach provides flexibility for both Smithery's per-request
    config model and local environment-based configuration.
    """
    if request and "config" in request.query_params:
        config_b64 = request.query_params["config"]
        logger.debug("Found 'config' in query parameters.")
        try:
            config_json = base64.b64decode(config_b64).decode('utf-8')
            config_data = json.loads(config_json)
            logger.info("Successfully decoded configuration from query parameters.")
            return MAESTROConfig.from_dict(config_data)
        except (json.JSONDecodeError, UnicodeDecodeError, ValueError) as e:
            logger.error(
                f"Failed to decode/parse config from query parameter: {e}. "
                "Falling back to environment configuration."
            )
            return get_config_from_env()

    # Fallback for local development or non-request contexts
    return get_config_from_env() 