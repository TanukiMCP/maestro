"""
MAESTRO API Router

Provides FastAPI router for all MAESTRO MCP server endpoints.
This is a backend-only, headless API designed to be called by external agentic IDEs.
"""

import logging
from typing import Dict, Any, Optional
from fastapi import APIRouter, Request, Depends, HTTPException, Header
from fastapi.responses import JSONResponse
from fastapi.security import APIKeyHeader
from datetime import datetime

from .config import MAESTROConfig
from .orchestrator import MAESTROOrchestrator
from .iae_discovery import IAEDiscovery
from .schemas import (
    OrchestrationRequest,
    OrchestrationResponse,
    DiscoveryRequest,
    DiscoveryResponse,
    ToolRequest,
    ToolResponse,
    HealthResponse
)

logger = logging.getLogger(__name__)

# Create API router
router = APIRouter()

# API key security scheme
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

class MAESTROAPIRouter:
    """
    Main API router for the MAESTRO MCP server.
    Handles all HTTP endpoints and integrates with the orchestrator.
    """
    
    def __init__(self, config: MAESTROConfig):
        self.config = config
        self.discovery = IAEDiscovery()
        
    async def verify_api_key(self, api_key: Optional[str] = Depends(api_key_header)) -> bool:
        """Verify API key if required."""
        if not self.config.security.api_key_required:
            return True
            
        if not api_key:
            raise HTTPException(
                status_code=401,
                detail="API key required"
            )
            
        # In a real implementation, verify against secure storage
        valid_key = "your-api-key"  # This should come from secure storage
        if api_key != valid_key:
            raise HTTPException(
                status_code=401,
                detail="Invalid API key"
            )
            
        return True
        
    async def get_orchestrator(self, request: Request) -> MAESTROOrchestrator:
        """Get or create orchestrator instance with LLM client from request."""
        # The LLM client should be provided by the external IDE
        llm_client = request.state.llm_client
        if not llm_client:
            raise HTTPException(
                status_code=400,
                detail="LLM client not provided"
            )
            
        return MAESTROOrchestrator(llm_client=llm_client)
    
    @router.post("/orchestrate", response_model=OrchestrationResponse)
    async def orchestrate_task(
        self,
        request: OrchestrationRequest,
        api_key_valid: bool = Depends(verify_api_key),
        orchestrator: MAESTROOrchestrator = Depends(get_orchestrator)
    ) -> JSONResponse:
        """
        Orchestrate a complex task using the MAESTRO system.
        This endpoint requires an external LLM client to be provided.
        """
        try:
            result = await orchestrator.orchestrate_task(
                task_description=request.task_description,
                context=request.context
            )
            
            return JSONResponse(content=result)
            
        except Exception as e:
            logger.error(f"Orchestration failed: {e}", exc_info=True)
            raise HTTPException(
                status_code=500,
                detail=f"Orchestration failed: {str(e)}"
            )
    
    @router.post("/discover", response_model=DiscoveryResponse)
    async def discover_engines(
        self,
        request: DiscoveryRequest,
        api_key_valid: bool = Depends(verify_api_key)
    ) -> JSONResponse:
        """
        Discover Intelligence Amplification Engines (IAEs) for a task.
        """
        try:
            result = await self.discovery.discover_engines_for_task(
                task_type=request.task_type,
                domain_context=request.domain_context,
                complexity_requirements=request.complexity_requirements
            )
            
            return JSONResponse(content=result)
            
        except Exception as e:
            logger.error(f"Engine discovery failed: {e}", exc_info=True)
            raise HTTPException(
                status_code=500,
                detail=f"Engine discovery failed: {str(e)}"
            )
    
    @router.get("/health", response_model=HealthResponse)
    async def health_check(self) -> JSONResponse:
        """
        Check the health status of the MAESTRO server.
        """
        try:
            # Check core components
            discovery_ok = await self.discovery.registry.discover_engines(force=True)
            
            return JSONResponse(content={
                "status": "healthy",
                "timestamp": datetime.utcnow().isoformat(),
                "version": "1.0.0",
                "components": {
                    "discovery": "ok" if discovery_ok else "error",
                    "orchestrator": "ok",
                    "config": "ok"
                }
            })
            
        except Exception as e:
            logger.error(f"Health check failed: {e}", exc_info=True)
            return JSONResponse(
                content={
                    "status": "unhealthy",
                    "timestamp": datetime.utcnow().isoformat(),
                    "error": str(e)
                },
                status_code=500
            )
    
    @router.get("/")
    async def root(self) -> JSONResponse:
        """
        Root endpoint returning API information.
        """
        return JSONResponse(content={
            "name": "MAESTRO MCP Server",
            "version": "1.0.0",
            "description": "Backend-only, headless Intelligence Amplification server",
            "documentation": "/docs",
            "health": "/health"
        }) 