"""
MAESTRO Configuration Module

Provides centralized configuration management for the MAESTRO MCP server.
All configuration is done through environment variables to maintain a headless,
IDE-agnostic design.
"""

import os
from typing import Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum
import logging

# Configure logging
logging.basicConfig(
    level=os.getenv("MAESTRO_LOG_LEVEL", "INFO"),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class LogLevel(str, Enum):
    """Valid logging levels."""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"

class EngineMode(str, Enum):
    """Valid engine operation modes."""
    DEVELOPMENT = "development"
    PRODUCTION = "production"
    TESTING = "testing"

@dataclass
class ServerConfig:
    """Server configuration settings."""
    host: str
    port: int
    workers: int
    timeout: int
    cors_origins: list[str]
    
@dataclass
class SecurityConfig:
    """Security configuration settings."""
    api_key_required: bool
    allowed_origins: list[str]
    rate_limit_enabled: bool
    rate_limit_requests: int
    rate_limit_window: int  # seconds
    
@dataclass
class EngineConfig:
    """Engine configuration settings."""
    mode: EngineMode
    max_concurrent_tasks: int
    task_timeout: int
    memory_limit: int  # MB
    enable_gpu: bool
    
@dataclass
class LoggingConfig:
    """Logging configuration settings."""
    level: LogLevel
    file_enabled: bool
    file_path: Optional[str]
    rotation_size: int  # MB
    retention_days: int

@dataclass
class MAESTROConfig:
    """Main MAESTRO configuration."""
    server: ServerConfig
    security: SecurityConfig
    engine: EngineConfig
    logging: LoggingConfig
    
    @classmethod
    def from_env(cls) -> 'MAESTROConfig':
        """
        Create configuration from environment variables.
        Uses sensible defaults if variables are not set.
        """
        # Server config
        server = ServerConfig(
            host=os.getenv("MAESTRO_HOST", "0.0.0.0"),
            port=int(os.getenv("MAESTRO_PORT", "8000")),
            workers=int(os.getenv("MAESTRO_WORKERS", "4")),
            timeout=int(os.getenv("MAESTRO_TIMEOUT", "30")),
            cors_origins=os.getenv("MAESTRO_CORS_ORIGINS", "*").split(",")
        )
        
        # Security config
        security = SecurityConfig(
            api_key_required=os.getenv("MAESTRO_API_KEY_REQUIRED", "false").lower() == "true",
            allowed_origins=os.getenv("MAESTRO_ALLOWED_ORIGINS", "*").split(","),
            rate_limit_enabled=os.getenv("MAESTRO_RATE_LIMIT_ENABLED", "true").lower() == "true",
            rate_limit_requests=int(os.getenv("MAESTRO_RATE_LIMIT_REQUESTS", "100")),
            rate_limit_window=int(os.getenv("MAESTRO_RATE_LIMIT_WINDOW", "60"))
        )
        
        # Engine config
        engine = EngineConfig(
            mode=EngineMode(os.getenv("MAESTRO_MODE", "production").lower()),
            max_concurrent_tasks=int(os.getenv("MAESTRO_MAX_CONCURRENT_TASKS", "10")),
            task_timeout=int(os.getenv("MAESTRO_TASK_TIMEOUT", "300")),
            memory_limit=int(os.getenv("MAESTRO_MEMORY_LIMIT", "1024")),
            enable_gpu=os.getenv("MAESTRO_ENABLE_GPU", "false").lower() == "true"
        )
        
        # Logging config
        logging = LoggingConfig(
            level=LogLevel(os.getenv("MAESTRO_LOG_LEVEL", "INFO").upper()),
            file_enabled=os.getenv("MAESTRO_LOG_FILE_ENABLED", "false").lower() == "true",
            file_path=os.getenv("MAESTRO_LOG_FILE_PATH", None),
            rotation_size=int(os.getenv("MAESTRO_LOG_ROTATION_SIZE", "100")),
            retention_days=int(os.getenv("MAESTRO_LOG_RETENTION_DAYS", "30"))
        )
        
        return cls(
            server=server,
            security=security,
            engine=engine,
            logging=logging
        )
    
    def validate(self) -> None:
        """Validate the configuration."""
        # Server validation
        if self.server.port < 1 or self.server.port > 65535:
            raise ValueError("Invalid port number")
        if self.server.workers < 1:
            raise ValueError("Workers must be >= 1")
            
        # Security validation
        if self.security.rate_limit_requests < 1:
            raise ValueError("Rate limit requests must be >= 1")
        if self.security.rate_limit_window < 1:
            raise ValueError("Rate limit window must be >= 1")
            
        # Engine validation
        if self.engine.max_concurrent_tasks < 1:
            raise ValueError("Max concurrent tasks must be >= 1")
        if self.engine.task_timeout < 1:
            raise ValueError("Task timeout must be >= 1")
        if self.engine.memory_limit < 128:
            raise ValueError("Memory limit must be >= 128 MB")
            
        # Logging validation
        if self.logging.file_enabled and not self.logging.file_path:
            raise ValueError("Log file path required when file logging is enabled")
        if self.logging.rotation_size < 1:
            raise ValueError("Log rotation size must be >= 1 MB")
        if self.logging.retention_days < 1:
            raise ValueError("Log retention days must be >= 1")
            
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary format."""
        return {
            "server": {
                "host": self.server.host,
                "port": self.server.port,
                "workers": self.server.workers,
                "timeout": self.server.timeout,
                "cors_origins": self.server.cors_origins
            },
            "security": {
                "api_key_required": self.security.api_key_required,
                "allowed_origins": self.security.allowed_origins,
                "rate_limit_enabled": self.security.rate_limit_enabled,
                "rate_limit_requests": self.security.rate_limit_requests,
                "rate_limit_window": self.security.rate_limit_window
            },
            "engine": {
                "mode": self.engine.mode.value,
                "max_concurrent_tasks": self.engine.max_concurrent_tasks,
                "task_timeout": self.engine.task_timeout,
                "memory_limit": self.engine.memory_limit,
                "enable_gpu": self.engine.enable_gpu
            },
            "logging": {
                "level": self.logging.level.value,
                "file_enabled": self.logging.file_enabled,
                "file_path": self.logging.file_path,
                "rotation_size": self.logging.rotation_size,
                "retention_days": self.logging.retention_days
            }
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MAESTROConfig':
        """Create configuration from a dictionary."""
        server_data = data.get("server", {})
        # Handle cors_origins as either string or list
        cors_origins = server_data.get("cors_origins", "*")
        if isinstance(cors_origins, str):
            cors_origins = cors_origins.split(",") if cors_origins != "*" else ["*"]
        
        server = ServerConfig(
            host=server_data.get("host", "0.0.0.0"),
            port=int(server_data.get("port", 8000)),
            workers=int(server_data.get("workers", 4)),
            timeout=int(server_data.get("timeout", 30)),
            cors_origins=cors_origins,
        )

        security_data = data.get("security", {})
        # Handle allowed_origins as either string or list
        allowed_origins = security_data.get("allowed_origins", "*")
        if isinstance(allowed_origins, str):
            allowed_origins = allowed_origins.split(",") if allowed_origins != "*" else ["*"]
            
        security = SecurityConfig(
            api_key_required=str(security_data.get("api_key_required", "false")).lower() == "true",
            allowed_origins=allowed_origins,
            rate_limit_enabled=str(security_data.get("rate_limit_enabled", "true")).lower() == "true",
            rate_limit_requests=int(security_data.get("rate_limit_requests", 100)),
            rate_limit_window=int(security_data.get("rate_limit_window", 60)),
        )

        engine_data = data.get("engine", {})
        engine = EngineConfig(
            mode=EngineMode(str(engine_data.get("mode", "production")).lower()),
            max_concurrent_tasks=int(engine_data.get("max_concurrent_tasks", 10)),
            task_timeout=int(engine_data.get("task_timeout", 300)),
            memory_limit=int(engine_data.get("memory_limit", 1024)),
            enable_gpu=str(engine_data.get("enable_gpu", "false")).lower() == "true",
        )

        logging_data = data.get("logging", {})
        logging = LoggingConfig(
            level=LogLevel(str(logging_data.get("level", "INFO")).upper()),
            file_enabled=str(logging_data.get("file_enabled", "false")).lower() == "true",
            file_path=logging_data.get("file_path"),
            rotation_size=int(logging_data.get("rotation_size", 100)),
            retention_days=int(logging_data.get("retention_days", 30)),
        )

        return cls(
            server=server,
            security=security,
            engine=engine,
            logging=logging
        ) 