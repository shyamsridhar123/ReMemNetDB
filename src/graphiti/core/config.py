"""
core/config.py - Part of Graphiti E-commerce Agent Memory Platform
Configuration management for the Graphiti system.
"""

from pydantic_settings import BaseSettings
from pydantic import Field, validator
from typing import Optional, List
import os
from pathlib import Path

class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # ====================================================================
    # Database Configuration
    # ====================================================================
    database_url: str = Field(
        default="postgresql://postgres:postgres@localhost:54698/graphdb",
        description="Primary database connection URL"
    )
    local_database_url: Optional[str] = Field(
        default=None,
        description="Local development database URL"
    )
    db_host: str = Field(default="localhost", description="Database host")
    db_port: int = Field(default=5432, description="Database port")
    db_name: str = Field(default="graphdb", description="Database name")
    db_user: str = Field(default="postgres", description="Database user")
    db_password: str = Field(default="postgres", description="Database password")
    db_ssl_mode: str = Field(default="prefer", description="SSL mode")
    
    # Local database settings
    local_db_host: str = Field(default="localhost", description="Local database host")
    local_db_port: int = Field(default=5432, description="Local database port")
    local_db_name: str = Field(default="graphdb", description="Local database name")
    local_db_user: str = Field(default="postgres", description="Local database user")
    local_db_password: str = Field(default="postgres", description="Local database password")
    
    # Connection pool settings
    db_pool_size: int = Field(default=10, description="Database pool size")
    db_max_overflow: int = Field(default=20, description="Max pool overflow")
    db_pool_timeout: int = Field(default=30, description="Pool timeout")
    
    # ====================================================================
    # LLM Configuration
    # ====================================================================
    # OpenAI settings
    openai_api_key: Optional[str] = Field(default=None, description="OpenAI API key")
    openai_model: str = Field(default="gpt-4o-mini", description="OpenAI model")
    openai_embedding_model: str = Field(default="text-embedding-3-small", description="Embedding model")
    openai_max_tokens: int = Field(default=4000, description="Max tokens")
    openai_temperature: float = Field(default=0.1, description="Temperature")
    
    # Azure OpenAI settings
    azure_openai_endpoint: Optional[str] = Field(default=None, description="Azure OpenAI endpoint")
    azure_openai_api_key: Optional[str] = Field(default=None, description="Azure OpenAI API key")
    azure_openai_api_version: str = Field(default="2024-02-01", description="API version")
    azure_openai_gpt4_deployment_name: str = Field(default="gpt-4.1", description="GPT-4 deployment")
    azure_openai_gpt4_model_name: str = Field(default="gpt-4.1", description="GPT-4 model name")
    azure_openai_embedding_deployment_name: str = Field(default="text-embedding-3-small", description="Embedding deployment")
    azure_openai_embedding_model_name: str = Field(default="text-embedding-3-small", description="Embedding model name")
    azure_openai_embedding_dimensions: int = Field(default=1536, description="Embedding dimensions")
    
    # ====================================================================
    # Application Configuration
    # ====================================================================
    environment: str = Field(default="development", description="Environment")
    debug: bool = Field(default=True, description="Debug mode")
    log_level: str = Field(default="INFO", description="Log level")
    
    # Performance settings
    max_query_time: float = Field(default=2.0, description="Max query time")
    enable_query_caching: bool = Field(default=True, description="Enable caching")
    cache_ttl_seconds: int = Field(default=300, description="Cache TTL")
    
    # ====================================================================
    # Gradio UI Configuration
    # ====================================================================
    gradio_share: bool = Field(default=False, description="Gradio share")
    gradio_port: int = Field(default=7860, description="Gradio port")
    gradio_server_name: str = Field(default="127.0.0.1", description="Gradio server")
    gradio_show_tips: bool = Field(default=False, description="Show tips")
    gradio_show_error_msg: bool = Field(default=True, description="Show errors")
    
    # UI features
    enable_graph_visualization: bool = Field(default=True, description="Graph viz")
    enable_real_time_updates: bool = Field(default=True, description="Real-time updates")
    enable_scenario_demos: bool = Field(default=True, description="Scenario demos")
    
    # ====================================================================
    # FastAPI Configuration
    # ====================================================================
    api_host: str = Field(default="0.0.0.0", description="API host")
    api_port: int = Field(default=8000, description="API port")
    api_reload: bool = Field(default=True, description="API reload")
    
    # CORS settings
    cors_origins: str = Field(default='["http://localhost:3000", "http://localhost:7860", "http://127.0.0.1:7860"]', description="CORS origins")
    
    # Security
    secret_key: str = Field(default="dev-secret-key", description="Secret key")
    algorithm: str = Field(default="HS256", description="JWT algorithm")
    access_token_expire_minutes: int = Field(default=30, description="Token expiry")
    
    # ====================================================================
    # Data Generation Configuration
    # ====================================================================
    generate_sample_data: bool = Field(default=True, description="Generate sample data")
    sample_customers: int = Field(default=50, description="Sample customers")
    sample_products: int = Field(default=100, description="Sample products")
    sample_events_per_day: int = Field(default=200, description="Events per day")
    simulation_start_date: str = Field(default="2024-01-01", description="Sim start")
    simulation_end_date: str = Field(default="2024-12-31", description="Sim end")
    
    # ====================================================================
    # Graph Processing Configuration
    # ====================================================================
    age_graph_name: str = Field(default="graphiti_graph", description="AGE graph name")
    pgvector_extension_enabled: bool = Field(default=True, description="pgvector enabled")
    entity_extraction_batch_size: int = Field(default=10, description="Batch size")
    relationship_extraction_enabled: bool = Field(default=True, description="Relationship extraction")
    contradiction_detection_enabled: bool = Field(default=True, description="Contradiction detection")
    
    # ====================================================================
    # Feature Flags
    # ====================================================================
    feature_temporal_queries: bool = Field(default=True, description="Temporal queries")
    feature_hybrid_search: bool = Field(default=True, description="Hybrid search")
    feature_graph_visualization: bool = Field(default=True, description="Graph visualization")
    feature_real_time_events: bool = Field(default=True, description="Real-time events")
    feature_scenario_testing: bool = Field(default=True, description="Scenario testing")
    feature_export_data: bool = Field(default=True, description="Export data")
    feature_user_authentication: bool = Field(default=False, description="User auth")
    
    # ====================================================================
    # Development Settings
    # ====================================================================    enable_auto_reload: bool = Field(default=True, description="Auto reload")
    enable_debug_toolbar: bool = Field(default=True, description="Debug toolbar")
    show_sql_queries: bool = Field(default=True, description="Show SQL")
    mock_llm_responses: bool = Field(default=False, description="Mock LLM")
    
    @property
    def is_development(self) -> bool:
        """Check if running in development mode."""
        return self.environment.lower() in ['dev', 'development', 'local']
    
    @property
    def is_production(self) -> bool:
        """Check if running in production mode."""
        return self.environment.lower() in ['prod', 'production']
    
    @property
    def effective_llm_api_key(self) -> Optional[str]:
        """Get the effective LLM API key (Azure OpenAI or OpenAI)."""
        return self.azure_openai_api_key or self.openai_api_key
    
    @property
    def use_azure_openai(self) -> bool:
        """Check if using Azure OpenAI."""
        return bool(self.azure_openai_endpoint and self.azure_openai_api_key)

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
        extra = "ignore"  # Ignore extra fields from .env file

_settings = None

def get_settings() -> Settings:
    """Get application settings (singleton)."""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings
