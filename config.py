import os  
from typing import Optional  
from pydantic import BaseSettings, Field  
  
class Settings(BaseSettings):  
    """Application settings"""  
      
    # API Keys  
    google_api_key: Optional[str] = Field(None, env="GOOGLE_API_KEY")  
      
    # Application settings  
    max_file_size_mb: int = Field(100, env="MAX_FILE_SIZE_MB")  
    max_memory_interactions: int = Field(100, env="MAX_MEMORY_INTERACTIONS")  
      
    # Analysis settings  
    default_correlation_threshold: float = Field(0.5, env="DEFAULT_CORRELATION_THRESHOLD")  
    max_clusters: int = Field(10, env="MAX_CLUSTERS")  
    outlier_contamination: float = Field(0.1, env="OUTLIER_CONTAMINATION")  
      
    # Streamlit settings  
    page_title: str = Field("Agente EDA Avançado", env="PAGE_TITLE")  
    page_icon: str = Field("🤖", env="PAGE_ICON")  
      
    class Config:  
        env_file = ".env"  
        case_sensitive = False  
  
settings = Settings()