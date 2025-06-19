"""
core/database.py - Part of Graphiti E-commerce Agent Memory Platform
"""

from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import StaticPool
from contextlib import contextmanager
import os
from typing import Generator

from .config import get_settings
from .models import Base

class DatabaseManager:
    """Database connection and session management."""
    
    def __init__(self):
        self.settings = get_settings()
        self.engine = create_engine(
            self.settings.database_url,
            poolclass=StaticPool,
            echo=self.settings.debug
        )
        self.SessionLocal = sessionmaker(
            autocommit=False, 
            autoflush=False, 
            bind=self.engine
        )
        
    def create_tables(self):
        """Create all database tables."""
        Base.metadata.create_all(bind=self.engine)
        
    @contextmanager
    def get_session(self) -> Generator[Session, None, None]:
        """Get database session with context management."""
        session = self.SessionLocal()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

# Global database manager instance
db_manager = DatabaseManager()

def get_database_session():
    """Get database session for backwards compatibility."""
    return db_manager.get_session()

def test_connection():
    """Test database connection."""
    try:
        with db_manager.get_session() as session:
            result = session.execute(text("SELECT 1"))
            return True
    except Exception as e:
        print(f"Database connection failed: {e}")
        return False
