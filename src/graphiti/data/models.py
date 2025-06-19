"""
data/models.py - Part of Graphiti E-commerce Agent Memory Platform
"""

from datetime import datetime
from typing import Optional, Dict, Any, List
from uuid import UUID
from pydantic import BaseModel, Field
from sqlalchemy import Column, String, DateTime, Boolean, JSON, ForeignKey
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship

Base = declarative_base()

class Node(Base):
    """Temporal graph node model."""
    __tablename__ = "nodes"
    
    id = Column(PG_UUID(as_uuid=True), primary_key=True)
    type = Column(String(50), nullable=False)
    properties = Column(JSON)
    valid_from = Column(DateTime(timezone=True), nullable=False)
    valid_to = Column(DateTime(timezone=True))
    recorded_at = Column(DateTime(timezone=True), default=datetime.utcnow)
    
class Edge(Base):
    """Temporal graph edge model."""
    __tablename__ = "edges"
    
    id = Column(PG_UUID(as_uuid=True), primary_key=True)
    source_node_id = Column(PG_UUID(as_uuid=True), ForeignKey("nodes.id"))
    target_node_id = Column(PG_UUID(as_uuid=True), ForeignKey("nodes.id"))
    relationship_type = Column(String(50), nullable=False)
    properties = Column(JSON)
    valid_from = Column(DateTime(timezone=True), nullable=False)
    valid_to = Column(DateTime(timezone=True))
    recorded_at = Column(DateTime(timezone=True), default=datetime.utcnow)

class Event(Base):
    """Event model for temporal tracking."""
    __tablename__ = "events"
    
    id = Column(PG_UUID(as_uuid=True), primary_key=True)
    event_type = Column(String(50), nullable=False)
    event_data = Column(JSON, nullable=False)
    timestamp = Column(DateTime(timezone=True), nullable=False)
    processed = Column(Boolean, default=False)
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)

# Pydantic models for API
class NodeSchema(BaseModel):
    id: Optional[UUID] = None
    type: str
    properties: Dict[str, Any] = {}
    valid_from: datetime
    valid_to: Optional[datetime] = None
    
    class Config:
        from_attributes = True

class EdgeSchema(BaseModel):
    id: Optional[UUID] = None
    source_node_id: UUID
    target_node_id: UUID
    relationship_type: str
    properties: Dict[str, Any] = {}
    valid_from: datetime
    valid_to: Optional[datetime] = None
    
    class Config:
        from_attributes = True

class EventSchema(BaseModel):
    id: Optional[UUID] = None
    event_type: str
    event_data: Dict[str, Any]
    timestamp: datetime
    processed: bool = False
    
    class Config:
        from_attributes = True
