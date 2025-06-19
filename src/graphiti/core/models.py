"""
core/models.py - Part of Graphiti E-commerce Agent Memory Platform
"""

import uuid
from datetime import datetime
from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field
from sqlalchemy import Column, String, DateTime, Boolean, ForeignKey, Index, func, Text
from sqlalchemy.dialects.postgresql import UUID as PG_UUID, JSONB
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
# Note: Vector type will be defined as Text for now, can be updated later
# from pgvector.sqlalchemy import Vector

Base = declarative_base()


class TemporalNode(Base):
    """
    Temporal node in the knowledge graph with bitemporal support.
    Tracks both valid-time (when the fact was true in reality) and 
    transaction-time (when the fact was recorded in the database).
    """
    __tablename__ = "nodes"
    __table_args__ = (
        Index('idx_nodes_type', 'type'),
        Index('idx_nodes_valid_from', 'valid_from'),
        Index('idx_nodes_valid_to', 'valid_to'),
        Index('idx_nodes_recorded_at', 'recorded_at'),
        Index('idx_nodes_embedding', 'embedding', postgresql_using='ivfflat',
              postgresql_with={'lists': 100}, postgresql_ops={'embedding': 'vector_l2_ops'}),
        {'schema': 'temporal_graph'},
    )
    
    id = Column(PG_UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    type = Column(String(50), nullable=False)
    properties = Column(JSONB)
    valid_from = Column(DateTime(timezone=True), nullable=False)
    valid_to = Column(DateTime(timezone=True))
    recorded_at = Column(DateTime(timezone=True), server_default=func.now())
    # Vector embedding - using Text for now, will be converted to proper VECTOR type in database
    embedding = Column(Text)  # Will store vector as text representation
    
    # Relationships
    outgoing_edges = relationship("TemporalEdge", foreign_keys="TemporalEdge.source_node_id", back_populates="source_node")
    incoming_edges = relationship("TemporalEdge", foreign_keys="TemporalEdge.target_node_id", back_populates="target_node")
    def __repr__(self):
        return f"<TemporalNode(id={self.id}, type='{self.type}', valid_from='{self.valid_from}')>"
    
    def to_dict(self):
        """Convert node to dictionary for safe serialization."""
        return {
            'id': str(self.id),
            'type': self.type,
            'properties': self.properties,
            'valid_from': self.valid_from.isoformat() if self.valid_from else None,
            'valid_to': self.valid_to.isoformat() if self.valid_to else None,
            'recorded_at': self.recorded_at.isoformat() if self.recorded_at else None
        }
    
    @property
    def is_currently_valid(self) -> bool:
        """Check if this node version is currently valid."""
        now = datetime.utcnow()
        return self.valid_from <= now and (self.valid_to is None or self.valid_to > now)


class TemporalEdge(Base):    
    """
    Temporal edge representing relationships between nodes with bitemporal support.
    """
    __tablename__ = "edges"
    __table_args__ = (
        Index('idx_edges_relationship_type', 'relationship_type'),
        Index('idx_edges_source_target', 'source_node_id', 'target_node_id'),
        Index('idx_edges_valid_from', 'valid_from'),
        Index('idx_edges_valid_to', 'valid_to'),
        Index('idx_edges_recorded_at', 'recorded_at'),
        {'schema': 'temporal_graph'},
    )
    
    id = Column(PG_UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    source_node_id = Column(PG_UUID(as_uuid=True), ForeignKey('temporal_graph.nodes.id'), nullable=False)
    target_node_id = Column(PG_UUID(as_uuid=True), ForeignKey('temporal_graph.nodes.id'), nullable=False)
    relationship_type = Column(String(50), nullable=False)
    properties = Column(JSONB)
    valid_from = Column(DateTime(timezone=True), nullable=False)
    valid_to = Column(DateTime(timezone=True))
    recorded_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Relationships
    source_node = relationship("TemporalNode", foreign_keys=[source_node_id], back_populates="outgoing_edges")
    target_node = relationship("TemporalNode", foreign_keys=[target_node_id], back_populates="incoming_edges")
    
    def __repr__(self):
        return f"<TemporalEdge(id={self.id}, type='{self.relationship_type}', {self.source_node_id}->{self.target_node_id})>"
    
    @property
    def is_currently_valid(self) -> bool:
        """Check if this edge version is currently valid."""
        now = datetime.utcnow()
        return self.valid_from <= now and (self.valid_to is None or self.valid_to > now)
    
    def to_dict(self) -> dict:
        """Convert to dictionary without triggering lazy loading."""
        return {
            'id': str(self.id),
            'source_node_id': str(self.source_node_id),
            'target_node_id': str(self.target_node_id),
            'relationship_type': self.relationship_type,
            'properties': self.properties,
            'valid_from': self.valid_from,
            'valid_to': self.valid_to,
            'recorded_at': self.recorded_at
        }


class Event(Base):    
    """
    Raw events that get processed into the temporal knowledge graph.
    """
    __tablename__ = "events"
    __table_args__ = (
        Index('idx_events_event_type', 'event_type'),
        Index('idx_events_timestamp', 'timestamp'),
        Index('idx_events_processed', 'processed'),
        {'schema': 'temporal_graph'},
    )
    
    id = Column(PG_UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    event_type = Column(String(50), nullable=False)
    event_data = Column(JSONB, nullable=False)
    timestamp = Column(DateTime(timezone=True), nullable=False)
    processed = Column(Boolean, default=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    def __repr__(self):
        return f"<Event(id={self.id}, type='{self.event_type}', timestamp='{self.timestamp}')>"


# Pydantic models for API and data validation
class NodeSchema(BaseModel):
    id: Optional[str] = None
    type: str
    properties: Dict[str, Any] = {}
    valid_from: datetime
    valid_to: Optional[datetime] = None
    embedding: Optional[List[float]] = None
    
    class Config:
        from_attributes = True


class EdgeSchema(BaseModel):
    id: Optional[str] = None
    source_node_id: str
    target_node_id: str
    relationship_type: str
    properties: Dict[str, Any] = {}
    valid_from: datetime
    valid_to: Optional[datetime] = None
    
    class Config:
        from_attributes = True


class EventSchema(BaseModel):
    id: Optional[str] = None
    event_type: str
    event_data: Dict[str, Any]
    timestamp: datetime
    processed: bool = False
    
    class Config:
        from_attributes = True


# E-commerce specific models
class Customer(BaseModel):
    """E-commerce customer model."""
    id: str
    name: str
    email: str
    demographics: Dict[str, Any] = {}
    preferences: List[str] = []
    behavior_pattern: Optional[str] = None


class Product(BaseModel):
    """E-commerce product model."""
    id: str
    name: str
    category: str
    price: float
    attributes: Dict[str, Any] = {}
    reviews: List[Dict[str, Any]] = []


class Order(BaseModel):
    """E-commerce order model."""
    id: str
    customer_id: str
    products: List[Dict[str, Any]]
    timestamp: datetime
    status: str
    total: float
