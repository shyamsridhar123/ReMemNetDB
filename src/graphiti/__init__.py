"""
Graphiti E-commerce Agent Memory Platform

A temporal knowledge graph system for e-commerce agent memory with semantic search capabilities.
"""

__version__ = "0.1.0"
__author__ = "shyamsridhar123"

# Import main components
from .ui.enhanced_gradio_app import EnhancedGraphitiUI as GraphitiUI

__all__ = [
    "GraphitiUI",
]
