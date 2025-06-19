"""
Comprehensive logging setup for Graphiti E-commerce Agent Memory Platform
"""

import logging
import sys
import os
from datetime import datetime
from pathlib import Path

def setup_logging(log_level=logging.INFO, log_to_file=True, log_dir="logs"):
    """
    Setup comprehensive logging for the application
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_to_file: Whether to log to file in addition to console
        log_dir: Directory to store log files
    """
    
    # Create logs directory if it doesn't exist
    if log_to_file:
        Path(log_dir).mkdir(exist_ok=True)
    
    # Create formatters
    console_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )
    
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(funcName)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Setup root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    
    # Clear any existing handlers
    root_logger.handlers.clear()
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)
    
    # File handler
    if log_to_file:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(log_dir, f"graphiti_{timestamp}.log")
        
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(log_level)
        file_handler.setFormatter(file_formatter)
        root_logger.addHandler(file_handler)
        
        # Also create a detailed debug log file
        debug_log_file = os.path.join(log_dir, f"graphiti_debug_{timestamp}.log")
        debug_handler = logging.FileHandler(debug_log_file, encoding='utf-8')
        debug_handler.setLevel(logging.DEBUG)
        debug_handler.setFormatter(file_formatter)
        root_logger.addHandler(debug_handler)
        
        print(f"📝 Logging to files: {log_file} and {debug_log_file}")
    
    # Setup specific loggers for different components
    setup_component_loggers()
    
    return root_logger

def setup_component_loggers():
    """Setup specific loggers for different components"""
    
    # Entity Extraction Logger
    extraction_logger = logging.getLogger('graphiti.extraction')
    extraction_logger.setLevel(logging.DEBUG)
    
    # Search Logger
    search_logger = logging.getLogger('graphiti.search')
    search_logger.setLevel(logging.DEBUG)
    
    # Database Logger
    db_logger = logging.getLogger('graphiti.database')
    db_logger.setLevel(logging.DEBUG)
    
    # Embedding Logger
    embedding_logger = logging.getLogger('graphiti.embedding')
    embedding_logger.setLevel(logging.DEBUG)
    
    # HTTP Logger (for API calls)
    http_logger = logging.getLogger('httpx')
    http_logger.setLevel(logging.WARNING)  # Only log warnings and errors for HTTP
    
    # OpenAI Logger
    openai_logger = logging.getLogger('openai')
    openai_logger.setLevel(logging.WARNING)

def log_function_entry(func_name, **kwargs):
    """Log function entry with parameters"""
    logger = logging.getLogger('graphiti.function_trace')
    args_str = ', '.join([f"{k}={v}" for k, v in kwargs.items()])
    logger.debug(f"ENTER {func_name}({args_str})")

def log_function_exit(func_name, result=None, duration=None):
    """Log function exit with result and duration"""
    logger = logging.getLogger('graphiti.function_trace')
    result_str = f"result={result}" if result is not None else "result=None"
    duration_str = f"duration={duration:.3f}s" if duration is not None else ""
    logger.debug(f"EXIT {func_name} {result_str} {duration_str}")

def log_embedding_details(entity_type, entity_id, embedding_type, embedding_length, similarity_score=None):
    """Log detailed embedding information"""
    logger = logging.getLogger('graphiti.embedding')
    msg = f"Embedding - Type: {entity_type}, ID: {entity_id}, EmbeddingType: {embedding_type}, Length: {embedding_length}"
    if similarity_score is not None:
        msg += f", Similarity: {similarity_score:.4f}"
    logger.debug(msg)

def log_search_details(query, search_type, num_results, threshold=None):
    """Log search operation details"""
    logger = logging.getLogger('graphiti.search')
    msg = f"Search - Query: '{query}', Type: {search_type}, Results: {num_results}"
    if threshold is not None:
        msg += f", Threshold: {threshold}"
    logger.info(msg)

def log_entity_extraction(event_type, num_entities, entities):
    """Log entity extraction details"""
    logger = logging.getLogger('graphiti.extraction')
    logger.info(f"Extracted {num_entities} entities from {event_type} event")
    for i, entity in enumerate(entities):
        logger.debug(f"  Entity {i+1}: {entity.type} - {entity.identifier}")

def log_hybrid_search_result(query, results, keyword_count, semantic_count):
    """Log hybrid search results"""
    logger = logging.getLogger('graphiti.search')
    logger.info(f"Hybrid Search Results for '{query}': {len(results)} total, {keyword_count} keyword, {semantic_count} semantic")
    for i, result in enumerate(results[:5]):  # Log top 5 results
        logger.debug(f"  Result {i+1}: Score={result.get('total_score', 0):.4f}, Type={result.get('source', 'unknown')}")

if __name__ == "__main__":
    # Test logging setup
    setup_logging(log_level=logging.DEBUG)
    
    logger = logging.getLogger('graphiti.test')
    logger.info("Testing logging setup")
    logger.debug("This is a debug message")
    logger.warning("This is a warning message")
    logger.error("This is an error message")
    
    # Test component-specific logging
    log_function_entry("test_function", param1="value1", param2=123)
    log_embedding_details("product", "laptop_001", "rich", 1536, 0.8345)
    log_search_details("laptop computer", "hybrid", 5, 0.3)
    log_function_exit("test_function", result="success", duration=1.234)
    
    print("✅ Logging setup test completed!")
