# Implementation Plan: Graphiti E-commerce Agent Memory Platform

## Overview
This document outlines the implementation steps for the Graphiti E-commerce Agent Memory Platform with Gradio UI, as specified in the PRD. The project uses modern Python tooling with `uv` for fast dependency management and project setup.

**Key Technology Choices:**
- **uv**: Ultra-fast Python package manager for dependency resolution and virtual environment management
- **PostgreSQL with MCP**: Local development with PostgreSQL + MCP, production deployment on Azure PostgreSQL + MCP
- **Apache AGE**: Graph database extension for PostgreSQL
- **pgvector**: Vector similarity search extension for PostgreSQL
- **Gradio**: Web UI framework for interactive demos and testing

---

## Step 1: Database Infrastructure Setup
**Technologies:** Local PostgreSQL (development), Azure PostgreSQL (production), Apache AGE, pgvector, MCP

**What to do:**
- Set up local PostgreSQL database instance in Docker
- Install and configure Apache AGE extension for graph capabilities
- Install and configure pgvector extension for semantic search
- Configure MCP server for database access
- Create database schemas for temporal knowledge graph
- Plan migration path to Azure PostgreSQL with same MCP interface

**Database Schema Design:**
```sql
-- Core temporal graph tables
CREATE TABLE nodes (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    type VARCHAR(50) NOT NULL,
    properties JSONB,
    valid_from TIMESTAMP WITH TIME ZONE NOT NULL,
    valid_to TIMESTAMP WITH TIME ZONE,
    recorded_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    embedding VECTOR(1536) -- For semantic search
);

CREATE TABLE edges (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    source_node_id UUID REFERENCES nodes(id),
    target_node_id UUID REFERENCES nodes(id),
    relationship_type VARCHAR(50) NOT NULL,
    properties JSONB,
    valid_from TIMESTAMP WITH TIME ZONE NOT NULL,
    valid_to TIMESTAMP WITH TIME ZONE,
    recorded_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE TABLE events (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    event_type VARCHAR(50) NOT NULL,
    event_data JSONB NOT NULL,
    timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
    processed BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
```

**Deliverables:**
- Local PostgreSQL instance with extensions and MCP integration
- Azure PostgreSQL configuration for production deployment  
- Database schema documentation
- MCP server configuration for consistent database access

## Step 2: Python Environment Setup
**Technologies:** Python 3.11+, uv, FastAPI, SQLAlchemy, Pydantic

**What to do:**
- Install uv for fast Python package and project management
- Initialize project with uv and create project structure
- Configure dependencies using pyproject.toml
- Set up development dependencies and scripts
- Create Docker configuration for development using uv

**Project Structure:**
```
graphiti-postgres/
├── src/
│   ├── graphiti/
│   │   ├── __init__.py
│   │   ├── core/
│   │   │   ├── __init__.py
│   │   │   ├── models.py
│   │   │   ├── database.py
│   │   │   └── config.py
│   │   ├── memory/
│   │   │   ├── __init__.py
│   │   │   ├── temporal_graph.py
│   │   │   ├── extraction.py
│   │   │   └── contradiction_handler.py
│   │   ├── search/
│   │   │   ├── __init__.py
│   │   │   ├── semantic.py
│   │   │   ├── keyword.py
│   │   │   └── graph_search.py
│   │   ├── data/
│   │   │   ├── __init__.py
│   │   │   ├── generator.py
│   │   │   └── models.py
│   │   └── ui/
│   │       ├── __init__.py
│   │       ├── gradio_app.py
│   │       └── components/
├── tests/
├── docker/
├── pyproject.toml          # uv project configuration
├── uv.lock                 # uv lock file (auto-generated)
├── README.md
├── PRD.md
└── IMPLEMENTATION_PLAN.md
```

**Deliverables:**
- Python project structure initialized with uv
- Docker development environment using uv
- Configuration management system
- pyproject.toml with all dependencies and project metadata

**uv Setup Instructions:**
```powershell
# Install uv (Windows)
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# Initialize the project
uv init graphiti-postgres --package
cd graphiti-postgres

# Add core dependencies
uv add fastapi sqlalchemy pydantic psycopg2-binary
uv add gradio plotly networkx pandas numpy
uv add openai tiktoken
uv add uvicorn gunicorn

# Add development dependencies
uv add --dev pytest pytest-asyncio pytest-cov
uv add --dev black isort flake8 mypy
uv add --dev pre-commit

# Add optional dependencies for different features
uv add --optional azure azure-storage-blob azure-keyvault-secrets
uv add --optional monitoring prometheus-client structlog

# Run the application
uv run src/main.py
```

**pyproject.toml Configuration:**
```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "graphiti-postgres"
dynamic = ["version"]
description = "E-commerce Agent Memory Platform with Temporal Knowledge Graph"
readme = "README.md"
license = {file = "LICENSE"}
requires-python = ">=3.11"
authors = [
    {name = "Your Name", email = "your.email@example.com"},
]
classifiers = [
    "Development Status :: 3 - Alpha",
    "Intended Audience :: Developers",
    "License :: OSI Approved :: MIT License",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.11",
    "Programming Language :: Python :: 3.12",
]

dependencies = [
    "fastapi>=0.104.0",
    "sqlalchemy>=2.0.0",
    "pydantic>=2.4.0",
    "psycopg2-binary>=2.9.0",
    "gradio>=4.0.0",
    "plotly>=5.17.0",
    "networkx>=3.2.0",
    "pandas>=2.1.0",
    "numpy>=1.25.0",
    "openai>=1.3.0",
    "tiktoken>=0.5.0",
    "uvicorn>=0.24.0",
    "python-multipart>=0.0.6",
    "python-dotenv>=1.0.0",
]

[project.optional-dependencies]
azure = [
    "azure-storage-blob>=12.19.0",
    "azure-keyvault-secrets>=4.7.0",
    "azure-identity>=1.15.0",
]
monitoring = [
    "prometheus-client>=0.19.0",
    "structlog>=23.2.0",
]
dev = [
    "pytest>=7.4.0",
    "pytest-asyncio>=0.21.0",
    "pytest-cov>=4.1.0",
    "black>=23.9.0",
    "isort>=5.12.0",
    "flake8>=6.1.0",
    "mypy>=1.6.0",
    "pre-commit>=3.5.0",
]

[project.scripts]
graphiti-server = "graphiti.ui.gradio_app:main"
graphiti-cli = "graphiti.cli:main"

[project.urls]
Homepage = "https://github.com/yourusername/graphiti-postgres"
Documentation = "https://graphiti-postgres.readthedocs.io"
Repository = "https://github.com/yourusername/graphiti-postgres"
Issues = "https://github.com/yourusername/graphiti-postgres/issues"

[tool.hatch.version]
path = "src/graphiti/__init__.py"

[tool.hatch.build.targets.sdist]
include = [
    "/src",
    "/tests",
    "/README.md",
    "/LICENSE",
]

[tool.hatch.build.targets.wheel]
packages = ["src/graphiti"]

[tool.black]
line-length = 88
target-version = ['py311']
include = '\.pyi?$'
exclude = '''
/(
    \.eggs
  | \.git
  | \.hg
  | \.mypy_cache
  | \.tox
  | \.venv
  | _build
  | buck-out
  | build
  | dist
)/
'''

[tool.isort]
profile = "black"
multi_line_output = 3
line_length = 88
known_first_party = ["graphiti"]

[tool.mypy]
python_version = "3.11"
warn_return_any = true
warn_unused_configs = true
disallow_untyped_defs = true
disallow_incomplete_defs = true
check_untyped_defs = true
disallow_untyped_decorators = true
no_implicit_optional = true
warn_redundant_casts = true
warn_unused_ignores = true
warn_unreachable = true
strict_equality = true

[[tool.mypy.overrides]]
module = [
    "gradio.*",
    "plotly.*",
    "networkx.*",
]
ignore_missing_imports = true

[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = ["test_*.py", "*_test.py"]
python_classes = ["Test*"]
python_functions = ["test_*"]
addopts = [
    "--cov=src/graphiti",
    "--cov-report=term-missing",
    "--cov-report=html",
    "--cov-fail-under=80",
]
```

## Step 3: Temporal Knowledge Graph Engine
**Core Features:** Bi-temporal data handling, entity extraction, relationship management

**What to do:**
- Implement temporal graph data models
- Create temporal query engine for valid-time and transaction-time queries
- Build entity extraction pipeline using LLMs
- Implement relationship detection and classification
- Create contradiction detection and resolution system

**Key Components:**

**TemporalGraph Class:**
```python
class TemporalGraph:
    def __init__(self, db_session):
        self.db = db_session
        
    def add_event(self, event: Event) -> None:
        """Process and store event in temporal graph"""
        
    def query_at_time(self, query: str, timestamp: datetime) -> List[Node]:
        """Query graph state at specific time"""
        
    def get_entity_history(self, entity_id: str) -> List[Tuple[Node, TimeRange]]:
        """Get complete history of entity changes"""
```

**EntityExtractor Class:**
```python
class EntityExtractor:
    def __init__(self, llm_client):
        self.llm = llm_client
        
    def extract_entities(self, event_data: dict) -> List[Entity]:
        """Extract entities from event data using LLM"""
        
    def extract_relationships(self, entities: List[Entity]) -> List[Relationship]:
        """Extract relationships between entities"""
```

**Deliverables:**
- Temporal graph engine
- Entity extraction pipeline
- Contradiction handling system
- Unit tests for core functionality

## Step 4: Memory Storage and Retrieval
**Core Features:** Event storage, temporal queries, consistency management

**What to do:**
- Implement event ingestion pipeline
- Create temporal query interface
- Build consistency checking mechanisms
- Implement graph traversal algorithms
- Create memory consolidation processes

**Key Components:**

**MemoryStore Class:**
```python
class MemoryStore:
    def __init__(self, temporal_graph: TemporalGraph):
        self.graph = temporal_graph
        
    def store_episode(self, episode: Episode) -> None:
        """Store complete episode in memory"""
        
    def retrieve_memories(self, query: MemoryQuery) -> List[Memory]:
        """Retrieve relevant memories based on query"""
        
    def get_temporal_sequence(self, entity_id: str, time_range: TimeRange) -> List[Event]:
        """Get chronological sequence of events for entity"""
```

**Deliverables:**
- Memory storage system
- Temporal query engine
- Graph traversal algorithms

## Step 5: Semantic Search with pgvector
**Core Features:** Embedding generation, vector similarity search

**What to do:**
- Implement embedding generation for entities and events
- Create semantic search interface using pgvector
- Build similarity scoring algorithms
- Optimize vector index performance
- Implement embedding caching

**SemanticSearch Class:**
```python
class SemanticSearch:
    def __init__(self, db_session, embedding_model):
        self.db = db_session
        self.embedder = embedding_model
        
    def generate_embeddings(self, text: str) -> List[float]:
        """Generate embeddings for text content"""
        
    def semantic_search(self, query: str, limit: int = 10) -> List[SearchResult]:
        """Perform semantic similarity search"""
        
    def hybrid_search(self, query: str, filters: dict) -> List[SearchResult]:
        """Combine semantic search with metadata filters"""
```

## Step 6: Keyword Search and Graph Traversal
**Core Features:** BM25 search, graph-based retrieval

**What to do:**
- Implement BM25 keyword search using PostgreSQL full-text search
- Create graph traversal algorithms for relationship-based queries
- Build hybrid search orchestrator
- Implement search result ranking and fusion

**HybridSearchEngine Class:**
```python
class HybridSearchEngine:
    def __init__(self, semantic_search, keyword_search, graph_search):
        self.semantic = semantic_search
        self.keyword = keyword_search
        self.graph = graph_search
        
    def search(self, query: SearchQuery) -> SearchResults:
        """Orchestrate hybrid search across all modalities"""
        
    def rank_results(self, results: List[SearchResult]) -> List[SearchResult]:
        """Rank and fuse results from different search types"""
```

**Deliverables:**
- Semantic search engine
- Keyword search implementation
- Graph traversal system
- Hybrid search orchestrator

## Step 7: E-commerce Data Models
**Core Features:** Realistic e-commerce entities and events

**What to do:**
- Design comprehensive e-commerce data models
- Create customer persona generation
- Implement product catalog generation
- Build realistic transaction patterns
- Create temporal event sequences

**Data Models:**
```python
@dataclass
class Customer:
    id: str
    name: str
    email: str
    demographics: Dict
    preferences: List[str]
    behavior_pattern: str

@dataclass
class Product:
    id: str
    name: str
    category: str
    price: float
    attributes: Dict
    reviews: List[Review]

@dataclass
class Order:
    id: str
    customer_id: str
    products: List[OrderItem]
    timestamp: datetime
    status: str
    total: float
```

## Step 8: Event Generation Engine
**Core Features:** Time-series event generation, realistic patterns

**What to do:**
- Implement event generation algorithms
- Create realistic temporal patterns (seasonal, weekly, daily)
- Build customer journey simulation
- Generate product lifecycle events
- Create support interaction patterns

**EventGenerator Class:**
```python
class EventGenerator:
    def __init__(self, config: GenerationConfig):
        self.config = config
        
    def generate_customer_journey(self, customer: Customer, duration: timedelta) -> List[Event]:
        """Generate realistic customer journey events"""
        
    def generate_product_lifecycle(self, product: Product, duration: timedelta) -> List[Event]:
        """Generate product-related events over time"""
        
    def generate_seasonal_patterns(self, base_events: List[Event]) -> List[Event]:
        """Add seasonal variations to event patterns"""
```

**Deliverables:**
- Comprehensive data models
- Event generation engine
- Realistic temporal patterns
- Sample datasets for testing

## Step 9: Core UI Components
**Core Features:** Interactive panels, real-time updates, visualization

**What to do:**
- Design responsive Gradio interface layout
- Implement episodic query panel
- Create hybrid search interface
- Build graph visualization components
- Implement real-time update panel

**UI Architecture:**
```python
class GraphitiUI:
    def __init__(self, memory_store, search_engine):
        self.memory = memory_store
        self.search = search_engine
        
    def create_episodic_query_tab(self) -> gr.Tab:
        """Create episodic memory query interface"""
        
    def create_hybrid_search_tab(self) -> gr.Tab:
        """Create hybrid search interface"""
        
    def create_visualization_tab(self) -> gr.Tab:
        """Create graph visualization interface"""
        
    def create_real_time_tab(self) -> gr.Tab:
        """Create real-time update interface"""
```

## Step 10: Advanced UI Features
**Core Features:** Graph visualization, agent memory demo, scenario testing

**What to do:**
- Implement interactive graph visualization using Plotly/NetworkX
- Create agent memory evolution demo
- Build scenario testing framework
- Implement export and sharing features
- Add performance monitoring dashboard

**Visualization Components:**
```python
class GraphVisualizer:
    def __init__(self, temporal_graph):
        self.graph = temporal_graph
        
    def visualize_subgraph(self, entity_id: str, time_range: TimeRange) -> plotly.Figure:
        """Create interactive subgraph visualization"""
        
    def visualize_temporal_evolution(self, entity_id: str) -> plotly.Figure:
        """Show how entity relationships change over time"""
        
    def create_community_visualization(self, community_id: str) -> plotly.Figure:
        """Visualize entity communities and their evolution"""
```

**Deliverables:**
- Complete Gradio web interface
- Interactive graph visualizations
- Agent memory demo
- Scenario testing framework

## Step 11: Scenario Implementation
**Core Features:** All 7 scenarios from PRD working end-to-end

**What to do:**
- Implement Scenario 1: Personalized Customer Journey Recall
- Implement Scenario 2: Real-Time Trend and Community Shifts
- Implement Scenario 3: Temporal Reasoning for Support
- Implement Scenario 4: Hybrid Search for Product Discovery
- Implement Scenario 5: Fact Contradiction and Update
- Implement Scenario 6: Agentic Memory Evolution
- Implement Scenario 7: Relationship Evolution

**Scenario Testing Framework:**
```python
class ScenarioTester:
    def __init__(self, graphiti_system):
        self.system = graphiti_system
        
    def run_scenario_1(self) -> ScenarioResult:
        """Execute personalized customer journey scenario"""
        
    def run_all_scenarios(self) -> List[ScenarioResult]:
        """Execute all PRD scenarios and validate results"""
        
    def validate_performance(self) -> PerformanceMetrics:
        """Validate system meets performance criteria"""
```

## Step 12: System Integration and Testing
**Core Features:** End-to-end testing, performance validation

**What to do:**
- Integrate all components into cohesive system
- Implement comprehensive test suite
- Performance testing and optimization
- User acceptance testing preparation
- Documentation and deployment guides

**Deliverables:**
- All 7 scenarios working in UI
- Comprehensive test suite
- Performance benchmarks
- Integration documentation

## Step 13: Documentation
**What to do:**
- Create comprehensive API documentation
- Write user guides and tutorials
- Document deployment procedures
- Create troubleshooting guides
- Record demo videos

## Step 14: Deployment Preparation
**What to do:**
- Create Docker deployment configuration
- Set up CI/CD pipeline
- Configure monitoring and logging
- Prepare demo environment
- Create backup and recovery procedures

**Deliverables:**
- Complete documentation suite
- Deployment-ready system
- Demo environment
- User training materials

## Success Metrics

**Technical Metrics:**
- Query response time < 2 seconds (95th percentile)
- System handles 1000+ concurrent events
- All 7 scenarios execute successfully
- 95%+ test coverage

**User Experience Metrics:**
- Non-technical users can complete all demo scenarios
- UI responsive on standard web browsers
- Clear error messages and help text
- Intuitive navigation and workflow

---

## Next Steps

**Immediate Actions:**
- Install uv package manager ✅ **COMPLETED**
- Set up local PostgreSQL with Docker ✅ **COMPLETED**
- Install Apache AGE and pgvector extensions ✅ **COMPLETED**
- Configure MCP server for database access ✅ **COMPLETED**
- Initialize Python project structure with uv ✅ **COMPLETED**
- Set up development environment using uv ✅ **COMPLETED**

**Next Development Phase:**
- Implement entity extraction pipeline using OpenAI
- Complete search engine implementation  
- Enhance Gradio UI with graph visualization
- Plan Azure PostgreSQL migration with MCP

**Key Decisions Needed:**
- LLM provider selection (OpenAI vs Azure OpenAI)
- Specific visualization library for graph rendering
- Azure PostgreSQL migration timeline
- Production deployment strategy (Azure Container Instances vs App Service)

**Resource Requirements:**
- **Development**: Local PostgreSQL instance in Docker (✅ Configured)
- **Production**: Azure PostgreSQL instance (Standard tier recommended)
- **MCP Server**: Consistent database interface for both environments 
- **LLM API**: OpenAI access (GPT-4 or equivalent)
- **Development Tools**: VS Code with PostgreSQL extension and MCP integration

**Architecture Notes:**
- **Development Environment**: Local PostgreSQL + Apache AGE + pgvector + MCP
- **Production Environment**: Azure PostgreSQL + Apache AGE + pgvector + MCP  
- **Consistent Interface**: MCP server provides same database access pattern for both environments
- **Migration Path**: Seamless transition from local to Azure using identical MCP interface

This implementation plan provides a comprehensive roadmap for building the Graphiti E-commerce Agent Memory Platform according to the PRD specifications.

## Docker Configuration with uv:
```dockerfile
# Dockerfile
FROM python:3.11-slim

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

# Set working directory
WORKDIR /app

# Copy uv configuration files
COPY pyproject.toml uv.lock ./

# Install dependencies
RUN uv sync --frozen --no-cache

# Copy source code
COPY src/ ./src/
COPY tests/ ./tests/

# Expose port
EXPOSE 8000

# Run the application
CMD ["uv", "run", "src/graphiti/ui/gradio_app.py"]
```

**docker-compose.yml:**
```yaml
version: '3.8'

services:
  postgres:
    image: postgres:15
    environment:
      POSTGRES_DB: graphiti
      POSTGRES_USER: graphiti_user
      POSTGRES_PASSWORD: graphiti_pass
    ports:
      - "5432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./docker/init-db.sql:/docker-entrypoint-initdb.d/init-db.sql

  graphiti-app:
    build: .
    ports:
      - "7860:7860"
    environment:
      - DATABASE_URL=postgresql://graphiti_user:graphiti_pass@postgres:5432/graphiti
      - DEBUG=true
    depends_on:
      - postgres
    volumes:
      - .:/app
    command: uv run --reload src/graphiti/ui/gradio_app.py

volumes:
  postgres_data:
```

## Database Architecture: Local Development + Azure Production

### Development Environment
```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Python App    │────│   MCP Server    │────│ Local PostgreSQL│
│   (Graphiti)    │    │                 │    │ + AGE + pgvector│
│                 │    │                 │    │   (Docker)      │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

### Production Environment  
```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Python App    │────│   MCP Server    │────│ Azure PostgreSQL│
│   (Deployed)    │    │   (Same Code)   │    │ + AGE + pgvector│
│                 │    │                 │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

### Key Benefits
- **Consistent Interface**: Same MCP calls work in both environments
- **Easy Migration**: Change connection string, keep all code the same
- **Development Speed**: Local development with full PostgreSQL features
- **Production Ready**: Azure PostgreSQL with enterprise features

---
