# Graphiti PostgreSQL: Temporal Knowledge Graph Memory Store

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://python.org)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-16+-blue.svg)](https://postgresql.org)
[![Gradio](https://img.shields.io/badge/Gradio-5.34+-orange.svg)](https://gradio.app)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**A PostgreSQL-based implementation of temporal knowledge graphs for AI agent memory**

*Inspired by Zep AI's Graphiti research with production-ready database architecture*

</div>

## 🎓 Educational Disclaimer

> **📚 IMPORTANT**: This project is created for **educational and research purposes only**. It serves as a demonstration of temporal knowledge graph concepts, AI agent memory systems, and modern Python development practices. While the implementation follows production-ready patterns, please conduct thorough testing and security reviews before considering any production deployment.

---

## 🚀 Features

This implementation provides a PostgreSQL-based temporal knowledge graph system that offers:

- **Bi-Temporal Memory**: Track both when events happened (valid time) and when they were recorded (transaction time)
- **Semantic Understanding**: Vector embeddings for semantic similarity search
- **Dynamic Knowledge Graphs**: Automatically extract entities and relationships from events
- **Hybrid Search**: Combine semantic, keyword, and graph traversal search methods
- **Web Interface**: Gradio-based UI for exploring and visualizing temporal graphs

## 🎥 Demo

**View the system in action:**

<div align="center">

![Graphiti PostgreSQL Demo](assets/Graphiti-PostgreSQL.gif)

*Demo showcasing temporal graph visualization, semantic search, and entity extraction*

</div>

**Features demonstrated:**
- Interactive Gradio interface with memory visualization
- Temporal graph analytics showing memory evolution over time
- Hybrid search capabilities across semantic, keyword, and graph dimensions
- AI entity extraction creating knowledge graphs from raw data
- Bi-temporal timeline views displaying both valid-time and transaction-time

## 💡 The Inspiration: Beyond Traditional Databases

Traditional databases store data as static snapshots. **Graphiti implements temporal knowledge graphs** (inspired by Zep AI's research [[1]](#-academic-references)):

```
Traditional Database          →          Graphiti Temporal Graph
├── Static tables                       ├── Dynamic knowledge web
├── Point-in-time snapshots            ├── Full temporal history
├── Isolated data silos                ├── Connected entity relationships  
├── Manual relationship tracking       ├── Automatic semantic linking
└── No memory of changes               └── Complete audit trail of evolution
```

### Core Innovation: Bi-Temporal Knowledge Graphs

Unlike traditional approaches that lose historical context, Graphiti tracks **two timelines**:

- **Valid Time**: When events actually happened in the real world
- **Transaction Time**: When your system learned about those events

**Example:**
```python
# Alice moved on January 1st (valid time)
# But you discovered this on January 5th (transaction time)
# Graphiti preserves both timelines for complete context

customer_move = TemporalEdge(
    valid_from="2025-01-01",      # When Alice actually moved
    recorded_at="2025-01-05",     # When you learned about it
    relationship="LIVES_AT",
    properties={"confidence": 0.95}
)
```

## 🏗️ Architecture: Three-Tier Knowledge Hierarchy

Based on research from **"Graphiti: A Temporal Knowledge Graph for AI Agent Memory"** by [Zep AI](https://www.getzep.com/) [[1]](#-academic-references), our implementation features:

```
Graphiti Architecture

├── Episode Layer (Raw Memory)
│   ├── Customer interactions, events, messages
│   ├── Preserved in original format (no information loss)
│   └── Timestamped and contextually linked
│
├── Entity Layer (Semantic Knowledge)  
│   ├── AI-extracted entities (customers, products, concepts)
│   ├── Vector embeddings for semantic similarity
│   ├── Dynamic relationship discovery
│   └── Automatic contradiction resolution
│
└── Community Layer (Knowledge Clusters)
    ├── Automatically discovered topic clusters
    ├── Dynamic community evolution
    └── Hierarchical knowledge organization
```

## 🌟 Key Differentiators from the Original Graphiti

While deeply inspired by the Graphiti research from Zep AI [[1]](#-academic-references), our implementation provides several enhancements:

### Production-Ready Enhancements

| **Aspect** | **Original Graphiti** | **Our PostgreSQL Implementation** | **Advantage** |
|------------|-------------------|----------------------|---------------|
| **Database** | Neo4j-based prototype | Production PostgreSQL + pgvector + Apache AGE | Enterprise scalability |
| **Embeddings** | Configurable dimensions | 1536-dimensional OpenAI embeddings | Higher semantic precision |
| **UI** | Python SDK only | Gradio web interface | Visual exploration & debugging |
| **Error Handling** | Basic | Comprehensive logging + error recovery | Production reliability |
| **Search** | Graph-native | Semantic + keyword + graph traversal | Hybrid search capabilities |
| **Deployment** | Local development | Docker + cloud-ready configuration | Scalable deployment |

### Additional Features

1. **Interactive Visualization**: Explore temporal graphs visually, not just programmatically
2. **Real-Time Analytics**: Live dashboards showing memory evolution and patterns
3. **Developer Experience**: Modern Python tooling with `uv`, comprehensive logging
4. **Enterprise Features**: Connection pooling, monitoring, backup strategies
5. **Demo-Ready**: Pre-loaded with realistic e-commerce scenarios

## 🛠️ Technology Stack

```
Infrastructure Stack
├── Python 3.11+ (Modern async/await patterns)
├── uv (Ultra-fast dependency management)  
├── PostgreSQL 16+ (Local or Azure Flexible Server)
├── pgvector (Vector similarity search)
├── Apache AGE (Graph database capabilities)
├── Gradio 5.34+ (Web interfaces)
├── OpenAI GPT-4 (Entity extraction & reasoning)
├── Plotly (Interactive visualizations)
├── Azure Cloud Services (Optional production deployment)
└── SQLAlchemy 2.0+ (Modern ORM patterns)
```

## ⚡ Quick Start

### 1. Clone & Setup
```bash
# Clone the repository
git clone https://github.com/shyamsridhar123/ReMemNetDB
cd graphiti-postgres

# Install with uv
pip install uv
uv install

# Set up environment
cp .env.example .env
# Edit .env with your database and OpenAI credentials
```

### 2. Database Setup

Choose your preferred PostgreSQL deployment option:

#### **Option A: Local PostgreSQL Server**
```bash
# Install PostgreSQL locally (if not already installed)
# Windows: Download from https://www.postgresql.org/download/windows/
# macOS: brew install postgresql
# Linux: sudo apt-get install postgresql postgresql-contrib

# Start PostgreSQL service
# Windows: Start via Services or pg_ctl
# macOS/Linux: sudo systemctl start postgresql

# Create database with extensions
createdb graphiti_db
psql graphiti_db -c "CREATE EXTENSION IF NOT EXISTS vector;"
psql graphiti_db -c "CREATE EXTENSION IF NOT EXISTS age;"

# Update .env file with local connection
DATABASE_URL=postgresql://username:password@localhost:5432/graphiti_db
```

#### **Option B: Azure PostgreSQL (Recommended for Production)**
```bash
# 1. Create Azure PostgreSQL Flexible Server
az postgres flexible-server create \
    --resource-group myResourceGroup \
    --name graphiti-postgres-server \
    --location eastus \
    --admin-user graphiti_admin \
    --admin-password YourSecurePassword123! \
    --sku-name Standard_D2s_v3 \
    --tier GeneralPurpose \
    --public-access 0.0.0.0 \
    --storage-size 128 \
    --version 16

# 2. Create database
az postgres flexible-server db create \
    --resource-group myResourceGroup \
    --server-name graphiti-postgres-server \
    --database-name graphiti_db

# 3. Install extensions (connect via Azure portal or psql)
psql "host=graphiti-postgres-server.postgres.database.azure.com port=5432 dbname=graphiti_db user=graphiti_admin password=YourSecurePassword123! sslmode=require" \
    -c "CREATE EXTENSION IF NOT EXISTS vector;" \
    -c "CREATE EXTENSION IF NOT EXISTS age;"

# 4. Update .env file with Azure connection
DATABASE_URL=postgresql://graphiti_admin:YourSecurePassword123!@graphiti-postgres-server.postgres.database.azure.com:5432/graphiti_db?sslmode=require
```

#### **Complete Setup**
```bash
# Run database migrations (after choosing local or Azure)
uv run alembic upgrade head

# Verify installation
uv run python -c "from graphiti.core.database import DatabaseManager; dm = DatabaseManager(); print('✅ Database connection successful!')"
```

#### **Environment Configuration**

Create your `.env` file with the appropriate settings:

```bash
# Copy example configuration
cp .env.example .env
```

**For Local PostgreSQL:**
```env
# Database Configuration
DATABASE_URL=postgresql://username:password@localhost:5432/graphiti_db
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=graphiti_db
POSTGRES_USER=your_username
POSTGRES_PASSWORD=your_password

# OpenAI Configuration
OPENAI_API_KEY=your_openai_api_key_here

# Gradio Configuration  
GRADIO_PORT=7860
GRADIO_SERVER_NAME=127.0.0.1
GRADIO_SHARE=false
DEBUG=true
```

**For Azure PostgreSQL:**
```env
# Database Configuration
DATABASE_URL=postgresql://graphiti_admin:YourSecurePassword123!@graphiti-postgres-server.postgres.database.azure.com:5432/graphiti_db?sslmode=require
POSTGRES_HOST=graphiti-postgres-server.postgres.database.azure.com
POSTGRES_PORT=5432
POSTGRES_DB=graphiti_db
POSTGRES_USER=graphiti_admin
POSTGRES_PASSWORD=YourSecurePassword123!
POSTGRES_SSL_MODE=require

# OpenAI Configuration
OPENAI_API_KEY=your_openai_api_key_here

# Gradio Configuration
GRADIO_PORT=7860
GRADIO_SERVER_NAME=0.0.0.0  # For cloud deployment
GRADIO_SHARE=false
DEBUG=false  # Set to false for production
```

### 3. Launch the Application
```bash
# Start the Gradio interface
uv run python src/graphiti/ui/enhanced_gradio_app.py

# Open http://localhost:7860 and explore
```

### 4. Try It Out
```python
# Generate sample e-commerce data
uv run python sample_data_generator.py

# Explore the temporal graph features:
# - Customer journeys across time
# - Product relationships and evolution  
# - Semantic search across memories
# - Visual graph exploration
```

## 🎯 Use Cases

### E-Commerce Intelligence
- **Customer Journey Mapping**: Track how preferences evolve over time
- **Product Recommendation**: Semantic understanding of customer intent
- **Fraud Detection**: Temporal pattern analysis across interactions
- **Inventory Intelligence**: Predict demand based on historical relationships

### AI Agent Memory
- **Conversation Context**: Maintain rich context across long interactions  
- **Learning Evolution**: Track how agent understanding improves
- **Personalization**: Deep customer memory for tailored experiences
- **Knowledge Discovery**: Automatically discover new insights from interactions

### Business Analytics
- **Temporal Analysis**: "How did customer sentiment change after our product launch?"
- **Relationship Discovery**: "Which customers influence others' purchase decisions?"
- **Contradiction Detection**: Identify and resolve conflicting information
- **Predictive Insights**: Forecast trends based on temporal patterns

## 🌈 The UI Experience

Our Gradio interface provides:

### Interactive Dashboards
- **Real-time memory statistics** with live updates
- **Visual graph exploration** with zooming and filtering  
- **Temporal timeline views** showing memory evolution
- **Multi-modal search** combining text, semantic, and graph queries

### Visual Graph Explorer
- **Dynamic node positioning** with physics-based layouts
- **Color-coded entity types** for instant recognition
- **Interactive relationship exploration** with click-to-expand
- **Temporal playback** to watch memories form over time

### Memory Inspector
- **Detailed entity views** with full property inspection
- **Relationship timeline** showing connection evolution  
- **Confidence scoring** for AI-extracted information
- **Contradiction highlighting** with resolution suggestions

## 🔬 Technical Implementation

### Bi-Temporal Data Model
```python
class TemporalNode(Base):
    # When the fact was true in reality
    valid_from = Column(DateTime(timezone=True), nullable=False)
    valid_to = Column(DateTime(timezone=True))
    
    # When you recorded this information  
    recorded_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Rich semantic properties
    properties = Column(JSONB)
    embedding = Column(Vector(1536))  # OpenAI embeddings
```

### Entity Extraction
```python
class EntityExtractor:
    """AI-powered entity extraction with confidence scoring"""
    
    async def extract_entities(self, event: Event) -> List[ExtractedEntity]:
        # Use GPT-4 with contextual understanding
        # Apply confidence thresholds
        # Resolve entity duplicates
        # Generate semantic embeddings
```

### Hybrid Search Engine
```python
class HybridSearchEngine:
    """Combine semantic, keyword, and graph search"""
    
    async def search(self, query: str) -> SearchResults:
        # Semantic similarity with vector search
        # Keyword matching with full-text search  
        # Graph traversal for relationship discovery
        # Intelligent result fusion and ranking
```

## 🎓 Documentation

### Documentation Deep Dive
- [`TEMPORAL_GRAPH_ANALYSIS.md`](docs/TEMPORAL_GRAPH_ANALYSIS.md) - Understanding bi-temporal concepts
- [`HYBRID_SEARCH_IMPLEMENTATION.md`](docs/HYBRID_SEARCH_IMPLEMENTATION.md) - Search architecture details  
- [`COMPREHENSIVE_DEMO_GUIDE.md`](docs/COMPREHENSIVE_DEMO_GUIDE.md) - Complete feature showcase
- [`GRAPHITI_ALIGNMENT_ANALYSIS.md`](docs/GRAPHITI_ALIGNMENT_ANALYSIS.md) - Research paper comparison

### Quick References
- [`QUICK_DEMO_REFERENCE.md`](docs/QUICK_DEMO_REFERENCE.md) - Get started in minutes
- [`IMPLEMENTATION_PLAN.md`](docs/IMPLEMENTATION_PLAN.md) - Architecture decisions


### 🤝 Contributing

We welcome contributions! Whether you're:
- **Bug Hunters**: Found an issue? Open an issue or PR
- **Feature Developers**: Have an idea? Let's discuss it  
- **Documentation Writers**: Help make our docs even better
- **Test Writers**: Help us build bulletproof reliability

## 🏆 Project Goals

**Graphiti PostgreSQL** aims to provide a production-ready implementation of temporal knowledge graphs for AI agent memory systems.

### Target Audience:
- **AI/ML Engineers** building sophisticated agent systems
- **Data Scientists** exploring temporal relationship analysis  
- **Enterprise Teams** needing scalable memory solutions
- **Researchers** working on knowledge graph applications
- **Developers** building intelligent applications

<div align="center">

## Getting Started

**[⚡ Quick Start](#-quick-start)** | **[📚 Read the Docs](docs/)** | **[🎮 Try the Demo](http://localhost:7860)**

**Star ⭐ this repo if Graphiti PostgreSQL is useful for your projects!**

</div>

---

## 📖 Academic References

**[1]** Zep AI Team. "Graphiti: A Temporal Knowledge Graph for AI Agent Memory." *Zep AI Research*, 2024. Available at: [https://www.getzep.com/graphiti](https://www.getzep.com/graphiti)

**Key Concepts Referenced:**
- Bi-temporal knowledge representation (valid-time vs transaction-time)
- Three-tier architecture: Episodes → Entities → Communities
- Semantic similarity search with vector embeddings
- Automatic entity extraction and relationship discovery
- Temporal contradiction detection and resolution

**Research Paper Access:**
- 📄 Original paper: [arXiv:2501.13956v1](https://arxiv.org/abs/2501.13956v1) (if available)
- 🌐 Project homepage: [https://www.getzep.com/graphiti](https://www.getzep.com/graphiti)
- 💻 Original implementation: [https://github.com/getzep/graphiti](https://github.com/getzep/graphiti)

**Citation (BibTeX):**
```bibtex
@misc{zep2024graphiti,
  title={Graphiti: A Temporal Knowledge Graph for AI Agent Memory},
  author={Zep AI Team},
  year={2024},
  publisher={Zep AI},
  url={https://www.getzep.com/graphiti},
  note={Accessed: July 2025}
}
```

**Acknowledgments:**
This implementation is inspired by and builds upon the foundational work of the Zep AI team in developing Graphiti. While our PostgreSQL-based implementation differs significantly in technical architecture and scope, the core concepts of temporal knowledge graphs for AI agent memory remain faithful to their original research. We extend our gratitude to the Zep AI team for their pioneering work in this field.
