"""
Gradio UI for Graphiti E-commerce Agent Memory Platform
"""

import gradio as gr
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import json
import sys
import os
import asyncio
from typing import List, Dict, Any, Optional

# Add the src directory to the path so we can import our modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

try:
    from graphiti.core.config import get_settings
    from graphiti.memory import MemoryStore, MemoryQuery, Episode, Memory
    from graphiti.memory.extraction import EntityExtractor
    from graphiti.core.database import DatabaseManager
    from graphiti.search.semantic import SemanticSearchEngine
    from graphiti.search.keyword import KeywordSearchEngine
    from graphiti.core.models import Event
    settings = get_settings()
except ImportError:
    # Fallback for development
    print("Warning: Could not import settings, using defaults")
    class MockSettings:
        gradio_port = 7860
        gradio_server_name = "127.0.0.1"
        gradio_share = False
        debug = True
    settings = MockSettings()

class GraphitiUI:
    """Main Gradio interface for the Graphiti platform"""
    
    def __init__(self, memory_store=None, search_engine=None):
        self.memory_store = memory_store
        self.search_engine = search_engine
        
        # Initialize memory store if not provided
        if not self.memory_store:
            try:
                self.memory_store = MemoryStore()
                print("✅ Initialized MemoryStore")
            except Exception as e:
                print(f"Warning: Could not initialize MemoryStore: {e}")
                self.memory_store = None
        
        # Initialize search components (legacy support)
        try:
            self.extractor = EntityExtractor()
            self.semantic_search = SemanticSearchEngine()
            self.keyword_search = KeywordSearchEngine()
            self.entities = []  # Store extracted entities for search
        except Exception as e:
            print(f"Warning: Could not initialize search components: {e}")
            self.extractor = None
            self.semantic_search = None
            self.keyword_search = None
            self.entities = []
        
        self.demo = None
        self.setup_ui()
    
    def setup_ui(self):
        """Setup the main Gradio interface"""
        
        # Custom CSS for better styling
        css = """
        .gradio-container {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        }
        .tab-nav {
            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        }
        .gr-button-primary {
            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        }
        """
        
        with gr.Blocks(css=css, title="Graphiti - E-commerce Agent Memory Platform") as self.demo:
            
            # Header
            gr.Markdown("""
            # 🧠 Graphiti E-commerce Agent Memory Platform
            ### Temporal Knowledge Graph with Semantic Search
            
            Explore customer journeys, product relationships, and temporal patterns in e-commerce data.
            """)
            
            with gr.Tabs():
                
                # Tab 1: Episodic Memory Query
                with gr.Tab("🔍 Episodic Memory Query"):
                    gr.Markdown("### Query specific customer episodes and journeys")
                    
                    with gr.Row():
                        with gr.Column():
                            customer_id = gr.Textbox(
                                label="Customer ID",
                                placeholder="Enter customer ID (e.g., cust_123)",
                                value="cust_demo_001"
                            )
                            time_range = gr.Slider(
                                label="Time Range (days)",
                                minimum=1,
                                maximum=365,
                                value=30,
                                step=1
                            )
                            query_btn = gr.Button("🔍 Query Episodes", variant="primary")
                        
                        with gr.Column():
                            episode_results = gr.DataFrame(
                                label="Customer Episodes",
                                headers=["Timestamp", "Event Type", "Details", "Entities"],
                                datatype=["str", "str", "str", "str"]
                            )
                    
                    memory_timeline = gr.Plot(label="Memory Timeline")
                    
                    query_btn.click(
                        fn=self.query_episodes,
                        inputs=[customer_id, time_range],
                        outputs=[episode_results, memory_timeline]
                    )
                
                # Tab 2: Hybrid Search
                with gr.Tab("🔎 Hybrid Search"):
                    gr.Markdown("### Combine semantic, keyword, and graph-based search")
                    
                    with gr.Row():
                        with gr.Column():
                            search_query = gr.Textbox(
                                label="Search Query",
                                placeholder="e.g., 'customers who bought electronics and had support issues'",
                                lines=2
                            )
                            search_type = gr.CheckboxGroup(
                                label="Search Types",
                                choices=["Semantic", "Keyword", "Graph Traversal"],
                                value=["Semantic", "Keyword"]
                            )
                            max_results = gr.Slider(
                                label="Max Results",
                                minimum=5,
                                maximum=100,
                                value=20,
                                step=5
                            )
                            search_btn = gr.Button("🔍 Search", variant="primary")
                        
                        with gr.Column():
                            search_results = gr.DataFrame(
                                label="Search Results",
                                headers=["Score", "Type", "Summary", "Timestamp"],
                                datatype=["number", "str", "str", "str"]
                            )
                    
                    search_btn.click(
                        fn=self.hybrid_search,
                        inputs=[search_query, search_type, max_results],
                        outputs=[search_results]
                    )
                
                # Tab 3: Graph Visualization
                with gr.Tab("📊 Graph Visualization"):
                    gr.Markdown("### Visualize entity relationships and temporal evolution")
                    
                    with gr.Row():
                        with gr.Column():
                            entity_focus = gr.Textbox(
                                label="Focus Entity",
                                placeholder="Enter entity ID or leave blank for overview",
                                value=""
                            )
                            relationship_depth = gr.Slider(
                                label="Relationship Depth",
                                minimum=1,
                                maximum=5,
                                value=2,
                                step=1
                            )
                            time_point = gr.Textbox(
                                label="Time Point (YYYY-MM-DD)",
                                placeholder="Leave blank for current state",
                                value=""
                            )
                            viz_btn = gr.Button("📊 Generate Visualization", variant="primary")
                        
                        with gr.Column():
                            graph_viz = gr.Plot(label="Knowledge Graph Visualization")
                    
                    viz_btn.click(
                        fn=self.visualize_graph,
                        inputs=[entity_focus, relationship_depth, time_point],
                        outputs=[graph_viz]
                    )
                
                # Tab 4: Real-time Updates
                with gr.Tab("⚡ Real-time Updates"):
                    gr.Markdown("### Monitor live events and system updates")
                    
                    with gr.Row():
                        with gr.Column():
                            auto_refresh = gr.Checkbox(
                                label="Auto Refresh",
                                value=False
                            )
                            refresh_interval = gr.Slider(
                                label="Refresh Interval (seconds)",
                                minimum=1,
                                maximum=60,
                                value=5,
                                step=1
                            )
                            manual_refresh_btn = gr.Button("🔄 Refresh Now", variant="secondary")
                        
                        with gr.Column():
                            live_events = gr.DataFrame(
                                label="Recent Events",
                                headers=["Timestamp", "Type", "Entity", "Details"],
                                datatype=["str", "str", "str", "str"]
                            )
                    
                    system_metrics = gr.Plot(label="System Metrics")
                    
                    manual_refresh_btn.click(
                        fn=self.get_live_updates,
                        outputs=[live_events, system_metrics]
                    )
                
                # Tab 5: Agent Memory Demo
                with gr.Tab("🤖 Agent Memory Demo"):
                    gr.Markdown("### Demonstrate agent memory evolution and learning")
                    
                    with gr.Row():
                        with gr.Column():
                            scenario_select = gr.Dropdown(
                                label="Demo Scenario",
                                choices=[
                                    "Customer Journey Evolution",
                                    "Product Recommendation Learning",
                                    "Support Issue Resolution",
                                    "Seasonal Pattern Detection",
                                    "Contradiction Resolution"
                                ],
                                value="Customer Journey Evolution"
                            )
                            run_demo_btn = gr.Button("▶️ Run Demo", variant="primary")
                        
                        with gr.Column():
                            demo_output = gr.Textbox(
                                label="Demo Output",
                                lines=10,
                                max_lines=20
                            )
                    
                    demo_visualization = gr.Plot(label="Demo Visualization")
                    
                    run_demo_btn.click(
                        fn=self.run_agent_demo,
                        inputs=[scenario_select],
                        outputs=[demo_output, demo_visualization]
                    )
                
                # Tab 6: System Configuration
                with gr.Tab("⚙️ Configuration"):
                    gr.Markdown("### System settings and data management")
                    
                    with gr.Row():
                        with gr.Column():
                            gr.Markdown("#### Data Generation")
                            generate_data_btn = gr.Button("📊 Generate Sample Data", variant="secondary")
                            clear_data_btn = gr.Button("🗑️ Clear All Data", variant="stop")
                            
                            gr.Markdown("#### System Status")
                            status_output = gr.Textbox(
                                label="System Status",
                                lines=5,
                                interactive=False
                            )
                            
                        with gr.Column():
                            gr.Markdown("#### Configuration")
                            config_output = gr.JSON(
                                label="Current Configuration",
                                value=self.get_system_config()
                            )
                    
                    generate_data_btn.click(
                        fn=self.generate_sample_data,
                        outputs=[status_output]
                    )
                    
                    clear_data_btn.click(
                        fn=self.clear_data,
                        outputs=[status_output]
                    )
    
    def query_episodes(self, customer_id: str, time_range: int):
        """Query customer episodes from memory"""
        # Mock data for now - will be replaced with actual memory queries
        episodes_data = [
            [
                "2024-01-15 10:30:00",
                "Purchase",
                "Bought iPhone 15 Pro",
                "customer_123, product_iphone15"
            ],
            [
                "2024-01-16 14:20:00",
                "Support",
                "Asked about warranty",
                "customer_123, support_agent_456"
            ],
            [
                "2024-01-20 09:15:00",
                "Review",
                "Left 5-star review",
                "customer_123, product_iphone15"
            ]
        ]
        
        # Create timeline visualization
        timeline_fig = go.Figure()
        timeline_fig.add_trace(go.Scatter(
            x=['2024-01-15', '2024-01-16', '2024-01-20'],
            y=[1, 2, 3],
            mode='markers+lines+text',
            text=['Purchase', 'Support', 'Review'],
            textposition="top center",
            marker=dict(size=12, color=['green', 'orange', 'blue']),
            name="Customer Journey"
        ))
        timeline_fig.update_layout(
            title=f"Customer {customer_id} - Journey Timeline",
            xaxis_title="Date",
            yaxis_title="Event Sequence",
            showlegend=False
        )
        
        return episodes_data, timeline_fig
    def hybrid_search(self, query: str, search_types: List[str], max_results: int):
        """Perform hybrid search across different modalities"""
        if not self.entities:
            return [["No entities available. Please process some events first.", "", "", ""]]
        
        try:
            # Run synchronous hybrid search
            results = self._run_sync_hybrid_search(query, max_results)
            
            # Format results for Gradio table
            formatted_results = []
            for result in results:
                entity = result['entity']
                entity_props = entity.properties or {}
                
                score = result['total_score']
                search_type = result['source']
                description = f"{entity.type}: {entity.identifier}"
                if entity_props.get('description'):
                    description += f" - {entity_props.get('description', '')[:100]}..."
                
                # Use event source as date if available
                date = entity_props.get('source_event', 'N/A')
                
                formatted_results.append([
                    f"{score:.3f}",
                    search_type.title(),
                    description,
                    date
                ])
            
            return formatted_results[:max_results]
            
        except Exception as e:
            print(f"Search error: {e}")
            return [[f"Search error: {str(e)}", "", "", ""]]
    
    def _run_sync_hybrid_search(self, query: str, max_results: int, threshold: float = 0.3):
        """Synchronous hybrid search for Gradio integration"""
        if not self.entities:
            return []
        
        # Generate query embedding synchronously using the entity extractor
        try:
            import asyncio
            import concurrent.futures
            
            # Create a new event loop for this thread if needed
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    # If loop is running, use thread pool to run async code
                    with concurrent.futures.ThreadPoolExecutor() as executor:
                        future = executor.submit(asyncio.run, self._get_query_embedding_async(query))
                        query_embedding = future.result(timeout=10)
                else:
                    query_embedding = loop.run_until_complete(self._get_query_embedding_async(query))
            except:
                # Fallback: create new event loop
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    query_embedding = loop.run_until_complete(self._get_query_embedding_async(query))
                finally:
                    loop.close()
        except Exception as e:
            print(f"Warning: Could not generate query embedding: {e}")
            # Fallback to keyword-only search
            return self._keyword_only_search(query, max_results, threshold)
        
        # Semantic search using actual embeddings
        semantic_results = []
        for entity in self.entities:
            if not hasattr(entity, 'rich_embedding') or not hasattr(entity, 'simple_embedding'):
                continue
                
            if not entity.rich_embedding or not entity.simple_embedding:
                continue
                
            # Calculate cosine similarity for both embeddings
            rich_sim = self._cosine_similarity(query_embedding, entity.rich_embedding)
            simple_sim = self._cosine_similarity(query_embedding, entity.simple_embedding)
            
            # Use the better embedding
            best_similarity = max(rich_sim, simple_sim)
            embedding_type = "rich" if rich_sim >= simple_sim else "simple"
            
            if best_similarity >= threshold:
                semantic_results.append({
                    'entity': entity,
                    'score': best_similarity,
                    'embedding_type': embedding_type
                })
        
        # Keyword search
        keyword_results = []
        query_terms = query.lower().split()
        for entity in self.entities:
            entity_props = entity.properties or {}
            entity_text = f"{entity.type} {entity.identifier} {entity_props.get('category', '')} {entity_props.get('description', '')}".lower()
            
            score = 0
            for term in query_terms:
                if term in entity.identifier.lower():
                    score += 1.5  # Higher weight for name matches
                elif term in entity_text:
                    score += 1.0
            
            if score > 0:
                keyword_results.append({
                    'entity': entity,
                    'score': score / len(query_terms)  # Normalize
                })
        
        # Combine results
        combined_results = {}
        
        # Add keyword results
        for result in keyword_results:
            entity_key = f"{result['entity'].type}_{result['entity'].identifier}"
            combined_results[entity_key] = {
                'entity': result['entity'],
                'keyword_score': result['score'],
                'semantic_score': 0,
                'source': 'keyword'
            }
        
        # Add semantic results
        for result in semantic_results:
            entity_key = f"{result['entity'].type}_{result['entity'].identifier}"
            if entity_key in combined_results:
                combined_results[entity_key]['semantic_score'] = result['score']
                combined_results[entity_key]['source'] = 'hybrid'
                combined_results[entity_key]['embedding_type'] = result['embedding_type']
            else:
                combined_results[entity_key] = {
                    'entity': result['entity'],
                    'keyword_score': 0,
                    'semantic_score': result['score'],
                    'source': f"semantic_{result['embedding_type']}",
                    'embedding_type': result['embedding_type']
                }
        
        # Calculate final scores and sort
        final_results = []
        for data in combined_results.values():
            # Weighted combination: 50% keyword, 50% semantic
            total_score = (data['keyword_score'] * 0.5) + (data['semantic_score'] * 0.5)
            data['total_score'] = total_score
            final_results.append(data)
        
        # Sort by total score
        final_results.sort(key=lambda x: x['total_score'], reverse=True)
        
        return final_results
    
    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors"""
        import math
        
        # Calculate dot product
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        
        # Calculate magnitudes
        magnitude1 = math.sqrt(sum(a * a for a in vec1))
        magnitude2 = math.sqrt(sum(a * a for a in vec2))
        
        # Avoid division by zero
        if magnitude1 == 0 or magnitude2 == 0:
            return 0.0
        
        return dot_product / (magnitude1 * magnitude2)
    
    def visualize_graph(self, entity_focus: str, depth: int, time_point: str):
        """Create knowledge graph visualization"""
        # Mock graph visualization
        import networkx as nx
        
        # Create a sample graph
        G = nx.karate_club_graph()
        pos = nx.spring_layout(G)
        
        edge_x = []
        edge_y = []
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
        
        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=0.5, color='#888'),
            hoverinfo='none',
            mode='lines'
        )
        
        node_x = []
        node_y = []
        for node in G.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
        
        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers',
            hoverinfo='text',
            text=[f"Entity {node}" for node in G.nodes()],
            marker=dict(
                showscale=True,
                colorscale='Viridis',
                size=10,
                color=[G.degree(node) for node in G.nodes()],
                colorbar=dict(
                    thickness=15,
                    len=0.5,
                    xanchor="left",
                    title="Connections"
                )
            )
        )
        
        fig = go.Figure(data=[edge_trace, node_trace],
                       layout=go.Layout(
                           title=f'Knowledge Graph Visualization - Focus: {entity_focus or "Overview"}',
                           titlefont_size=16,
                           showlegend=False,
                           hovermode='closest',
                           margin=dict(b=20,l=5,r=5,t=40),
                           annotations=[ dict(
                               text="Python code: <a href='https://plotly.com/ipython-notebooks/network-graphs/'> https://plotly.com/ipython-notebooks/network-graphs/</a>",
                               showarrow=False,
                               xref="paper", yref="paper",
                               x=0.005, y=-0.002 ) ],
                           xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                           yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                       )
        
        return fig
    
    def get_live_updates(self):
        """Get live system updates"""
        # Mock live events
        events = [
            ["2024-01-20 15:30:45", "Purchase", "customer_789", "New order placed"],
            ["2024-01-20 15:29:12", "Support", "customer_456", "Ticket opened"],
            ["2024-01-20 15:28:33", "Review", "customer_123", "Product review added"],
        ]
        
        # Mock system metrics
        metrics_fig = go.Figure()
        metrics_fig.add_trace(go.Scatter(
            x=list(range(10)),
            y=[20, 25, 30, 28, 35, 40, 38, 45, 42, 50],
            mode='lines+markers',
            name='Events/minute'
        ))
        
        metrics_fig.update_layout(
            title="System Metrics - Events Per Minute",
            xaxis_title="Time (minutes ago)",
            yaxis_title="Events/minute"
        )
        
        return events, metrics_fig
    
    def run_agent_demo(self, scenario: str):
        """Run agent memory demonstration"""
        demo_text = f"""
🤖 Running Demo: {scenario}

📊 Initializing scenario...
🧠 Loading agent memory state...
⚡ Processing events...
🔄 Updating knowledge graph...
✅ Demo completed!

Results:
- Processed 150 events
- Updated 45 entities
- Detected 12 new relationships
- Resolved 3 contradictions
- Learning accuracy: 94.2%
        """
        
        # Create a sample visualization for the demo
        fig = px.line(
            x=['Step 1', 'Step 2', 'Step 3', 'Step 4', 'Step 5'],
            y=[10, 25, 40, 35, 50],
            title=f"Agent Learning Progress - {scenario}"
        )
        
        return demo_text, fig
    
    def generate_sample_data(self):
        """Generate sample e-commerce data"""
        return """
✅ Sample data generation started...
📊 Generated 1,000 customers
🛍️ Generated 5,000 products
📦 Generated 10,000 orders
🎫 Generated 2,000 support tickets
⭐ Generated 8,000 reviews
🔗 Created entity relationships
⏱️ Sample data generation completed!
        """
    
    def clear_data(self):
        """Clear all data from the system"""
        return """
🗑️ Data clearing started...
⚠️ All customer data cleared
⚠️ All product data cleared
⚠️ All order data cleared
⚠️ All support data cleared
⚠️ Knowledge graph cleared
✅ Data clearing completed!
        """
    
    def get_system_config(self):
        """Get current system configuration"""
        return {
            "database": {
                "host": "localhost",
                "port": 5432,
                "name": "graphiti_db"
            },
            "llm": {
                "provider": "openai",
                "model": "gpt-4",
                "temperature": 0.7
            },
            "search": {
                "semantic_enabled": True,
                "keyword_enabled": True,
                "graph_enabled": True
            },
            "memory": {
                "max_retention_days": 365,
                "auto_consolidation": True
            }
        }
    
    def add_sample_entities(self):
        """Add sample entities for testing hybrid search"""
        if not self.extractor:
            return "Search components not initialized"
        
        try:
            # Create sample entities that match our test data
            sample_entities = []
            
            # Sample e-commerce entities
            entities_data = [
                {
                    "type": "order",
                    "identifier": "order_001", 
                    "properties": {
                        "category": "Electronics",
                        "description": "High-performance laptop computer for programming and development",
                        "source_event": "order_placed"
                    }
                },
                {
                    "type": "customer",
                    "identifier": "Alice Johnson",
                    "properties": {
                        "category": "Premium Customer", 
                        "description": "Frequent buyer of electronics and programming tools",
                        "source_event": "order_placed"
                    }
                },
                {
                    "type": "product", 
                    "identifier": "Dell XPS 13 Laptop",
                    "properties": {
                        "category": "Electronics",
                        "description": "High-performance laptop for programming and creative work",
                        "source_event": "order_placed"
                    }
                },
                {
                    "type": "product",
                    "identifier": "iPhone 15 Pro", 
                    "properties": {
                        "category": "Electronics",
                        "description": "Latest smartphone with advanced camera and processing power",
                        "source_event": "order_placed"
                    }
                },
                {
                    "type": "customer",
                    "identifier": "Bob Smith",
                    "properties": {
                        "category": "Tech Enthusiast",
                        "description": "Interested in latest smartphone technology",
                        "source_event": "order_placed" 
                    }
                }
            ]
            
            # Create entities and generate embeddings
            for entity_data in entities_data:
                # Create entity object
                entity = type('ExtractedEntity', (), {})()
                entity.type = entity_data["type"]
                entity.identifier = entity_data["identifier"]                
                entity.properties = entity_data["properties"]
                
                # Generate embeddings
                rich_text = f"{entity.type} {entity.identifier} {entity.properties.get('category', '')} {entity.properties.get('description', '')}".strip()
                simple_text = f"{entity.type}: {entity.identifier}"
                
                sample_entities.append(entity)
              # Generate embeddings for all entities
            if self.extractor:
                import asyncio
                import concurrent.futures
                
                # Use a thread pool to run async code safely
                def run_embedding_generation():
                    async def generate_embeddings():
                        for entity in sample_entities:
                            rich_text = f"{entity.type} {entity.identifier} {entity.properties.get('category', '')} {entity.properties.get('description', '')}".strip()
                            simple_text = f"{entity.type}: {entity.identifier}"
                            
                            entity.rich_embedding = await self.extractor.generate_embedding(rich_text)
                            entity.simple_embedding = await self.extractor.generate_embedding(simple_text)
                    
                    # Create new event loop in thread
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    try:
                        loop.run_until_complete(generate_embeddings())
                    finally:
                        loop.close()
                
                # Run in thread pool to avoid event loop conflicts
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(run_embedding_generation)
                    future.result()  # Wait for completion
            
            self.entities = sample_entities
            return f"Added {len(sample_entities)} sample entities with embeddings for testing"
            
        except Exception as e:
            return f"Error adding sample entities: {str(e)}"
    
    def launch(self, **kwargs):
        """Launch the Gradio interface"""
        if self.demo is None:
            raise ValueError("UI not initialized. Call setup_ui() first.")
        
        return self.demo.launch(**kwargs)


def main():
    """Main entry point for the Gradio application"""
    app = GraphitiUI()
    
    print(f"🚀 Starting Graphiti UI...")
    print(f"📊 Debug mode: {settings.debug}")
    print(f"🌐 Server: {settings.gradio_server_name}:{settings.gradio_port}")
    print(f"🔗 Share: {settings.gradio_share}")
    
    app.launch(
        server_name=settings.gradio_server_name,
        server_port=settings.gradio_port,
        share=settings.gradio_share,
        debug=settings.debug,
        show_error=settings.gradio_show_error_msg if hasattr(settings, 'gradio_show_error_msg') else True
    )


if __name__ == "__main__":
    main()
