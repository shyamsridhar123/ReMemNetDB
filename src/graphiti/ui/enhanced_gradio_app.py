"""
Enhanced Gradio UI for Graphiti PostgreSQL Memory Store
Showcasing all implemented features with real backend integration
"""

import gradio as gr
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import json
import sys
import os
from typing import List, Dict, Any, Optional
import uuid

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from graphiti.memory import MemoryStore, MemoryQuery, Episode
from graphiti.ui.memory_integration import MemoryStoreUIAdapter
from graphiti.core.config import get_settings

class EnhancedGraphitiUI:
    """Enhanced Gradio interface showcasing all Graphiti features"""
    
    def __init__(self):
        self.settings = get_settings()
        self.memory_store = None
        self.adapter = None
        self.demo = None
        
        # Initialize components
        self._initialize_memory_store()
        self._setup_ui()
    
    def _initialize_memory_store(self):
        """Initialize the memory store and adapter"""
        try:
            self.memory_store = MemoryStore()
            self.adapter = MemoryStoreUIAdapter(self.memory_store)
            print("✅ MemoryStore initialized successfully")
        except Exception as e:
            print(f"❌ Failed to initialize MemoryStore: {e}")
            self.memory_store = None
            self.adapter = None
    
    def _setup_ui(self):
        """Setup the enhanced Gradio interface"""
        
        # Enhanced CSS
        css = """
        .gradio-container {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            max-width: 1400px;
            margin: 0 auto;
        }
        .tab-nav {
            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        }
        .gr-button-primary {
            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
            border: none;
            border-radius: 8px;
        }
        .status-success {
            background-color: #d4edda;
            border: 1px solid #c3e6cb;
            color: #155724;
            padding: 10px;
            border-radius: 5px;
        }
        .status-error {
            background-color: #f8d7da;
            border: 1px solid #f5c6cb;
            color: #721c24;
            padding: 10px;
            border-radius: 5px;
        }
        """
        
        with gr.Blocks(css=css, title="Graphiti PostgreSQL - Memory Store Demo") as self.demo:
            
            # Header
            gr.Markdown("""
            # 🧠 Graphiti PostgreSQL Memory Store
            ## Temporal Knowledge Graph with Hybrid Search
            
            **Features Demonstrated:**
            - ✅ Entity Extraction & Embedding Generation
            - ✅ PostgreSQL Vector Storage  
            - ✅ Semantic, Keyword, Hybrid & Temporal Search
            - ✅ Episode Retrieval & Customer Journey Analysis
            - ✅ Real-time Memory Analytics
            """)
            
            # Status indicator
            with gr.Row():
                status_indicator = gr.HTML(
                    value=self._get_status_html(),
                    label="System Status"
                )
            
            with gr.Tabs():
                
                # Tab 1: Event Storage & Processing
                with gr.Tab("📝 Event Storage & Processing"):
                    gr.Markdown("### Store events and watch entity extraction in action")
                    
                    with gr.Row():
                        with gr.Column(scale=1):
                            gr.Markdown("#### Event Input")
                            event_type = gr.Dropdown(
                                choices=["order_placed", "product_viewed", "support_request", "review_posted"],
                                label="Event Type",
                                value="order_placed"
                            )
                            customer_id_input = gr.Textbox(
                                label="Customer ID",
                                value=str(uuid.uuid4()),
                                placeholder="Auto-generated UUID"
                            )
                            customer_name = gr.Textbox(
                                label="Customer Name",
                                value="Alice Johnson",
                                placeholder="Enter customer name"
                            )
                            product_name = gr.Textbox(
                                label="Product Name", 
                                value="Dell XPS 13 Laptop",
                                placeholder="Enter product name"
                            )
                            
                            with gr.Row():
                                price = gr.Number(
                                    label="Price ($)",
                                    value=1299.99,
                                    minimum=0
                                )
                                category = gr.Textbox(
                                    label="Category",
                                    value="Electronics"
                                )
                            
                            store_event_btn = gr.Button("🚀 Store Event", variant="primary")
                            
                        with gr.Column(scale=1):
                            gr.Markdown("#### Entity Extraction Results")
                            extraction_results = gr.DataFrame(
                                headers=["Entity Type", "Identifier", "Properties", "Embedding Size"],
                                label="Extracted Entities",
                                interactive=False
                            )
                            
                            event_storage_status = gr.Textbox(
                                label="Storage Status",
                                interactive=False,
                                lines=3
                            )
                    
                    # Event processing
                    store_event_btn.click(
                        fn=self.store_event_demo,
                        inputs=[event_type, customer_id_input, customer_name, product_name, price, category],
                        outputs=[extraction_results, event_storage_status]
                    )
                
                # Tab 2: Customer Journey Analysis
                with gr.Tab("🔍 Customer Journey Analysis"):
                    gr.Markdown("### Analyze customer episodes and temporal patterns")
                    
                    with gr.Row():
                        with gr.Column(scale=1):
                            journey_customer_id = gr.Textbox(
                                label="Customer ID",
                                placeholder="Enter customer ID to analyze"
                            )
                            journey_days = gr.Slider(
                                label="Days to Look Back",
                                minimum=1,
                                maximum=90,
                                value=30,
                                step=1
                            )
                            analyze_journey_btn = gr.Button("🔍 Analyze Journey", variant="primary")
                            
                            # Sample data button
                            sample_customer_btn = gr.Button("📊 Load Sample Customer Data", variant="secondary")
                        
                        with gr.Column(scale=2):
                            journey_timeline = gr.Plot(
                                label="Customer Journey Timeline",
                                value=self._create_empty_plot("No customer data loaded")
                            )
                    
                    # Episode details
                    episode_details = gr.DataFrame(
                        headers=["Timestamp", "Event Types", "Episode Summary", "Entities Involved"],
                        label="Episode Details",
                        interactive=False
                    )
                    
                    # Event handlers
                    analyze_journey_btn.click(
                        fn=self.analyze_customer_journey,
                        inputs=[journey_customer_id, journey_days],
                        outputs=[episode_details, journey_timeline]
                    )
                    
                    sample_customer_btn.click(
                        fn=self.create_sample_customer_data,
                        outputs=[journey_customer_id, event_storage_status]
                    )
                
                # Tab 3: Hybrid Search Demonstration
                with gr.Tab("🔎 Hybrid Search Demo"):
                    gr.Markdown("### Experience semantic, keyword, and temporal search capabilities")
                    
                    with gr.Row():
                        with gr.Column(scale=1):
                            search_query = gr.Textbox(
                                label="Search Query",
                                placeholder="e.g., 'laptop computer', 'Alice', 'Electronics'",
                                value="laptop computer"
                            )
                            
                            search_types = gr.CheckboxGroup(
                                label="Search Methods",
                                choices=["Semantic", "Keyword", "Temporal"],
                                value=["Semantic", "Keyword"]
                            )
                            
                            max_results = gr.Slider(
                                label="Max Results",
                                minimum=1,
                                maximum=20,
                                value=10,
                                step=1
                            )
                            
                            search_btn = gr.Button("🔍 Search", variant="primary")
                            
                        with gr.Column(scale=2):
                            search_results = gr.DataFrame(
                                headers=["Score", "Search Type", "Entity Description", "Timestamp"],
                                label="Search Results",
                                interactive=False
                            )
                    
                    # Search performance metrics
                    with gr.Row():
                        search_metrics = gr.JSON(
                            label="Search Performance Metrics",
                            value={"query_time": 0, "results_found": 0, "embedding_time": 0}
                        )
                    
                    search_btn.click(
                        fn=self.perform_hybrid_search_demo,
                        inputs=[search_query, search_types, max_results],
                        outputs=[search_results, search_metrics]
                    )
                
                # Tab 4: Memory Analytics Dashboard
                with gr.Tab("📊 Memory Analytics"):
                    gr.Markdown("### Real-time insights into the knowledge graph")
                    
                    with gr.Row():
                        refresh_analytics_btn = gr.Button("🔄 Refresh Analytics", variant="primary")
                    
                    with gr.Row():
                        with gr.Column():
                            entity_stats = gr.DataFrame(
                                headers=["Entity Type", "Count", "Latest Update"],
                                label="Entity Statistics"
                            )
                        
                        with gr.Column():
                            relationship_stats = gr.DataFrame(
                                headers=["Relationship Type", "Count", "Avg Confidence"],
                                label="Relationship Statistics"
                            )
                    
                    with gr.Row():
                        memory_growth_chart = gr.Plot(
                            label="Memory Growth Over Time",
                            value=self._create_empty_plot("Analytics not loaded")
                        )
                    
                    refresh_analytics_btn.click(
                        fn=self.get_memory_analytics,
                        outputs=[entity_stats, relationship_stats, memory_growth_chart]
                    )
                
                # Tab 5: System Configuration
                with gr.Tab("⚙️ System Status"):
                    gr.Markdown("### System configuration and health checks")
                    
                    with gr.Row():
                        check_status_btn = gr.Button("🔍 Check System Status", variant="primary")
                        reset_demo_btn = gr.Button("🔄 Reset Demo Data", variant="secondary")
                    
                    system_status = gr.JSON(
                        label="System Status",
                        value=self._get_system_status()
                    )
                    
                    database_stats = gr.DataFrame(
                        headers=["Table", "Row Count", "Last Updated"],
                        label="Database Statistics"
                    )
                    
                    check_status_btn.click(
                        fn=self.check_system_status,
                        outputs=[system_status, database_stats]
                    )
                    
                    reset_demo_btn.click(
                        fn=self.reset_demo_data,
                        outputs=[event_storage_status]
                    )
    
    def _get_status_html(self) -> str:
        """Get HTML status indicator"""
        if self.memory_store is not None:
            return '<div class="status-success">🟢 <strong>Memory Store:</strong> Connected and Ready</div>'
        else:
            return '<div class="status-error">🔴 <strong>Memory Store:</strong> Not Available</div>'
    
    def _create_empty_plot(self, message: str):
        """Create an empty plot with a message"""
        fig = go.Figure()
        fig.add_annotation(
            text=message,
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(size=16)
        )
        fig.update_layout(
            title=message,
            xaxis=dict(visible=False),
            yaxis=dict(visible=False)
        )
        return fig
    
    def _get_system_status(self) -> Dict[str, Any]:
        """Get current system status"""
        return {
            "memory_store_connected": self.memory_store is not None,
            "database_available": self.memory_store is not None,
            "search_engines_ready": self.adapter is not None,
            "ui_version": "2.0.0",
            "backend_features": [
                "Entity Extraction",
                "Embedding Generation", 
                "Vector Storage",
                "Hybrid Search",
                "Episode Retrieval"
            ]
        }    
    # Implementation methods for UI interactions
    def store_event_demo(self, event_type: str, customer_id: str, customer_name: str, 
                        product_name: str, price: float, category: str):
        """Store an event and show extraction results"""
        if not self.memory_store:
            return [], "❌ Memory store not available"
        
        try:
            # Prepare event data
            event_data = {
                'customer_id': customer_id,
                'customer_name': customer_name,
                'product_name': product_name,
                'price': price,
                'category': category
            }
            
            # Store the event using the real backend
            episode_id = self.memory_store.store_event(
                customer_id=customer_id,
                event_data=event_data,
                event_type=event_type,
                timestamp=datetime.now()
            )
            
            # Get the actual temporal sequence to show real extraction results
            temporal_data = self.memory_store.get_temporal_sequence(customer_id)
            
            # Format real extraction results from the stored data
            extraction_results = []
            if temporal_data and temporal_data.get('nodes'):
                for node in temporal_data['nodes'][-3:]:  # Show last 3 entities created
                    entity_type = node.get('type', 'unknown')
                    properties = node.get('properties', {})
                    identifier = properties.get('identifier', str(node.get('id', 'N/A')))
                    embedding_size = len(node.get('embedding', [])) if node.get('embedding') else 0
                    
                    extraction_results.append([
                        entity_type,
                        identifier,
                        json.dumps(properties, indent=2)[:100] + "..." if len(json.dumps(properties)) > 100 else json.dumps(properties),
                        str(embedding_size)
                    ])
              # If no results from temporal data, show basic confirmation
            if not extraction_results:
                extraction_results = [
                    ["Event stored", event_type, f"Customer: {customer_name}, Product: {product_name}", "Processing..."]
                ]
            
            status = f"✅ Event stored successfully!\n📊 Episode ID: {episode_id}\n🔗 Entities extracted and embedded\n💾 Stored in PostgreSQL with vector embeddings"
            
            return extraction_results, status
            
        except Exception as e:
            return [], f"❌ Error storing event: {str(e)}"
    
    def analyze_customer_journey(self, customer_id: str, days_back: int):
        """Analyze customer journey and episodes"""
        if not self.adapter or not customer_id.strip():
            empty_data = [["No data available", "", "", ""]]
            empty_plot = self._create_empty_plot("Enter a customer ID to analyze")
            return empty_data, empty_plot
        
        try:
            episodes_data, timeline_plot = self.adapter.query_customer_episodes(customer_id, days_back)
            return episodes_data, timeline_plot
        except Exception as e:
            error_data = [[f"Error: {str(e)}", "", "", ""]]
            error_plot = self._create_empty_plot(f"Error loading customer data: {str(e)}")
            return error_data, error_plot
    
    def create_sample_customer_data(self):
        """Create sample customer data for demonstration"""
        if not self.adapter:
            return "", "❌ Memory store not available"
        sample_customer_id = str(uuid.uuid4())
        status = self.adapter.store_sample_events(sample_customer_id)
        
        return sample_customer_id, status
    
    def perform_hybrid_search_demo(self, query: str, search_types: List[str], max_results: int):
        """Perform hybrid search demonstration"""
        if not self.adapter or not query.strip():
            return [["Enter a search query", "", "", ""]], {"error": "No query provided"}
        
        start_time = datetime.now()
        
        try:
            # Use the real backend search
            results = self.adapter.perform_hybrid_search(query, search_types, max_results)
            
            end_time = datetime.now()
            query_time = (end_time - start_time).total_seconds()
            
            metrics = {
                "query_time_ms": round(query_time * 1000, 2),
                "results_found": len(results),
                "search_types_used": search_types,
                "embedding_generated": "Semantic" in search_types
            }
            
            return results, metrics
            
        except Exception as e:
            error_results = [[f"Search error: {str(e)}", "", "", ""]]
            error_metrics = {"error": str(e)}
            return error_results, error_metrics
    
    def get_memory_analytics(self):
        """Get memory analytics data"""
        if not self.adapter:
            # Return empty data if adapter not available
            empty_data = [["No data", "0", "N/A"]]
            empty_fig = go.Figure()
            empty_fig.update_layout(title="Memory store not available")
            return empty_data, empty_data, empty_fig
        try:
            return self.adapter.get_system_analytics()
        except Exception as e:
            # Fallback to error data if real analytics fail
            error_data = [[f"Error: {str(e)}", "0", "N/A"]]
            error_fig = go.Figure()
            error_fig.update_layout(title=f"Analytics error: {str(e)}")
            return error_data, error_data, error_fig
    
    def check_system_status(self):
        """Check detailed system status"""        
        if not self.adapter:
            error_status = '<div class="status-error">❌ Memory store not available</div>'
            error_stats = [["Error", "0", "N/A"]]
            return error_status, error_stats
        
        try:
            return self.adapter.get_system_status()
        except Exception as e:
            error_status = f'<div class="status-error">❌ System error: {str(e)}</div>'
            error_stats = [[f"Error: {str(e)}", "0", "N/A"]]
            return error_status, error_stats
    
    def reset_demo_data(self):
        """Reset demonstration data"""
        if not self.memory_store:
            return "❌ Memory store not available"
        
        try:
            # Clear demo data - this would need to be implemented in the memory store
            # For now, return a message indicating what would happen
            return "⚠️ Demo data reset functionality would clear:\n• All stored events\n• Extracted entities\n• Generated embeddings\n• Customer episodes\n\n(Implementation pending)"
        except Exception as e:
            return f"❌ Error resetting data: {str(e)}"
    
    def launch(self, **kwargs):
        """Launch the Gradio interface"""
        if self.demo is None:
            raise ValueError("UI not initialized")
        
        return self.demo.launch(**kwargs)


def main():
    """Main entry point"""
    print("🚀 Starting Enhanced Graphiti PostgreSQL UI...")
    
    app = EnhancedGraphitiUI()
    
    try:
        settings = get_settings()
        app.launch(
            server_name=getattr(settings, 'gradio_server_name', '127.0.0.1'),
            server_port=getattr(settings, 'gradio_port', 7860),
            share=getattr(settings, 'gradio_share', False),
            debug=getattr(settings, 'debug', True),
            show_error=True
        )
    except Exception as e:
        print(f"❌ Error launching UI: {e}")
        app.launch(
            server_name='127.0.0.1',
            server_port=7860,
            share=False,
            debug=True
        )


if __name__ == "__main__":
    main()
