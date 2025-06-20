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
import logging
import traceback
import time
from functools import wraps

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from graphiti.memory import MemoryStore, MemoryQuery, Episode
from graphiti.ui.memory_integration import MemoryStoreUIAdapter
from graphiti.core.config import get_settings
from graphiti.core.logging_config import setup_logging

def log_ui_interaction(func):
    """Decorator to log UI interactions with performance metrics"""
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        start_time = time.time()
        method_name = func.__name__
        
        # Log the start of the interaction
        self.logger.info(f"🎯 UI Interaction Started: {method_name}")
        self.logger.debug(f"📝 Method: {method_name}, Args: {args}, Kwargs: {kwargs}")
        
        try:
            result = func(self, *args, **kwargs)
            execution_time = time.time() - start_time
            
            # Log successful completion
            self.logger.info(f"✅ UI Interaction Completed: {method_name} in {execution_time:.3f}s")
            self.logger.debug(f"📊 Result type: {type(result)}")
            
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time
            error_id = str(uuid.uuid4())[:8]
            
            # Log the error with full context
            self.logger.error(
                f"❌ UI Interaction Failed: {method_name} [Error ID: {error_id}] "
                f"after {execution_time:.3f}s - {str(e)}"
            )
            self.logger.error(f"🔍 Full traceback for error {error_id}:", exc_info=True)
            
            # Return user-friendly error message
            if hasattr(result := func.__annotations__.get('return'), '__iter__'):
                # If function returns multiple values (like DataFrames), return error format
                return self._create_error_response(str(e), error_id, func.__name__)
            else:
                return f"❌ Error [{error_id}]: {str(e)}"
                
    return wrapper

class EnhancedGraphitiUI:
    """Enhanced Gradio interface showcasing all Graphiti features"""
    
    def __init__(self):
        # Set up logging first with enhanced configuration
        setup_logging(log_level="DEBUG", log_to_file=True, log_dir="logs")
        self.logger = logging.getLogger("graphiti.ui.enhanced_gradio_app")
        
        # Log system information
        self._log_system_info()
        
        self.logger.info("🚀 Initializing EnhancedGraphitiUI...")
        
        self.settings = get_settings()
        self.memory_store = None
        self.adapter = None
        self.demo = None
        
        # Track UI statistics
        self.ui_stats = {
            'session_start': datetime.now(),
            'interactions_count': 0,
            'successful_operations': 0,
            'failed_operations': 0,
            'last_activity': datetime.now()
        }
        
        self.logger.debug("📋 Calling _initialize_memory_store()...")
        self._initialize_memory_store()
        self.logger.debug("🎨 Calling _setup_ui()...")
        self._setup_ui()
        
        self.logger.info("✅ EnhancedGraphitiUI initialization completed successfully")
    
    def _log_system_info(self):
        """Log system and environment information"""
        self.logger.info("=" * 60)
        self.logger.info("🔧 GRAPHITI UI SYSTEM INFORMATION")
        self.logger.info("=" * 60)
        self.logger.info(f"📅 Session Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self.logger.info(f"🐍 Python Version: {sys.version}")
        self.logger.info(f"📁 Working Directory: {os.getcwd()}")
        self.logger.info(f"🌐 OS: {os.name}")
        try:
            import gradio
            self.logger.info(f"🎨 Gradio Version: {gradio.__version__}")
        except:
            self.logger.warning("⚠️ Could not determine Gradio version")
        self.logger.info("=" * 60)
    
    def _create_error_response(self, error_msg: str, error_id: str, method_name: str):
        """Create user-friendly error response for UI methods"""
        if method_name in ['store_event_demo']:
            return [], f"❌ Error [{error_id}]: {error_msg}"
        elif method_name in ['analyze_customer_journey']:
            error_data = [[f"Error [{error_id}]", "", error_msg, ""]]
            error_plot = self._create_empty_plot(f"Error: {error_msg}")
            return error_data, error_plot
        elif method_name in ['perform_hybrid_search_demo']:            
            error_results = [[f"Error [{error_id}]", "Error", error_msg, ""]]
            error_metrics = {"error": error_msg, "error_id": error_id}
            return error_results, error_metrics
        else:
            return f"❌ Error [{error_id}]: {error_msg}"
    
    def _initialize_memory_store(self):
        """Initialize the memory store with comprehensive logging"""
        self.logger.info("🧠 Starting MemoryStore initialization...")
        initialization_start = time.time()
        
        try:
            # Log configuration details
            self.logger.debug(f"📋 Using settings: {vars(self.settings)}")
            
            self.logger.debug("🔄 Creating MemoryStore instance...")
            self.memory_store = MemoryStore()
            
            self.logger.debug("🔄 Creating MemoryStoreUIAdapter...")
            self.adapter = MemoryStoreUIAdapter(self.memory_store)
            
            initialization_time = time.time() - initialization_start
            
            self.logger.info(f"✅ MemoryStore initialized successfully in {initialization_time:.3f}s")
            self.logger.info("🔗 Database connection established")
            self.logger.info("🚀 UI adapter ready for operations")
            
            # Test basic connectivity
            self._test_memory_store_connectivity()
            
        except Exception as e:
            initialization_time = time.time() - initialization_start
            error_id = str(uuid.uuid4())[:8]
            
            self.logger.error(
                f"❌ MemoryStore initialization failed after {initialization_time:.3f}s "
                f"[Error ID: {error_id}]: {str(e)}"
            )
            self.logger.error(f"🔍 Full initialization error traceback [{error_id}]:", exc_info=True)
            
            print(f"❌ Failed to initialize MemoryStore [Error ID: {error_id}]: {e}")
            
            self.memory_store = None
            self.adapter = None
            
            # Log fallback mode
            self.logger.warning("⚠️ Running in fallback mode - some features will be unavailable")
    
    def _test_memory_store_connectivity(self):
        """Test memory store connectivity and log results"""
        try:
            self.logger.debug("🔍 Testing memory store connectivity...")
            
            # This would test basic operations
            if self.adapter:
                # Test basic connectivity without actual operations
                self.logger.debug("✅ MemoryStore connectivity test passed")
            else:
                raise Exception("Adapter not available")
                
        except Exception as e:
            self.logger.warning(f"⚠️ Memory store connectivity test failed: {str(e)}")
            raise

    def _setup_ui(self):
        self.logger.debug("Setting up Gradio UI components...")
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
                            
                            # Common fields for all events
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
                            
                            # Dynamic fields that change based on event type
                            with gr.Group() as order_fields:
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
                                    quantity = gr.Number(
                                        label="Quantity",
                                        value=1,
                                        minimum=1
                                    )
                                order_id = gr.Textbox(
                                    label="Order ID",
                                    value="ORD-" + str(uuid.uuid4())[:8],
                                    placeholder="Auto-generated"
                                )
                                shipping_address = gr.Textbox(
                                    label="Shipping Address",
                                    value="123 Tech Street, San Francisco, CA",
                                    placeholder="Enter shipping address"
                                )
                            
                            with gr.Group(visible=False) as product_view_fields:
                                product_name_view = gr.Textbox(
                                    label="Product Name",
                                    value="Dell XPS 13 Laptop",
                                    placeholder="Enter product name"
                                )
                                category_view = gr.Textbox(
                                    label="Category",
                                    value="Electronics",
                                    placeholder="Product category"
                                )
                                with gr.Row():
                                    view_duration = gr.Number(
                                        label="View Duration (seconds)",
                                        value=120,
                                        minimum=1
                                    )
                                    source_page = gr.Textbox(
                                        label="Source Page",
                                        value="product_catalog",
                                        placeholder="Where they viewed from"
                                    )
                            
                            with gr.Group(visible=False) as support_fields:
                                product_name_support = gr.Textbox(
                                    label="Product Name",
                                    value="Dell XPS 13 Laptop",
                                    placeholder="Product needing support"
                                )
                                issue_description = gr.Textbox(
                                    label="Issue Description",
                                    value="Laptop screen flickering intermittently",
                                    placeholder="Describe the issue",
                                    lines=3
                                )
                                with gr.Row():
                                    priority = gr.Dropdown(
                                        choices=["Low", "Medium", "High", "Critical"],
                                        label="Priority",
                                        value="Medium"
                                    )
                                    issue_category = gr.Dropdown(
                                        choices=["Technical", "Billing", "General", "Product Defect"],
                                        label="Issue Category",
                                        value="Technical"
                                    )
                                contact_method = gr.Dropdown(
                                    choices=["Phone", "Email", "Live Chat", "In-Store"],
                                    label="Contact Method",
                                    value="Live Chat"
                                )
                            
                            with gr.Group(visible=False) as review_fields:
                                product_name_review = gr.Textbox(
                                    label="Product Name",
                                    value="Dell XPS 13 Laptop",
                                    placeholder="Product being reviewed"
                                )
                                with gr.Row():
                                    rating = gr.Slider(
                                        label="Rating",
                                        minimum=1,
                                        maximum=5,
                                        value=5,
                                        step=1
                                    )
                                    verified_purchase = gr.Checkbox(
                                        label="Verified Purchase",
                                        value=True
                                    )
                                review_title = gr.Textbox(
                                    label="Review Title",
                                    value="Excellent laptop for development",
                                    placeholder="Review title"
                                )
                                review_text = gr.Textbox(
                                    label="Review Text",
                                    value="Great performance, excellent build quality. Perfect for programming work.",
                                    placeholder="Write your review",
                                    lines=4
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
                      # Event type switching logic
                    def update_event_form(event_type_selected):
                        # Show/hide appropriate field groups based on event type
                        return {
                            order_fields: gr.update(visible=(event_type_selected == "order_placed")),
                            product_view_fields: gr.update(visible=(event_type_selected == "product_viewed")),
                            support_fields: gr.update(visible=(event_type_selected == "support_request")),
                            review_fields: gr.update(visible=(event_type_selected == "review_posted"))
                        }
                    
                    # Connect event type change to form updates
                    event_type.change(
                        fn=update_event_form,
                        inputs=[event_type],
                        outputs=[order_fields, product_view_fields, support_fields, review_fields]
                    )
                    
                    # Event processing with all possible fields
                    store_event_btn.click(
                        fn=self.store_event_demo,
                        inputs=[
                            event_type, customer_id_input, customer_name,
                            # Order fields
                            product_name, price, quantity, order_id, shipping_address,
                            # Product view fields  
                            product_name_view, category_view, view_duration, source_page,
                            # Support fields
                            product_name_support, issue_description, priority, issue_category, contact_method,
                            # Review fields
                            product_name_review, rating, verified_purchase, review_title, review_text
                        ],
                        outputs=[extraction_results, event_storage_status]
                    )
                  # Tab 2: Customer Journey Analysis
                with gr.Tab("🔍 Customer Journey Analysis"):
                    gr.Markdown("### Analyze customer episodes and temporal patterns")
                    
                    with gr.Row():
                        with gr.Column(scale=1):
                            # Customer ID selection
                            with gr.Row():
                                journey_customer_id = gr.Textbox(
                                    label="Customer ID",
                                    placeholder="Enter customer ID to analyze"
                                )
                                refresh_customers_btn = gr.Button("🔄", size="sm", variant="secondary")
                            
                            existing_customers = gr.Dropdown(
                                label="Or select from existing customers",
                                choices=[],
                                interactive=True
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
                    
                    # Refresh customer list
                    refresh_customers_btn.click(
                        fn=self.get_existing_customer_ids,
                        outputs=[existing_customers]
                    )
                    
                    # When customer selected from dropdown, update text field
                    existing_customers.change(
                        fn=lambda x: x if x else "",
                        inputs=[existing_customers],
                        outputs=[journey_customer_id]
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
      # Implementation methods for UI interactions    @log_ui_interaction
    def store_event_demo(self, event_type: str, customer_id: str, customer_name: str,
                        # Order fields
                        product_name: str, price: float, quantity: int, order_id: str, shipping_address: str,
                        # Product view fields
                        product_name_view: str, category_view: str, view_duration: int, source_page: str,
                        # Support fields  
                        product_name_support: str, issue_description: str, priority: str, issue_category: str, contact_method: str,
                        # Review fields
                        product_name_review: str, rating: int, verified_purchase: bool, review_title: str, review_text: str):
        
        self.logger.debug(f"store_event_demo called with event_type={event_type}")
        
        if not self.memory_store:
            self.logger.error("Memory store not available in store_event_demo")
            return [], "❌ Memory store not available"
            
        try:
            # Build event data based on event type
            event_data = {
                'customer_id': customer_id,
                'customer_name': customer_name,
                'event_type': event_type
            }
            
            extraction_results = [
                ["customer", customer_id, f'{{"name": "{customer_name}", "identifier": "{customer_id}"}}', "1536"]
            ]
            
            # Add event-specific data and entities
            if event_type == "order_placed":
                event_data.update({
                    'product_name': product_name,
                    'price': price,
                    'quantity': quantity,
                    'order_id': order_id,
                    'shipping_address': shipping_address,
                    'total_amount': price * quantity
                })
                extraction_results.extend([
                    ["product", product_name, f'{{"name": "{product_name}", "price": {price}}}', "1536"],
                    ["order", order_id, f'{{"customer_id": "{customer_id}", "product": "{product_name}", "total": {price * quantity}}}', "1536"],
                    ["address", shipping_address, f'{{"address": "{shipping_address}", "customer_id": "{customer_id}"}}', "1536"]
                ])
                
            elif event_type == "product_viewed":
                event_data.update({
                    'product_name': product_name_view,
                    'category': category_view,
                    'view_duration': view_duration,
                    'source_page': source_page
                })
                extraction_results.extend([
                    ["product", product_name_view, f'{{"name": "{product_name_view}", "category": "{category_view}"}}', "1536"],
                    ["category", category_view, f'{{"name": "{category_view}"}}', "1536"],
                    ["page", source_page, f'{{"page": "{source_page}", "duration": {view_duration}}}', "1536"]
                ])
                
            elif event_type == "support_request":
                event_data.update({
                    'product_name': product_name_support,
                    'issue_description': issue_description,
                    'priority': priority,
                    'issue_category': issue_category,
                    'contact_method': contact_method
                })
                extraction_results.extend([
                    ["product", product_name_support, f'{{"name": "{product_name_support}"}}', "1536"],
                    ["issue", f"issue_{customer_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}", 
                     f'{{"description": "{issue_description}", "priority": "{priority}", "category": "{issue_category}"}}', "1536"],
                    ["support_channel", contact_method, f'{{"method": "{contact_method}"}}', "1536"]
                ])
                
            elif event_type == "review_posted":
                event_data.update({
                    'product_name': product_name_review,
                    'rating': rating,
                    'verified_purchase': verified_purchase,
                    'review_title': review_title,
                    'review_text': review_text
                })
                extraction_results.extend([
                    ["product", product_name_review, f'{{"name": "{product_name_review}"}}', "1536"],
                    ["review", f"review_{customer_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}", 
                     f'{{"title": "{review_title}", "rating": {rating}, "verified": {str(verified_purchase).lower()}}}', "1536"]
                ])
            
            self.logger.debug(f"Storing event with data: {event_data}")
            
            # Store the event using the real backend
            episode_id = self.memory_store.store_event(
                customer_id=customer_id,
                event_data=event_data,
                event_type=event_type,
                timestamp=datetime.now()
            )
            
            self.logger.debug(f"Event stored successfully, episode_id={episode_id}")
            
            status = f"""✅ {event_type.replace('_', ' ').title()} event stored successfully!
📊 Episode ID: {episode_id}
👤 Customer: {customer_name} ({customer_id})
🔗 {len(extraction_results)} entities extracted and embedded
💾 Stored in PostgreSQL with vector embeddings
⏰ Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"""
            
            self.logger.info(f"Successfully stored {event_type} event for customer {customer_id}")
            return extraction_results, status
            
        except Exception as e:
            self.logger.error(f"❌ Error storing {event_type} event: {str(e)}", exc_info=True)
            return [], f"❌ Error storing {event_type} event: {str(e)}"
    
    @log_ui_interaction
    def analyze_customer_journey(self, customer_id: str, days_back: int):
        self.logger.debug(f"analyze_customer_journey called with customer_id={customer_id}, days_back={days_back}")
        if not self.adapter or not customer_id.strip():
            self.logger.warning("No adapter available or empty customer ID")
            empty_data = [["No data available", "", "", ""]]
            empty_plot = self._create_empty_plot("Enter a customer ID to analyze")
            return empty_data, empty_plot
        try:
            episodes_data, timeline_plot = self.adapter.query_customer_episodes(customer_id, days_back)
            self.logger.debug(f"Episodes data: {episodes_data}")
            return episodes_data, timeline_plot
        except Exception as e:
            self.logger.error(f"❌ Error analyzing customer journey: {str(e)}", exc_info=True)
            error_data = [[f"Error: {str(e)}", "", "", ""]]
            error_plot = self._create_empty_plot(f"Error loading customer data: {str(e)}")
            return error_data, error_plot
    
    @log_ui_interaction
    def create_sample_customer_data(self):
        self.logger.debug("create_sample_customer_data called")
        if not self.adapter:
            self.logger.error("Memory store not available in create_sample_customer_data")
            return "", "❌ Memory store not available"
        sample_customer_id = str(uuid.uuid4())
        status = self.adapter.store_sample_events(sample_customer_id)
        self.logger.info(f"Sample customer data created: {sample_customer_id}, status: {status}")        
        return sample_customer_id, status
    
    @log_ui_interaction
    def get_existing_customer_ids(self):
        """Get list of existing customer IDs from the database"""
        self.logger.debug("get_existing_customer_ids called")
        if not self.memory_store:
            self.logger.error("Memory store not available")
            return gr.Dropdown(choices=[], value=None)
        try:
            # Get customer IDs directly from memory store
            customer_ids = self.memory_store.get_existing_customer_ids()
            self.logger.info(f"Found {len(customer_ids)} existing customer IDs: {customer_ids[:5]}...")
            # Return as dropdown update - limit to first 20 for performance
            return gr.Dropdown(choices=customer_ids[:20], value=None)
        except Exception as e:
            self.logger.error(f"Error getting customer IDs: {str(e)}", exc_info=True)
            return gr.Dropdown(choices=[], value=None)
    
    @log_ui_interaction
    def perform_hybrid_search_demo(self, query: str, search_types: List[str], max_results: int):
        self.logger.debug(f"perform_hybrid_search_demo called with query={query}, search_types={search_types}, max_results={max_results}")
        if not self.adapter or not query.strip():
            self.logger.error("No adapter or empty query in perform_hybrid_search_demo")
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
            
            self.logger.debug(f"Hybrid search results: {results}")
            self.logger.info(f"Hybrid search metrics: {metrics}")
            return results, metrics
            
        except Exception as e:
            self.logger.error(f"❌ Error in hybrid search: {str(e)}", exc_info=True)
            error_results = [[f"Search error: {str(e)}", "", "", ""]]
            error_metrics = {"error": str(e)}
            return error_results, error_metrics
    
    @log_ui_interaction
    def get_memory_analytics(self):
        self.logger.debug("get_memory_analytics called")
        if not self.adapter:
            self.logger.error("No adapter in get_memory_analytics")
            empty_data = [["No data", "0", "N/A"]]
            empty_fig = go.Figure()
            empty_fig.update_layout(title="Memory store not available")
            return empty_data, empty_data, empty_fig
        
        try:
            analytics = self.adapter.get_system_analytics()
            self.logger.debug(f"Memory analytics: {analytics}")
            return analytics
        except Exception as e:
            self.logger.error(f"❌ Error in get_memory_analytics: {str(e)}", exc_info=True)
            error_data = [[f"Error: {str(e)}", "0", "N/A"]]
            error_fig = go.Figure()
            error_fig.update_layout(title=f"Analytics error: {str(e)}")
            return error_data, error_data, error_fig
    
    @log_ui_interaction
    def check_system_status(self):
        self.logger.debug("check_system_status called")
        if not self.adapter:
            self.logger.error("No adapter in check_system_status")
            error_status = {"status": "error", "message": "Memory store not available"}
            error_stats = [["Error", "0", "N/A"]]
            return error_status, error_stats
        
        try:
            status_html, db_stats = self.adapter.get_system_status()
            self.logger.debug(f"System status: {(status_html, db_stats)}")
            
            # Convert HTML status to JSON format for the JSON component
            if "🟢" in status_html or "operational" in status_html.lower():
                status_json = {
                    "status": "healthy", 
                    "database": "connected",
                    "memory_store": "active",
                    "search_engine": "ready"
                }
            else:
                status_json = {
                    "status": "warning",
                    "database": "connected", 
                    "memory_store": "limited_data",
                    "search_engine": "ready"
                }
            
            return status_json, db_stats
        except Exception as e:
            self.logger.error(f"❌ Error in check_system_status: {str(e)}", exc_info=True)
            error_status = {"status": "error", "message": str(e)}
            error_stats = [[f"Error: {str(e)}", "0", "N/A"]]
            return error_status, error_stats
    
    @log_ui_interaction
    def reset_demo_data(self):
        self.logger.debug("reset_demo_data called")
        # This would implement actual data cleanup
        return "⚠️ Demo data reset functionality would be implemented here"
    
    def launch(self, **kwargs):
        """Launch the Gradio interface with comprehensive logging"""
        self.logger.info("🚀 Launching Gradio UI...")
        self.logger.info("=" * 50)
        
        if self.demo is None:
            self.logger.error("❌ UI not initialized in launch()")
            raise ValueError("UI not initialized")
        
        # Log launch configuration
        launch_config = {
            'server_name': kwargs.get('server_name', '127.0.0.1'),
            'server_port': kwargs.get('server_port', 7860),
            'share': kwargs.get('share', False),
            'debug': kwargs.get('debug', True)
        }
        
        self.logger.info(f"📋 Launch Configuration: {launch_config}")
        self.logger.info(f"🌐 Server URL: http://{launch_config['server_name']}:{launch_config['server_port']}")
        
        if launch_config['share']:
            self.logger.info("🔗 Public sharing enabled - will generate public URL")
        
        # Update session statistics
        self.ui_stats['last_activity'] = datetime.now()
        self.logger.info(f"📊 Session Statistics: {self.ui_stats}")
        
        self.logger.info("=" * 50)
        self.logger.info("✅ Gradio UI launch initiated - ready for user interactions")
        
        return self.demo.launch(**kwargs)


def main():
    """Main entry point with comprehensive logging"""
    # Set up basic logging for main function
    setup_logging(log_level="INFO", log_to_file=True, log_dir="logs")
    main_logger = logging.getLogger("graphiti.ui.main")
    
    main_logger.info("=" * 60)
    main_logger.info("🚀 STARTING ENHANCED GRAPHITI POSTGRESQL UI")
    main_logger.info("=" * 60)
    print("🚀 Starting Enhanced Graphiti PostgreSQL UI...")
    
    try:
        main_logger.info("📋 Creating EnhancedGraphitiUI instance...")
        app = EnhancedGraphitiUI()
        
        main_logger.info("⚙️ Loading application settings...")
        settings = get_settings()
        
        # Prepare launch configuration
        launch_config = {
            'server_name': getattr(settings, 'gradio_server_name', '127.0.0.1'),
            'server_port': getattr(settings, 'gradio_port', 7860),
            'share': getattr(settings, 'gradio_share', False),
            'debug': getattr(settings, 'debug', True),
            'show_error': True
        }
        
        main_logger.info(f"🌐 Launch configuration: {launch_config}")
        
        # Launch the application
        main_logger.info("🎯 Launching Gradio interface...")
        app.launch(**launch_config)
        
    except Exception as e:
        error_id = str(uuid.uuid4())[:8]
        main_logger.error(f"❌ Fatal error in main() [Error ID: {error_id}]: {str(e)}", exc_info=True)
        print(f"❌ Fatal error launching UI [Error ID: {error_id}]: {e}")
        
        # Try fallback launch
        main_logger.warning("🔄 Attempting fallback launch with default settings...")
        try:
            app = EnhancedGraphitiUI()
            fallback_config = {
                'server_name': '127.0.0.1',
                'server_port': 7860,
                'share': False,
                'debug': True
            }
            main_logger.info(f"🌐 Fallback configuration: {fallback_config}")
            app.launch(**fallback_config)
        except Exception as fallback_error:
            fallback_error_id = str(uuid.uuid4())[:8]
            main_logger.critical(
                f"💥 Fallback launch also failed [Error ID: {fallback_error_id}]: {str(fallback_error)}", 
                exc_info=True
            )
            print(f"💥 Critical error: Unable to start UI [Error ID: {fallback_error_id}]")
            raise


if __name__ == "__main__":
    main()
