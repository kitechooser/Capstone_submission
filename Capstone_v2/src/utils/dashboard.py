import webbrowser
import os
import http.server
import socketserver
import threading
from pathlib import Path
import json
import time
import logging
import numpy as np
from datetime import datetime
import socket
import shutil

# Configure logging
logging.getLogger('dashboard').setLevel(logging.WARNING)

class DashboardServer:
    def __init__(self, dashboard_dir, start_port=8000):
        self.dashboard_dir = dashboard_dir
        self.start_port = start_port
        self.port = None
        self.server = None
        self.server_thread = None
        self.running = False
        self.max_port_attempts = 10  # Try up to 10 different ports

    def _find_available_port(self):
        """Find an available port starting from start_port"""
        for port in range(self.start_port, self.start_port + self.max_port_attempts):
            try:
                # Test if port is available
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    s.bind(('', port))
                    return port
            except OSError:
                continue
        raise Exception(f"No available ports found between {self.start_port} and {self.start_port + self.max_port_attempts - 1}")

    def start(self):
        """Start a local server to serve the dashboard"""
        if self.running:
            return
            
        os.chdir(self.dashboard_dir)
        
        # Suppress server logs but handle connection errors
        class QuietHandler(http.server.SimpleHTTPRequestHandler):
            def log_message(self, format, *args):
                pass
                
            def handle(self):
                try:
                    super().handle()
                except ConnectionResetError:
                    print("\nConnection reset. This may happen if the computer went to sleep or lost connection.")
                    print("The dashboard will continue running. Just refresh the page.")
                except Exception as e:
                    print(f"\nError handling request: {str(e)}")
                    print("The dashboard will continue running. Try refreshing the page.")

        try:
            # Find an available port
            self.port = self._find_available_port()
            
            # Create and start server
            self.server = socketserver.TCPServer(("", self.port), QuietHandler)
            self.server_thread = threading.Thread(target=self._run_server)
            self.server_thread.daemon = True
            self.server_thread.start()
            self.running = True
            
            print(f"\nDashboard server started at http://localhost:{self.port}")
            print(f"Serving files from: {self.dashboard_dir}")
            print("Note: If the computer goes to sleep, just refresh the page when it wakes up.")
            
        except Exception as e:
            print(f"\nError starting server: {str(e)}")
            self.cleanup()
            raise

    def _run_server(self):
        """Run server in thread with error handling"""
        try:
            self.server.serve_forever()
        except Exception as e:
            if self.running:  # Only show error if we didn't stop intentionally
                print(f"\nServer error: {str(e)}")
                print("Try refreshing the page or restarting the dashboard.")

    def cleanup(self):
        """Clean up server resources"""
        if self.server:
            try:
                self.server.server_close()
            except Exception:
                pass
        self.server = None
        self.running = False
        self.port = None

    def stop(self):
        """Stop the dashboard server"""
        if self.running:
            self.running = False
            if self.server:
                try:
                    self.server.shutdown()
                    self.server.server_close()
                except Exception:
                    pass
            print("\nDashboard server stopped")
        self.cleanup()

    def open_dashboard(self, dashboard_name=None):
        """Open a specific dashboard or list available ones"""
        if not self.running:
            self.start()
            
        if dashboard_name:
            url = f"http://localhost:{self.port}/{dashboard_name}"
            webbrowser.open(url)
        else:
            webbrowser.open(f"http://localhost:{self.port}")

class DashboardManager:
    """Manages dashboard creation, updates, and visualization"""
    
    def __init__(self, config):
        self.config = config
        self.dashboard_dir = self.config.SAVE_DIR
        os.makedirs(self.dashboard_dir, exist_ok=True)
        
        # Copy dashboard templates
        self.template_dir = os.path.join(os.path.dirname(__file__), 'dashboard_templates')
        self._copy_templates()
        
        self.server = DashboardServer(self.dashboard_dir, config.DASHBOARD_PORT)
        self.metrics_data = {}
        self.status = {
            'initialized': False,
            'metrics_file_created': False,
            'server_running': False,
            'last_update': None
        }

    def _copy_templates(self):
        """Copy dashboard templates to save directory"""
        try:
            # Copy summary dashboard template
            summary_template = os.path.join(self.template_dir, 'summary_dashboard.html')
            if os.path.exists(summary_template):
                shutil.copy2(summary_template, os.path.join(self.dashboard_dir, 'summary_dashboard.html'))
            
            # Copy parameter space template
            param_template = os.path.join(self.template_dir, 'parameter_space.html')
            if os.path.exists(param_template):
                shutil.copy2(param_template, os.path.join(self.dashboard_dir, 'parameter_space.html'))
                
        except Exception as e:
            print(f"Error copying templates: {str(e)}")

    def initialize_dashboards(self):
        """Initialize dashboard files"""
        try:
            self._create_index_page()
            self.status['metrics_file_created'] = True
            self.status['initialized'] = True
            
        except Exception as e:
            logging.getLogger('dashboard').error(f"Error initializing dashboards: {str(e)}")
            self.status['initialized'] = False
            raise

    def start_server(self):
        """Start the dashboard server after all visualizations are generated"""
        try:
            # Regenerate index page to include all visualizations
            self._create_index_page()
            
            # Start server
            self.server.start()
            self.status['server_running'] = True
            
            # Open the dashboard in the browser
            self.server.open_dashboard('index.html')
            
        except Exception as e:
            logging.getLogger('dashboard').error(f"Error starting server: {str(e)}")
            raise

    def update_dashboards(self):
        """Update dashboards after new visualizations are generated"""
        try:
            # Regenerate index page to include new visualizations
            self._create_index_page()
            
            # Start server if not running
            if not self.status['server_running']:
                self.start_server()
                
        except Exception as e:
            logging.getLogger('dashboard').error(f"Error updating dashboards: {str(e)}")
            raise

    def _create_index_page(self):
        """Create index page that points to the actual visualization files"""
        # Define visualization types and their descriptions
        visualization_types = {
            'summary_dashboard': {
                'title': 'Training Results Dashboard',
                'description': 'View all model metrics, training history, and performance comparisons',
                'category': 'Overview'
            },
            'parameter_space': {
                'title': 'Parameter Space Analysis',
                'description': 'Explore the relationship between model parameters and performance in 3D space',
                'category': 'Parameter Analysis'
            },
            'correlation_matrix': {
                'title': 'Parameter Correlations',
                'description': 'Analyze correlations between different model parameters',
                'category': 'Parameter Analysis'
            },
            'confusion_matrix_plot': {
                'title': 'Confusion Matrix Plot',
                'description': 'Interactive visualization of model prediction accuracy',
                'category': 'Model Performance'
            },
            'parallel_coords': {
                'title': 'Parallel Coordinates',
                'description': 'Compare multiple parameter dimensions simultaneously',
                'category': 'Parameter Analysis'
            },
            'surface': {
                'title': 'Optimization Surface',
                'description': 'Visualize the parameter optimization landscape',
                'category': 'Optimization'
            },
            'confusion_matrix': {
                'title': 'Confusion Matrix',
                'description': 'View model prediction accuracy and error patterns',
                'category': 'Model Performance'
            },
            'parameters': {
                'title': 'Model Parameters',
                'description': 'View model configuration and hyperparameters',
                'category': 'Configuration'
            },
            'stats': {
                'title': 'Model Statistics',
                'description': 'View detailed model performance statistics',
                'category': 'Model Performance'
            }
        }
        
        # Find all visualization files
        html_files = []
        json_files = []
        for file in os.listdir(self.dashboard_dir):
            if file == 'index.html':
                continue
                
            # Parse filename
            parts = file.replace('.html', '').replace('.json', '').split('_')
            model_name = parts[0].upper()
            viz_type = '_'.join(parts[1:])
            
            if file.endswith('.html'):
                viz_info = visualization_types.get(viz_type, {
                    'title': ' '.join(word.capitalize() for word in viz_type.split('_')),
                    'description': '',
                    'category': 'Other'
                })
                html_files.append((model_name, viz_type, file, viz_info))
            elif file.endswith('.json'):
                json_files.append((model_name, viz_type, file))

        # Sort files by model name and visualization type
        html_files.sort()

        # Group files by model and category
        model_files = {}
        for model, viz_type, file, viz_info in html_files:
            if model not in model_files:
                model_files[model] = {}
            category = viz_info['category']
            if category not in model_files[model]:
                model_files[model][category] = []
            model_files[model][category].append((viz_type, file, viz_info))

        # Create the index page HTML
        index_html = '''
        <!DOCTYPE html>
        <html>
        <head>
            <title>Model Visualizations</title>
            <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
            <style>
                .model-section { 
                    margin-bottom: 30px;
                    background-color: #f8f9fa;
                    padding: 20px;
                    border-radius: 10px;
                }
                .category-section {
                    margin-bottom: 20px;
                    padding: 15px;
                    background-color: white;
                    border-radius: 8px;
                }
                .viz-link {
                    margin: 10px 0;
                    padding: 15px;
                    border: 1px solid #ddd;
                    border-radius: 5px;
                    transition: all 0.3s ease;
                    background-color: white;
                }
                .viz-link:hover {
                    transform: translateX(10px);
                    background-color: #e9ecef;
                    border-color: #0d6efd;
                }
                .timestamp {
                    font-size: 0.8em;
                    color: #666;
                }
                .summary-link {
                    background-color: #e3f2fd;
                    border-color: #90caf9;
                }
                .viz-description {
                    font-size: 0.9em;
                    color: #666;
                    margin-top: 5px;
                }
                .model-header {
                    border-bottom: 2px solid #dee2e6;
                    padding-bottom: 10px;
                    margin-bottom: 20px;
                }
                .category-header {
                    color: #495057;
                    font-size: 1.2em;
                    margin-bottom: 15px;
                    padding-bottom: 5px;
                    border-bottom: 1px solid #dee2e6;
                }
                .viz-type-badge {
                    font-size: 0.8em;
                    padding: 3px 8px;
                    border-radius: 10px;
                    background-color: #e9ecef;
                    color: #666;
                    margin-left: 10px;
                }
                .json-link {
                    font-size: 0.8em;
                    color: #0d6efd;
                    margin-left: 10px;
                    text-decoration: none;
                }
                .json-link:hover {
                    text-decoration: underline;
                }
            </style>
        </head>
        <body>
            <div class="container mt-5">
                <h1 class="mb-4">Model Visualizations</h1>
                <div class="row">
                    <div class="col-12">
                        <div class="list-group mb-4">
                            <a href="summary_dashboard.html" class="list-group-item list-group-item-action viz-link summary-link">
                                <h5>Training Results Dashboard</h5>
                                <div class="viz-description">View all model metrics, training history, and performance comparisons</div>
                            </a>
                        </div>
        '''

        # Add each model's visualizations
        for model, categories in model_files.items():
            index_html += f'''
                    <div class="col-12 model-section">
                        <h2 class="model-header">{model}</h2>
            '''
            
            # Add visualizations by category
            for category, files in sorted(categories.items()):
                index_html += f'''
                        <div class="category-section">
                            <h3 class="category-header">{category}</h3>
                            <div class="list-group">
                '''
                
                for viz_type, file, viz_info in sorted(files):
                    # Get file timestamp
                    file_path = os.path.join(self.dashboard_dir, file)
                    timestamp = datetime.fromtimestamp(os.path.getmtime(file_path))
                    timestamp_str = timestamp.strftime("%Y-%m-%d %H:%M:%S")
                    
                    # Check for corresponding JSON file
                    json_file = next((j for _, t, j in json_files if t == viz_type and j.startswith(model.lower())), None)
                    
                    index_html += f'''
                                <a href="{file}" class="list-group-item list-group-item-action viz-link">
                                    <div class="d-flex justify-content-between align-items-center">
                                        <h5>{viz_info['title']}</h5>
                                        <div>
                                            <span class="viz-type-badge">{viz_type}</span>
                                            {f'<a href="{json_file}" class="json-link">[JSON]</a>' if json_file else ''}
                                        </div>
                                    </div>
                                    <div class="viz-description">{viz_info['description']}</div>
                                    <span class="timestamp">Generated: {timestamp_str}</span>
                                </a>
                    '''
                
                index_html += '''
                            </div>
                        </div>
                '''
            
            index_html += '''
                    </div>
            '''

        index_html += '''
                </div>
            </div>
        </body>
        </html>
        '''

        with open(os.path.join(self.dashboard_dir, 'index.html'), 'w') as f:
            f.write(index_html)
        print(f"Created index page with {len(html_files)} visualization files")

    def update_metrics(self, model_name, metrics):
        """Update metrics with status tracking"""
        if not self.status['initialized']:
            logging.getLogger('dashboard').warning("Dashboard not initialized. Skipping metrics update.")
            return
            
        try:
            # Update metrics
            self._update_metrics_data(model_name, metrics)
            
            # Update status
            self.status['last_update'] = datetime.now().isoformat()
            
            # Update dashboards
            self.update_dashboards()
            
        except Exception as e:
            logging.getLogger('dashboard').error(f"Error updating metrics: {str(e)}")

    def _convert_to_serializable(self, value):
        """Convert numpy/numeric types to standard Python types"""
        if isinstance(value, (np.int_, np.intc, np.intp, np.int8, np.int16, np.int32, 
                            np.int64, np.uint8, np.uint16, np.uint32, np.uint64)):
            return int(value)
        elif isinstance(value, (np.float_, np.float16, np.float32, np.float64)):
            return float(value)
        elif isinstance(value, (np.ndarray,)):
            return value.tolist()
        return value

    def _update_metrics_data(self, model_name, metrics):
        """Internal method to update metrics data with validation and logging"""
        metrics_file = os.path.join(self.dashboard_dir, 'metrics.json')
        
        try:
            # Read current metrics
            if os.path.exists(metrics_file):
                with open(metrics_file, 'r') as f:
                    metrics_data = json.load(f)
            else:
                metrics_data = {}

            # Initialize or update model metrics
            if model_name not in metrics_data:
                metrics_data[model_name] = {
                    'accuracy': [],
                    'precision': [],
                    'recall': [],
                    'f1': [],
                    'timestamps': []
                }

            # Validate and update metrics
            for metric_name, value in metrics.items():
                if metric_name in ['accuracy', 'precision', 'recall', 'f1']:
                    try:
                        float_value = self._convert_to_serializable(value)
                        if 0 <= float_value <= 1:  # Valid metric range
                            metrics_data[model_name][metric_name].append(float_value)
                        else:
                            logging.getLogger('dashboard').warning(
                                f"Invalid metric value for {metric_name}: {float_value}")
                    except (TypeError, ValueError) as e:
                        logging.getLogger('dashboard').error(
                            f"Error converting metric value: {value} for {metric_name}")
                        continue

            # Add timestamp
            metrics_data[model_name]['timestamps'].append(datetime.now().isoformat())

            # Save updated metrics
            with open(metrics_file, 'w') as f:
                json.dump(metrics_data, f, indent=2)
                
            # Print debug info
            print("\nUpdated metrics data:")
            for model, model_metrics in metrics_data.items():
                print(f"\n{model}:")
                for metric_name, values in model_metrics.items():
                    if metric_name != 'timestamps':
                        print(f"  {metric_name}: {values}")
                
        except Exception as e:
            logging.getLogger('dashboard').error(f"Error updating metrics data: {str(e)}")
            raise
    
    def stop(self):
        """Stop the dashboard server"""
        self.server.stop()
    
    def wait_for_user(self):
        """Wait for user interaction"""
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            raise
