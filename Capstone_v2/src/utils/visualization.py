import os
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from scipy.interpolate import griddata

class AdvancedVisualizer:
    """Advanced visualization tools for optimization results"""
    
    def __init__(self, save_dir):
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)
        print(f"Visualization directory initialized at: {self.save_dir}")

    def _add_timestamp(self, fig):
        """Add timestamp to the plot"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        fig.add_annotation(
            text=f"Generated: {timestamp}",
            xref="paper", yref="paper",
            x=1, y=-0.15,
            showarrow=False,
            font=dict(size=10, color="gray"),
            align="right"
        )

    def create_correlation_matrix(self, trials_data, model_name):
        """Create correlation matrix visualization"""
        if not trials_data:
            print("No trials data available for correlation matrix")
            return
            
        # Prepare data
        df = pd.DataFrame([{
            **{k: (f"{v[0]}x{v[1]}" if isinstance(v, tuple) else 
                  float(v) if isinstance(v, (int, float, np.number)) else v)
               for k, v in trial['parameters'].items()},
            'metric': float(trial['metric'])
        } for trial in trials_data])
        
        # Select numeric columns only
        numeric_df = df.select_dtypes(include=[np.number])
        
        if numeric_df.empty:
            print("No numeric data available for correlation matrix")
            return
            
        # Calculate correlation matrix
        corr_matrix = numeric_df.corr()
        
        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            colorscale='RdBu',
            zmin=-1,
            zmax=1,
            text=np.round(corr_matrix.values, 3),
            texttemplate='%{text}',
            textfont={"size": 10},
            hoverongaps=False
        ))
        
        fig.update_layout(
            title=f"{model_name} Parameter Correlations",
            width=800,
            height=800,
            showlegend=False,
            plot_bgcolor='white'
        )
        
        # Add timestamp
        self._add_timestamp(fig)
        
        # Save plot
        save_path = os.path.join(self.save_dir, f'{model_name.lower()}_correlation_matrix.html')
        fig.write_html(save_path)
        print(f"Saved correlation matrix to: {save_path}")

    def create_parameter_space_visualization(self, trials_data, model_name):
        """Create interactive 3D visualization of parameter space"""
        if not trials_data:
            print("No trials data available for visualization")
            return
            
        # Convert trials data to DataFrame with special handling for tuples
        df = pd.DataFrame([{
            **{k: (f"{v[0]}x{v[1]}" if isinstance(v, tuple) else 
                  float(v) if isinstance(v, (int, float, np.number)) else v)
               for k, v in trial['parameters'].items()},
            'metric': float(trial['metric'])
        } for trial in trials_data])
        
        # Select numeric columns only
        numeric_df = df.select_dtypes(include=[np.number])
        if len(numeric_df.columns) < 3:
            print("Insufficient numeric parameters for 3D visualization")
            return
            
        # Select first two parameters and metric
        param1, param2 = numeric_df.columns[:-1][:2]
        
        # Create hover text with proper handling of all parameter types
        hover_text = []
        for i, params in enumerate(df.to_dict('records')):
            param_text = []
            for k, v in params.items():
                if isinstance(v, (int, float, np.number)):
                    param_text.append(f"{k}: {v:.4f}")
                else:
                    param_text.append(f"{k}: {v}")
            hover_text.append(f"Trial {i+1}<br>" + "<br>".join(param_text))
        
        # Create 3D scatter plot
        fig = go.Figure(data=[go.Scatter3d(
            x=numeric_df[param1],
            y=numeric_df[param2],
            z=numeric_df['metric'],
            mode='markers',
            marker=dict(
                size=8,
                color=numeric_df['metric'],
                colorscale='Viridis',
                opacity=0.8,
                colorbar=dict(title="Metric Value")
            ),
            text=hover_text,
            hoverinfo='text'
        )])
        
        fig.update_layout(
            title=f"{model_name} Parameter Space Exploration",
            scene=dict(
                xaxis_title=param1,
                yaxis_title=param2,
                zaxis_title="Metric Value"
            ),
            width=800,
            height=800
        )
        
        # Add timestamp
        self._add_timestamp(fig)
        
        # Save plot
        save_path = os.path.join(self.save_dir, f'{model_name.lower()}_parameter_space.html')
        fig.write_html(save_path)
        print(f"Saved parameter space visualization to: {save_path}")

    def create_parallel_coordinates(self, trials_data, model_name):
        """Create parallel coordinates plot for parameter relationships"""
        if not trials_data:
            print("No trials data available for visualization")
            return
            
        # Prepare data
        df = pd.DataFrame([{
            **{k: (f"{v[0]}x{v[1]}" if isinstance(v, tuple) else 
                  float(v) if isinstance(v, (int, float, np.number)) else v)
               for k, v in trial['parameters'].items()},
            'metric': float(trial['metric'])
        } for trial in trials_data])
        
        # Select numeric columns only
        numeric_df = df.select_dtypes(include=[np.number])
        
        if numeric_df.empty:
            print("No numeric data available for parallel coordinates")
            return
        
        # Create parallel coordinates plot
        fig = px.parallel_coordinates(
            numeric_df,
            color='metric',
            color_continuous_scale='Viridis',
            title=f"{model_name} Parameter Relationships"
        )
        
        fig.update_layout(
            width=1000,
            height=600,
            plot_bgcolor='white',
            paper_bgcolor='white'
        )
        
        # Add timestamp
        self._add_timestamp(fig)
        
        # Save plot
        save_path = os.path.join(self.save_dir, f'{model_name.lower()}_parallel_coords.html')
        fig.write_html(save_path)
        print(f"Saved parallel coordinates plot to: {save_path}")

    def create_optimization_surface(self, trials_data, model_name):
        """Create optimization surface plot"""
        if not trials_data or len(trials_data) < 3:
            print("Insufficient data for surface plot")
            return
            
        # Prepare data
        df = pd.DataFrame([{
            **{k: (f"{v[0]}x{v[1]}" if isinstance(v, tuple) else 
                  float(v) if isinstance(v, (int, float, np.number)) else v)
               for k, v in trial['parameters'].items()},
            'metric': float(trial['metric'])
        } for trial in trials_data])
        
        # Select numeric columns only
        numeric_df = df.select_dtypes(include=[np.number])
        if len(numeric_df.columns) < 3:  # Need at least 2 parameters + metric
            print("Insufficient numeric parameters for surface plot")
            return
            
        param1, param2 = numeric_df.columns[:-1][:2]  # Take first two parameters
        
        # Create grid for surface plot
        x_range = np.linspace(numeric_df[param1].min(), numeric_df[param1].max(), 20)
        y_range = np.linspace(numeric_df[param2].min(), numeric_df[param2].max(), 20)
        x_mesh, y_mesh = np.meshgrid(x_range, y_range)
        
        # Interpolate metric values
        z_mesh = griddata(
            points=(numeric_df[param1], numeric_df[param2]),
            values=numeric_df['metric'],
            xi=(x_mesh, y_mesh),
            method='cubic',
            fill_value=np.nan
        )
        
        # Create surface plot
        fig = go.Figure(data=[
            # Surface plot
            go.Surface(
                x=x_range,
                y=y_range,
                z=z_mesh,
                colorscale='Viridis',
                name='Interpolated Surface'
            ),
            # Scatter plot of actual points
            go.Scatter3d(
                x=numeric_df[param1],
                y=numeric_df[param2],
                z=numeric_df['metric'],
                mode='markers',
                marker=dict(
                    size=5,
                    color='red',
                    symbol='circle'
                ),
                name='Actual Points'
            )
        ])
        
        fig.update_layout(
            title=f"{model_name} Optimization Surface",
            scene=dict(
                xaxis_title=param1,
                yaxis_title=param2,
                zaxis_title="Metric Value",
                camera=dict(
                    up=dict(x=0, y=0, z=1),
                    center=dict(x=0, y=0, z=0),
                    eye=dict(x=1.5, y=1.5, z=1.5)
                )
            ),
            width=800,
            height=800,
            showlegend=True
        )
        
        # Add timestamp
        self._add_timestamp(fig)
        
        # Save plot
        save_path = os.path.join(self.save_dir, f'{model_name.lower()}_surface.html')
        fig.write_html(save_path)
        print(f"Saved optimization surface plot to: {save_path}")

    def create_confusion_matrix_plot(self, confusion_matrix, model_name):
        """Create interactive confusion matrix visualization"""
        if confusion_matrix is None:
            print("No confusion matrix data available")
            return
            
        # Create confusion matrix plot
        fig = go.Figure(data=go.Heatmap(
            z=confusion_matrix,
            x=['Predicted Fake', 'Predicted Real'],
            y=['Actual Fake', 'Actual Real'],
            text=confusion_matrix,
            texttemplate="%{text}",
            textfont={"size": 16},
            colorscale='Blues'
        ))
        
        fig.update_layout(
            title=f"{model_name} Confusion Matrix",
            xaxis_title="Predicted Class",
            yaxis_title="Actual Class",
            width=600,
            height=600
        )
        
        # Add timestamp
        self._add_timestamp(fig)
        
        # Save plot
        save_path = os.path.join(self.save_dir, f'{model_name.lower()}_confusion_matrix_plot.html')
        fig.write_html(save_path)
        print(f"Saved confusion matrix plot to: {save_path}")

        # Also save the data as JSON for reference
        import json
        json_path = os.path.join(self.save_dir, f'{model_name.lower()}_confusion_matrix.json')
        with open(json_path, 'w') as f:
            json.dump({
                'matrix': confusion_matrix.tolist(),
                'labels': ['Fake', 'Real'],
                'timestamp': datetime.now().isoformat()
            }, f, indent=2)

    def create_summary_dashboard(self, model_name, optimization_results, confusion_matrix=None):
        """Create comprehensive summary dashboard"""
        if not optimization_results["trials"]:
            print("No optimization results available for dashboard")
            return
            
        # Convert all numeric values to float
        trials_data = [{
            'parameters': {k: (f"{v[0]}x{v[1]}" if isinstance(v, tuple) else 
                             float(v) if isinstance(v, (int, float, np.number)) else v)
                         for k, v in trial['parameters'].items()},
            'metric': float(trial['metric'])
        } for trial in optimization_results["trials"]]
        
        # Create subplot figure with confusion matrix
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=(
                "Optimization Progress",
                "Parameter Distributions",
                "Parameter Correlations",
                "Best vs Worst Trials",
                "Confusion Matrix",
                "Parameter Importance"
            ),
            specs=[
                [{"type": "scatter"}, {"type": "box"}],
                [{"type": "heatmap"}, {"type": "bar"}],
                [{"type": "heatmap"}, {"type": "bar"}]
            ],
            vertical_spacing=0.12
        )
        
        # Optimization progress
        metrics = [trial['metric'] for trial in trials_data]
        fig.add_trace(
            go.Scatter(
                y=metrics,
                mode='lines+markers',
                name='Metric Value',
                hovertemplate='Trial %{x}<br>Metric: %{y:.4f}'
            ),
            row=1, col=1
        )
        
        # Parameter distributions
        params_df = pd.DataFrame([trial['parameters'] for trial in trials_data])
        numeric_params = params_df.select_dtypes(include=[np.number])
        for param in numeric_params.columns:
            fig.add_trace(
                go.Box(
                    y=numeric_params[param],
                    name=param,
                    hovertemplate='%{y}'
                ),
                row=1, col=2
            )
        
        # Parameter correlations
        if not numeric_params.empty:
            corr_matrix = numeric_params.corr()
            fig.add_trace(
                go.Heatmap(
                    z=corr_matrix.values,
                    x=corr_matrix.columns,
                    y=corr_matrix.columns,
                    colorscale='RdBu',
                    zmin=-1,
                    zmax=1
                ),
                row=2, col=1
            )
        
        # Best vs Worst trials comparison
        best_idx = np.argmax(metrics)
        worst_idx = np.argmin(metrics)
        
        fig.add_trace(
            go.Bar(
                x=['Best Trial', 'Worst Trial'],
                y=[metrics[best_idx], metrics[worst_idx]],
                text=[f'{metrics[best_idx]:.4f}', f'{metrics[worst_idx]:.4f}'],
                textposition='auto',
                hovertemplate='%{x}<br>Metric: %{y:.4f}'
            ),
            row=2, col=2
        )
        
        # Confusion Matrix
        if confusion_matrix is not None:
            fig.add_trace(
                go.Heatmap(
                    z=confusion_matrix,
                    x=['Predicted Fake', 'Predicted Real'],
                    y=['Actual Fake', 'Actual Real'],
                    text=confusion_matrix,
                    texttemplate="%{text}",
                    textfont={"size": 16},
                    colorscale='Blues'
                ),
                row=3, col=1
            )
        
        # Parameter Importance
        importance_scores = self._calculate_parameter_importance(trials_data)
        if importance_scores:
            params, scores = zip(*sorted(importance_scores.items(), key=lambda x: x[1], reverse=True))
            fig.add_trace(
                go.Bar(
                    x=params,
                    y=scores,
                    text=[f'{score:.4f}' for score in scores],
                    textposition='auto',
                    hovertemplate='%{x}<br>Importance: %{y:.4f}'
                ),
                row=3, col=2
            )
        
        # Update layout
        fig.update_layout(
            height=1500,
            title=f"{model_name} Optimization Summary",
            showlegend=True
        )
        
        # Update axes labels
        fig.update_xaxes(title_text="Trial Number", row=1, col=1)
        fig.update_yaxes(title_text="Metric Value", row=1, col=1)
        
        fig.update_xaxes(title_text="Parameter", row=1, col=2)
        fig.update_yaxes(title_text="Value", row=1, col=2)
        
        fig.update_xaxes(title_text="Parameter", row=2, col=1)
        fig.update_yaxes(title_text="Parameter", row=2, col=1)
        
        fig.update_xaxes(title_text="Trial", row=2, col=2)
        fig.update_yaxes(title_text="Metric Value", row=2, col=2)
        
        fig.update_xaxes(title_text="Predicted Class", row=3, col=1)
        fig.update_yaxes(title_text="Actual Class", row=3, col=1)
        
        fig.update_xaxes(title_text="Parameter", row=3, col=2)
        fig.update_yaxes(title_text="Importance Score", row=3, col=2)
        
        # Add timestamp
        self._add_timestamp(fig)
        
        # Save dashboard
        save_path = os.path.join(self.save_dir, f'{model_name.lower()}_summary_dashboard.html')
        fig.write_html(save_path)
        print(f"Saved summary dashboard to: {save_path}")
        
        # Save additional statistics
        stats = {
            'best_metric': float(max(metrics)),
            'worst_metric': float(min(metrics)),
            'mean_metric': float(np.mean(metrics)),
            'std_metric': float(np.std(metrics)),
            'total_trials': len(metrics),
            'best_parameters': trials_data[best_idx]['parameters'],
            'generated_at': datetime.now().isoformat()
        }
        
        stats_path = os.path.join(self.save_dir, f'{model_name.lower()}_stats.json')
        with open(stats_path, 'w') as f:
            import json
            json.dump(stats, f, indent=4)
        print(f"Saved optimization statistics to: {stats_path}")

    def _calculate_parameter_importance(self, trials_data):
        """Calculate parameter importance based on correlation with metric"""
        if not trials_data:
            return {}
            
        # Convert trials to DataFrame
        df = pd.DataFrame([{
            **{k: v for k, v in trial['parameters'].items()},
            'metric': trial['metric']
        } for trial in trials_data])
        
        # Select numeric columns only
        numeric_df = df.select_dtypes(include=[np.number])
        if 'metric' not in numeric_df.columns:
            return {}
            
        # Calculate correlation with metric
        correlations = numeric_df.corr()['metric'].abs()
        correlations = correlations.drop('metric')
        
        return correlations.to_dict()
