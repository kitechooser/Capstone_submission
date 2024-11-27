# src/utils/analysis.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import json
import os

class OptimizationAnalyzer:
    """Advanced analysis tools for optimization results"""
    
    def __init__(self, save_dir):
        self.save_dir = save_dir
        self.results_dir = os.path.join(save_dir, 'optimization_analysis')
        os.makedirs(self.results_dir, exist_ok=True)
        
    def parameter_sensitivity_analysis(self, trials_data, model_name):
        """Analyze parameter sensitivity using correlation and ANOVA"""
        # Flatten the trials data
        flattened_data = []
        for trial in trials_data:
            flat_trial = {}
            # Extract parameters
            if isinstance(trial['parameters'], dict):
                for param_name, param_value in trial['parameters'].items():
                    if isinstance(param_value, (int, float, str)):
                        flat_trial[param_name] = param_value
            # Add metric
            if isinstance(trial.get('metric'), (int, float)):
                flat_trial['metric'] = trial['metric']
            flattened_data.append(flat_trial)
        
        # Create DataFrame from flattened data
        df = pd.DataFrame(flattened_data)
        
        if df.empty or len(df.columns) < 2:
            print("Warning: Insufficient data for sensitivity analysis")
            return {}
            
        # Standardize numerical parameters
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        if not numerical_cols.empty:
            scaler = StandardScaler()
            df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
        
        # Correlation analysis
        plt.figure(figsize=(12, 8))
        correlation_matrix = df.corr()
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
        plt.title(f'Parameter Correlation Matrix - {model_name}')
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, f'{model_name}_correlation_matrix.png'))
        plt.close()
        
        # ANOVA for categorical parameters
        categorical_cols = df.select_dtypes(include=['object']).columns
        anova_results = {}
        for col in categorical_cols:
            try:
                groups = [group for _, group in df.groupby(col)['metric']]
                if len(groups) > 1:  # Need at least 2 groups for ANOVA
                    f_stat, p_val = stats.f_oneway(*groups)
                    anova_results[col] = {'f_statistic': f_stat, 'p_value': p_val}
            except Exception as e:
                print(f"Warning: Could not perform ANOVA for {col}: {str(e)}")
            
        return anova_results
    
    def convergence_analysis(self, optimization_history, model_name):
        """Analyze optimization convergence patterns"""
        metrics = np.array(optimization_history['metrics'])
        
        if len(metrics) == 0:
            print("Warning: No metrics available for convergence analysis")
            return {
                'best_value': None,
                'worst_value': None,
                'mean': None,
                'std': None,
                'improvement_rate': None
            }
        
        # Calculate convergence statistics
        convergence_stats = {
            'best_value': float(np.max(metrics)),
            'worst_value': float(np.min(metrics)),
            'mean': float(np.mean(metrics)),
            'std': float(np.std(metrics)),
            'improvement_rate': float((metrics[-1] - metrics[0]) / len(metrics))
        }
        
        # Plot convergence
        plt.figure(figsize=(12, 6))
        plt.plot(metrics, 'b-', label='Metric Value')
        plt.plot(np.maximum.accumulate(metrics), 'r--', label='Best So Far')
        
        # Add moving average
        if len(metrics) >= 5:
            window = 5
            moving_avg = np.convolve(metrics, np.ones(window)/window, mode='valid')
            plt.plot(range(window-1, len(metrics)), moving_avg, 'g:', 
                    label=f'{window}-trial Moving Average')
        
        plt.xlabel('Trial Number')
        plt.ylabel('Metric Value')
        plt.title(f'Optimization Convergence Analysis - {model_name}')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(self.results_dir, f'{model_name}_convergence_analysis.png'))
        plt.close()
        
        return convergence_stats
    
    def parameter_importance(self, trials_data, model_name):
        """Calculate and visualize parameter importance"""
        # Flatten the trials data
        flattened_data = []
        for trial in trials_data:
            flat_trial = {}
            # Extract parameters
            if isinstance(trial['parameters'], dict):
                for param_name, param_value in trial['parameters'].items():
                    if isinstance(param_value, (int, float, str)):
                        flat_trial[param_name] = param_value
            # Add metric
            if isinstance(trial.get('metric'), (int, float)):
                flat_trial['metric'] = trial['metric']
            flattened_data.append(flat_trial)
        
        df = pd.DataFrame(flattened_data)
        
        if df.empty or len(df.columns) < 2:
            print("Warning: Insufficient data for importance analysis")
            return {}
            
        importance_scores = {}
        
        # For numerical parameters
        numerical_cols = df.select_dtypes(include=[np.number]).columns.drop('metric', errors='ignore')
        for col in numerical_cols:
            correlation = abs(df[col].corr(df['metric']))
            importance_scores[col] = float(correlation)
        
        # For categorical parameters
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            # Use mutual information or similar metric
            unique_vals = df[col].nunique()
            variance = df.groupby(col)['metric'].std().mean()
            if pd.notnull(variance):
                importance_scores[col] = float(variance * np.log(unique_vals))
            else:
                importance_scores[col] = 0.0
            
        # Visualize importance scores
        if importance_scores:
            plt.figure(figsize=(10, 6))
            importance_df = pd.DataFrame(importance_scores.items(), 
                                       columns=['Parameter', 'Importance'])
            importance_df = importance_df.sort_values('Importance', ascending=True)
            
            plt.barh(importance_df['Parameter'], importance_df['Importance'])
            plt.title(f'Parameter Importance Analysis - {model_name}')
            plt.xlabel('Importance Score')
            plt.tight_layout()
            plt.savefig(os.path.join(self.results_dir, f'{model_name}_parameter_importance.png'))
            plt.close()
        
        return importance_scores

class RealTimeMonitor:
    """Real-time monitoring dashboard for optimization progress"""
    
    def __init__(self, save_dir):
        self.save_dir = save_dir
        self.dashboard_dir = os.path.join(save_dir, 'dashboard')
        os.makedirs(self.dashboard_dir, exist_ok=True)
        self.history = []
        
    def update(self, trial_info):
        """Update dashboard with new trial information"""
        # Create a copy of trial_info with only serializable data
        serializable_info = {
            'timestamp': datetime.now().isoformat(),
            'metric': float(trial_info['metric']) if isinstance(trial_info.get('metric'), (int, float)) else None,
            'model': str(trial_info.get('model', '')),
            'strategy': str(trial_info.get('strategy', ''))
        }
        
        # Add flattened parameters if they exist
        if 'parameters' in trial_info and isinstance(trial_info['parameters'], dict):
            for param_name, param_value in trial_info['parameters'].items():
                if isinstance(param_value, (int, float, str)):
                    serializable_info[f'param_{param_name}'] = param_value
        
        self.history.append(serializable_info)
        self._update_dashboard()
        
    def _update_dashboard(self):
        """Update dashboard visualizations"""
        if not self.history:
            return
            
        df = pd.DataFrame(self.history)
        
        # Create main dashboard figure
        fig = go.Figure()
        
        # Add metric progress
        fig.add_trace(go.Scatter(
            x=list(range(len(df))),
            y=df['metric'],
            mode='lines+markers',
            name='Current Metric'
        ))
        
        # Add best so far
        fig.add_trace(go.Scatter(
            x=list(range(len(df))),
            y=np.maximum.accumulate(df['metric']),
            mode='lines',
            name='Best So Far',
            line=dict(dash='dash')
        ))
        
        # Update layout
        fig.update_layout(
            title='Optimization Progress',
            xaxis_title='Trial Number',
            yaxis_title='Metric Value',
            hovermode='x unified'
        )
        
        # Save dashboard
        fig.write_html(os.path.join(self.dashboard_dir, 'dashboard.html'))
        
        # Save summary statistics
        stats = {
            'total_trials': len(df),
            'best_metric': float(df['metric'].max()),
            'worst_metric': float(df['metric'].min()),
            'average_metric': float(df['metric'].mean()),
            'last_update': datetime.now().isoformat()
        }
        
        with open(os.path.join(self.dashboard_dir, 'stats.json'), 'w') as f:
            json.dump(stats, f, indent=4)
