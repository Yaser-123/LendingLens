# ===============================================================================
# 🔍 EXPLAINABLE AI DASHBOARD - STREAMLIT APPLICATION
# ===============================================================================
# Interactive dashboard for ML pipeline results with SHAP, LIME, GenAI explanations, and AI Chatbot
# Features: Model performance, explainability analysis, feature overlap, AI interpretation, Interactive Chat
# ===============================================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io
import base64
from datetime import datetime
import warnings
import json
import os
import google.generativeai as genai
from typing import Dict, List, Any
import re
import json
from dotenv import load_dotenv
warnings.filterwarnings('ignore')

# Load environment variables
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '..', '.env'))

# =====================
# 🎨 PAGE CONFIGURATION
# =====================
st.set_page_config(
    page_title=os.getenv('DASHBOARD_TITLE', 'Explainable AI Dashboard'),
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize Gemini API
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
    # Use gemini-2.5-flash (available model from the list)
    gemini_model = genai.GenerativeModel('gemini-2.5-flash')
else:
    gemini_model = None
    st.error("🔴 Gemini API key not found. Please check your .env file.")

# Initialize session state for chat
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'data_context' not in st.session_state:
    st.session_state.data_context = {}

# Force dark theme
st.markdown("""
<script>
const doc = window.parent.document;
buttons = Array.from(doc.querySelectorAll('button[kind=primary]'));
const dark_theme_button = buttons.find(el => el.innerText === '☾');
if (dark_theme_button) {
    dark_theme_button.click();
}
</script>
""", unsafe_allow_html=True)

# Custom CSS for dark theme
st.markdown("""
<style>
    /* Force dark theme with black background and white text */
    .stApp {
        background-color: #000000 !important;
        color: #ffffff !important;
    }
    
    .main-header {
        font-size: 3rem;
        color: #00ff88 !important;
        text-align: center;
        margin-bottom: 2rem;
        background: linear-gradient(90deg, #00ff88, #00aaff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: bold;
    }
    
    .metric-card {
        background-color: #1a1a1a !important;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #00ff88;
        margin: 0.5rem 0;
        color: #ffffff !important;
    }
    
    .insight-box {
        background-color: #0a2a0a !important;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #00ff88;
        margin: 1rem 0;
        color: #ffffff !important;
        font-weight: 500;
    }
    
    .warning-box {
        background-color: #2a2a0a !important;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #ffaa00;
        margin: 1rem 0;
        color: #ffffff !important;
    }
    
    /* Force white text across all elements */
    .stMarkdown, .stText, .stDataFrame, .stSelectbox, .stFileUploader {
        color: #ffffff !important;
    }
    
    /* Sidebar dark styling */
    .css-1d391kg, .stSidebar {
        background-color: #111111 !important;
        color: #ffffff !important;
    }
    
    .stSidebar .stMarkdown {
        color: #ffffff !important;
    }
    
    /* Tab dark styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: #000000 !important;
    }
    
    .stTabs [data-baseweb="tab"] {
        color: #ffffff !important;
        background-color: #1a1a1a !important;
        border-radius: 8px;
        padding: 8px 16px;
        font-weight: 600;
    }
    
    .stTabs [data-baseweb="tab-highlight"] {
        background-color: #00ff88 !important;
    }
    
    /* Metrics dark styling */
    .metric-value {
        color: #00ff88 !important;
        font-size: 1.5rem !important;
        font-weight: bold !important;
    }
    
    /* Dataframe dark styling */
    .dataframe {
        border: 1px solid #333333;
        border-radius: 8px;
        background-color: #1a1a1a !important;
        color: #ffffff !important;
    }
    
    /* File uploader dark styling */
    .stFileUploader {
        background-color: #1a1a1a !important;
        color: #ffffff !important;
    }
    
    /* Alert boxes dark styling */
    .stAlert {
        background-color: #1a1a1a !important;
        color: #ffffff !important;
    }
    
    /* Success/Info/Warning/Error messages dark styling */
    .stSuccess, .stInfo, .stWarning, .stError {
        color: #ffffff !important;
        background-color: #1a1a1a !important;
    }
    
    /* Headers and subheaders white text */
    h1, h2, h3, h4, h5, h6 {
        color: #ffffff !important;
    }
    
    /* Input fields dark styling */
    .stSelectbox > div > div {
        background-color: #1a1a1a !important;
        color: #ffffff !important;
    }
    
    /* Button styling */
    .stButton > button {
        background-color: #00ff88 !important;
        color: #000000 !important;
        border: none !important;
    }
</style>
""", unsafe_allow_html=True)

# Main header
st.markdown('<h1 class="main-header">🔍 Explainable AI Dashboard</h1>', unsafe_allow_html=True)
st.markdown("### 🚀 Interactive ML Pipeline Results with SHAP, LIME & AI-Powered Insights")

# =====================
# 📊 UTILITY FUNCTIONS
# =====================
@st.cache_data
def load_csv_file(uploaded_file):
    """Load CSV file with caching and proper index handling"""
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        # If first column looks like an index, set it as index
        if df.columns[0] in ['Unnamed: 0', 'index', ''] or df.iloc[:, 0].dtype == 'object':
            df = df.set_index(df.columns[0])
        return df
    return None

@st.cache_data
def load_default_data():
    """Load default data from pipeline results"""
    try:
        # Use environment variable for base path
        base_path = os.getenv('RESULTS_PATH', 'c:/Users/T MOHAMED AMMAR/Desktop/Research/Coding/results')
        pipeline_results_path = f"{base_path}/pipeline_results/"
        
        perf_df = pd.read_csv(f"{pipeline_results_path}model_performance.csv", index_col=0)
        shap_df = pd.read_csv(f"{pipeline_results_path}shap_feature_importance.csv", index_col=0)
        lime_df = pd.read_csv(f"{pipeline_results_path}lime_feature_importance.csv", index_col=0)
        overlap_df = pd.read_csv(f"{pipeline_results_path}feature_overlap_analysis.csv", index_col=0)
        
        return perf_df, shap_df, lime_df, overlap_df
        
    except Exception as e:
        st.error(f"Error loading default data: {e}")
        return None, None, None, None

@st.cache_data
def load_genai_explanations():
    """Load GenAI explanations from saved files"""
    try:
        base_path = os.getenv('RESULTS_PATH', 'c:/Users/T MOHAMED AMMAR/Desktop/Research/Coding/results')
        genai_path = f"{base_path}/genai_explanations/"
        
        explanations = {}
        
        # Load text explanations
        for model in ['random_forest', 'xgboost', 'logistic_regression']:
            try:
                with open(f"{genai_path}{model}_explanation.txt", 'r', encoding='utf-8') as f:
                    explanations[model] = f.read()
            except Exception as e:
                st.warning(f"Could not load {model} explanation: {e}")
        
        # Load model comparison
        try:
            with open(f"{genai_path}model_comparison_analysis.txt", 'r', encoding='utf-8') as f:
                explanations['model_comparison'] = f.read()
        except Exception as e:
            st.warning(f"Could not load model comparison: {e}")
        
        # Load JSON data
        try:
            with open(f"{genai_path}genai_explanations.json", 'r', encoding='utf-8') as f:
                explanations['json_data'] = json.load(f)
        except Exception as e:
            st.warning(f"Could not load JSON explanations: {e}")
        
        return explanations
        
    except Exception as e:
        st.error(f"Error loading GenAI explanations: {e}")
        return {}

def prepare_data_context(df_perf, df_shap, df_lime, df_overlap, genai_explanations):
    """Prepare comprehensive data context for chatbot"""
    context = {
        "performance_data": df_perf.to_dict() if df_perf is not None else {},
        "shap_data": df_shap.to_dict() if df_shap is not None else {},
        "lime_data": df_lime.to_dict() if df_lime is not None else {},
        "overlap_data": df_overlap.to_dict() if df_overlap is not None else {},
        "genai_explanations": genai_explanations,
        "data_summary": {
            "models_count": len(df_perf) if df_perf is not None else 0,
            "features_count": len(df_shap) if df_shap is not None else 0,
            "best_model": df_perf['f1'].idxmax() if df_perf is not None and 'f1' in df_perf.columns else "Unknown",
            "best_f1": df_perf['f1'].max() if df_perf is not None and 'f1' in df_perf.columns else 0
        }
    }
    return context

def ai_analyze_plot_request(user_request: str, context: Dict[str, Any]) -> Dict[str, Any]:
    """Use AI to analyze the plot request and determine what to plot"""
    if not gemini_model:
        return {"error": "AI model not available"}
    
    try:
        # Prepare context for AI analysis
        available_data = {
            "models": list(context.get("performance_data", {}).keys()) if context.get("performance_data") else [],
            "metrics": list(context.get("performance_data", {}).get(list(context.get("performance_data", {}).keys())[0], {}).keys()) if context.get("performance_data") else [],
            "shap_features": list(context.get("shap_data", {}).keys()) if context.get("shap_data") else [],
            "lime_features": list(context.get("lime_data", {}).keys()) if context.get("lime_data") else []
        }
        
        analysis_prompt = f"""
        You are a data visualization expert. Analyze this plot request and provide a JSON response with the exact specifications.
        
        USER REQUEST: "{user_request}"
        
        AVAILABLE DATA:
        - Models: {available_data['models']}
        - Metrics: {available_data['metrics']}
        - SHAP data available for: {available_data['shap_features']}
        - LIME data available for: {available_data['lime_features']}
        
        Provide a JSON response with this exact structure:
        {{
            "plot_type": "comparison|feature_importance|single_metric|custom",
            "data_source": "performance|shap|lime|overlap",
            "models_to_compare": ["model1", "model2"],
            "metrics_to_show": ["metric1", "metric2"],
            "features_count": 10,
            "title": "Custom plot title",
            "chart_type": "bar|line|scatter|heatmap",
            "specific_analysis": "Brief description of what to analyze"
        }}
        
        ONLY return the JSON, no other text.
        """
        
        response = gemini_model.generate_content(analysis_prompt)
        
        # Parse the JSON response
        import json
        try:
            plot_specs = json.loads(response.text.strip())
            return plot_specs
        except json.JSONDecodeError:
            # If JSON parsing fails, try to extract JSON from the response
            import re
            json_match = re.search(r'\{.*\}', response.text, re.DOTALL)
            if json_match:
                plot_specs = json.loads(json_match.group())
                return plot_specs
            else:
                return {"error": "Could not parse AI response"}
                
    except Exception as e:
        return {"error": f"AI analysis failed: {str(e)}"}

def generate_ai_dynamic_plot(user_request: str, context: Dict[str, Any]) -> go.Figure:
    """Generate truly dynamic plots based on AI analysis of the request"""
    
    # Get AI analysis of the request
    plot_specs = ai_analyze_plot_request(user_request, context)
    
    if "error" in plot_specs:
        # Fallback to simple analysis
        return generate_fallback_plot(user_request, context, plot_specs.get("error", "Unknown error"))
    
    try:
        plot_type = plot_specs.get("plot_type", "comparison")
        data_source = plot_specs.get("data_source", "performance")
        models_to_compare = plot_specs.get("models_to_compare", [])
        metrics_to_show = plot_specs.get("metrics_to_show", [])
        title = plot_specs.get("title", "AI-Generated Plot")
        chart_type = plot_specs.get("chart_type", "bar")
        
        # Generate plot based on specifications
        if data_source == "performance" and plot_type == "comparison":
            return create_custom_performance_plot(models_to_compare, metrics_to_show, title, chart_type, context)
        elif data_source == "shap":
            return create_custom_shap_plot(models_to_compare, plot_specs.get("features_count", 10), title, context)
        elif data_source == "lime":
            return create_custom_lime_plot(models_to_compare, plot_specs.get("features_count", 10), title, context)
        else:
            return create_custom_performance_plot(models_to_compare, metrics_to_show, title, chart_type, context)
            
    except Exception as e:
        return generate_fallback_plot(user_request, context, f"Plot generation error: {str(e)}")

def create_custom_performance_plot(models: List[str], metrics: List[str], title: str, chart_type: str, context: Dict[str, Any]) -> go.Figure:
    """Create custom performance plots based on AI specifications"""
    perf_data = context.get("performance_data", {})
    if not perf_data:
        return create_error_plot("No performance data available")
    
    df_perf = pd.DataFrame(perf_data)
    
    # Filter models if specified
    if models:
        available_models = [m for m in models if m in df_perf.index]
        if available_models:
            df_perf = df_perf.loc[available_models]
    
    # Filter metrics if specified
    if metrics:
        available_metrics = [m for m in metrics if m in df_perf.columns]
        if available_metrics:
            df_perf = df_perf[available_metrics]
    
    # Generate plot based on chart type
    if chart_type == "bar":
        if len(df_perf.columns) == 1:
            # Single metric bar chart
            fig = px.bar(
                x=df_perf.index,
                y=df_perf.iloc[:, 0],
                title=title,
                labels={'x': 'Models', 'y': df_perf.columns[0].title()},
                color=df_perf.iloc[:, 0],
                color_continuous_scale='Viridis'
            )
        else:
            # Multiple metrics
            fig = px.bar(
                df_perf.reset_index(),
                x='index',
                y=df_perf.columns.tolist(),
                title=title,
                labels={'index': 'Models', 'value': 'Score'},
                barmode='group'
            )
    elif chart_type == "line":
        fig = px.line(
            df_perf.reset_index(),
            x='index',
            y=df_perf.columns.tolist(),
            title=title,
            labels={'index': 'Models', 'value': 'Score'}
        )
    else:
        # Default to bar
        fig = px.bar(
            df_perf.reset_index(),
            x='index',
            y=df_perf.columns.tolist(),
            title=title,
            barmode='group'
        )
    
    # Apply dark theme
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font_color='white',
        title_font_color='white'
    )
    
    return fig

def create_custom_shap_plot(models: List[str], features_count: int, title: str, context: Dict[str, Any]) -> go.Figure:
    """Create custom SHAP plots based on AI specifications"""
    shap_data = context.get("shap_data", {})
    if not shap_data:
        return create_error_plot("No SHAP data available")
    
    # If specific models requested, use the first available one
    model_to_use = None
    if models:
        for model in models:
            if model in shap_data:
                model_to_use = model
                break
    
    if not model_to_use:
        model_to_use = list(shap_data.keys())[0]
    
    model_data = pd.Series(shap_data[model_to_use])
    model_data = pd.to_numeric(model_data, errors='coerce').dropna()
    model_data = model_data[model_data != 0]
    top_features = model_data.nlargest(features_count)
    
    fig = px.bar(
        x=top_features.values,
        y=top_features.index,
        orientation='h',
        title=title,
        labels={'x': 'SHAP Importance', 'y': 'Features'},
        color=top_features.values,
        color_continuous_scale='Viridis'
    )
    
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font_color='white',
        title_font_color='white'
    )
    
    return fig

def create_custom_lime_plot(models: List[str], features_count: int, title: str, context: Dict[str, Any]) -> go.Figure:
    """Create custom LIME plots based on AI specifications"""
    lime_data = context.get("lime_data", {})
    if not lime_data:
        return create_error_plot("No LIME data available")
    
    # If specific models requested, use the first available one
    model_to_use = None
    if models:
        for model in models:
            if model in lime_data:
                model_to_use = model
                break
    
    if not model_to_use:
        model_to_use = list(lime_data.keys())[0]
    
    model_data = pd.Series(lime_data[model_to_use])
    model_data = pd.to_numeric(model_data, errors='coerce').dropna()
    model_data = model_data[model_data != 0]
    top_features = model_data.nlargest(features_count)
    
    fig = px.bar(
        x=top_features.values,
        y=top_features.index,
        orientation='h',
        title=title,
        labels={'x': 'LIME Weight', 'y': 'Feature Rules'},
        color=top_features.values,
        color_continuous_scale='Plasma'
    )
    
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font_color='white',
        title_font_color='white'
    )
    
    return fig

def generate_fallback_plot(user_request: str, context: Dict[str, Any], error_msg: str) -> go.Figure:
    """Generate a fallback plot when AI analysis fails"""
    perf_data = context.get("performance_data", {})
    
    if perf_data:
        df_perf = pd.DataFrame(perf_data)
        
        # Simple keyword analysis as fallback
        if "xgboost" in user_request.lower() and "random" in user_request.lower():
            # Compare XGBoost and Random Forest
            models_to_show = ["XGBoost", "Random Forest"]
            available_models = [m for m in models_to_show if m in df_perf.index]
            
            if available_models:
                df_filtered = df_perf.loc[available_models]
                
                fig = px.bar(
                    df_filtered.reset_index(),
                    x='index',
                    y=df_filtered.columns.tolist(),
                    title=f"🤖 AI-Requested Comparison: {' vs '.join(available_models)}",
                    labels={'index': 'Models', 'value': 'Performance Score'},
                    barmode='group'
                )
                
                fig.update_layout(
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font_color='white',
                    title_font_color='white'
                )
                
                return fig
    
    return create_error_plot(f"Could not generate custom plot: {error_msg}")

def create_error_plot(error_message: str) -> go.Figure:
    """Create an error plot with informative message"""
    fig = go.Figure()
    fig.add_annotation(
        text=f"⚠️ {error_message}<br><br>Try requests like:<br>• 'Compare XGBoost and Random Forest F1 scores'<br>• 'Show SHAP features for XGBoost'<br>• 'Plot all model accuracies'",
        x=0.5, y=0.5,
        showarrow=False,
        font=dict(size=14, color="white")
    )
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font_color='white',
        title="🔍 Plot Generation Assistant"
    )
    return fig
    """Generate dynamic plots based on user requests"""
    plot_request = plot_request.lower()
    
    # SHAP plots
    if "shap" in plot_request and ("top" in plot_request or "importance" in plot_request):
        try:
            # Extract number if specified
            num_features = 10
            if "5" in plot_request:
                num_features = 5
            elif "15" in plot_request:
                num_features = 15
            elif "20" in plot_request:
                num_features = 20
            
            # Extract model if specified
            model = "Random Forest"  # default
            if "xgboost" in plot_request:
                model = "XGBoost"
            elif "logistic" in plot_request:
                model = "Logistic Regression"
            
            # Get SHAP data from context
            shap_data = context.get("shap_data", {})
            if model in shap_data:
                model_data = pd.Series(shap_data[model])
                model_data = pd.to_numeric(model_data, errors='coerce').dropna()
                model_data = model_data[model_data != 0]
                top_features = model_data.nlargest(num_features)
                
                fig = px.bar(
                    x=top_features.values,
                    y=top_features.index,
                    orientation='h',
                    title=f"🎯 Top {num_features} SHAP Features - {model}",
                    labels={'x': 'SHAP Importance', 'y': 'Features'},
                    color=top_features.values,
                    color_continuous_scale='Viridis'
                )
                fig.update_layout(
                    height=600,
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font_color='white',
                    title_font_color='white'
                )
                return fig
        except Exception as e:
            st.error(f"Error generating SHAP plot: {e}")
    
    # Performance comparison plots
    elif "performance" in plot_request or "comparison" in plot_request or "f1" in plot_request or "model" in plot_request:
        try:
            perf_data = context.get("performance_data", {})
            if perf_data:
                df_perf = pd.DataFrame(perf_data)
                
                # Single metric plot if specified
                if "f1" in plot_request:
                    fig = px.bar(
                        x=df_perf.index,
                        y=df_perf['f1'] if 'f1' in df_perf.columns else df_perf.iloc[:, 0],
                        title="🏆 Model F1-Score Comparison",
                        labels={'x': 'Models', 'y': 'F1 Score'},
                        color=df_perf['f1'] if 'f1' in df_perf.columns else df_perf.iloc[:, 0],
                        color_continuous_scale='Viridis'
                    )
                    fig.update_layout(
                        height=500,
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)',
                        font_color='white',
                        title_font_color='white'
                    )
                    return fig
                
                # Multi-metric comparison
                else:
                    fig = make_subplots(
                        rows=2, cols=2,
                        subplot_titles=['Accuracy', 'F1 Score', 'Precision', 'Recall'],
                        specs=[[{"type": "bar"}, {"type": "bar"}],
                               [{"type": "bar"}, {"type": "bar"}]]
                    )
                    
                    metrics = ['accuracy', 'f1', 'precision', 'recall']
                    colors = ['#00ff88', '#00aaff', '#ff8800', '#ff0088']
                    
                    for i, metric in enumerate(metrics):
                        if metric in df_perf.columns:
                            row = (i // 2) + 1
                            col = (i % 2) + 1
                            
                            fig.add_trace(
                                go.Bar(
                                    x=df_perf.index,
                                    y=df_perf[metric],
                                    name=metric.capitalize(),
                                    marker_color=colors[i],
                                    showlegend=False
                                ),
                                row=row, col=col
                            )
                    
                    fig.update_layout(
                        height=600,
                        title_text="🏆 Model Performance Comparison",
                        title_x=0.5,
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)',
                        font_color='white',
                        title_font_color='white'
                    )
                    return fig
        except Exception as e:
            st.error(f"Error generating performance plot: {e}")
    
    # LIME plots
    elif "lime" in plot_request and ("top" in plot_request or "importance" in plot_request):
        try:
            # Extract number if specified
            num_features = 10
            if "5" in plot_request:
                num_features = 5
            elif "15" in plot_request:
                num_features = 15
            elif "20" in plot_request:
                num_features = 20
            
            # Extract model if specified
            model = "Random Forest"  # default
            if "xgboost" in plot_request:
                model = "XGBoost"
            elif "logistic" in plot_request:
                model = "Logistic Regression"
            
            # Get LIME data from context
            lime_data = context.get("lime_data", {})
            if model in lime_data:
                model_data = pd.Series(lime_data[model])
                model_data = pd.to_numeric(model_data, errors='coerce').dropna()
                model_data = model_data[model_data != 0]
                top_features = model_data.nlargest(num_features)
                
                fig = px.bar(
                    x=top_features.values,
                    y=top_features.index,
                    orientation='h',
                    title=f"🍋 Top {num_features} LIME Features - {model}",
                    labels={'x': 'LIME Weight', 'y': 'Feature Rules'},
                    color=top_features.values,
                    color_continuous_scale='Plasma'
                )
                fig.update_layout(
                    height=600,
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font_color='white',
                    title_font_color='white'
                )
                return fig
        except Exception as e:
            st.error(f"Error generating LIME plot: {e}")
    
    # Overlap analysis plot
    elif "overlap" in plot_request:
        try:
            overlap_data = context.get("overlap_data", {})
            if overlap_data and 'overlap_percentage' in overlap_data:
                df_overlap = pd.DataFrame(overlap_data)
                
                fig = px.bar(
                    x=df_overlap.index,
                    y=df_overlap['overlap_percentage'],
                    title="🔄 SHAP-LIME Overlap Percentage by Model",
                    labels={'x': 'Models', 'y': 'Overlap Percentage (%)'},
                    color=df_overlap['overlap_percentage'],
                    color_continuous_scale='Viridis'
                )
                fig.update_layout(
                    height=500,
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font_color='white',
                    title_font_color='white'
                )
                return fig
        except Exception as e:
            st.error(f"Error generating overlap plot: {e}")
    
    # Default: Create a sample plot showing data is available
    try:
        perf_data = context.get("performance_data", {})
        if perf_data:
            df_perf = pd.DataFrame(perf_data)
            if 'f1' in df_perf.columns:
                fig = px.bar(
                    x=df_perf.index,
                    y=df_perf['f1'],
                    title="🏆 Model F1-Score Comparison (Default View)",
                    labels={'x': 'Models', 'y': 'F1 Score'},
                    color=df_perf['f1'],
                    color_continuous_scale='Viridis'
                )
                fig.update_layout(
                    height=500,
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font_color='white',
                    title_font_color='white'
                )
                return fig
    except Exception as e:
        st.error(f"Error generating default plot: {e}")
    
    # If all else fails, create an informational plot
    fig = go.Figure()
    fig.add_annotation(
        text="📊 Plot Generation Available!<br><br>Try specific requests like:<br>• 'Show SHAP importance for top 10 features'<br>• 'Generate performance comparison'<br>• 'Show F1 scores for all models'<br>• 'Display LIME analysis'",
        x=0.5, y=0.5,
        showarrow=False,
        font=dict(size=16, color="white")
    )
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font_color='white',
        title="🎯 Dynamic Plot Generator Ready"
    )
    return fig

def ai_chatbot_response(user_question: str, context: Dict[str, Any]) -> str:
    """Generate AI response using Gemini with data context"""
    if not gemini_model:
        return "🔴 Gemini AI is not configured. Please check the API key."
    
    try:
        # Prepare context summary for the AI
        context_summary = f"""
        You are an expert ML explainability analyst. Answer questions about this loan default prediction project.
        
        DATA SUMMARY:
        - Models analyzed: {context['data_summary']['models_count']}
        - Best performing model: {context['data_summary']['best_model']} (F1: {context['data_summary']['best_f1']:.3f})
        - Features analyzed: {context['data_summary']['features_count']}
        
        AVAILABLE DATA:
        - Model performance metrics (accuracy, F1, precision, recall)
        - SHAP feature importance (global explanations)
        - LIME feature importance (local explanations)  
        - Feature overlap analysis between SHAP and LIME
        - GenAI explanations for each model
        
        GENAI EXPLANATIONS AVAILABLE:
        {', '.join(context['genai_explanations'].keys())}
        
        Please provide business-friendly explanations and actionable insights. 
        If asked about specific models or features, reference the actual data when possible.
        """
        
        # Combine context and user question
        prompt = f"{context_summary}\n\nUSER QUESTION: {user_question}\n\nPROVIDE A HELPFUL RESPONSE:"
        
        response = gemini_model.generate_content(prompt)
        return response.text
        
    except Exception as e:
        return f"🔴 Error generating AI response: {str(e)}"

def generate_ai_insights(feature_data, model_performance, analysis_type="SHAP"):
    """Generate AI-powered insights from the analysis results"""
    if feature_data is None or feature_data.empty:
        return "No data available for analysis."
    
    insights = []
    
    # Performance insights
    if model_performance is not None and not model_performance.empty:
        try:
            if 'f1' in model_performance.columns:
                best_model = model_performance['f1'].idxmax()
                best_f1 = model_performance['f1'].max()
                insights.append(f"🏆 **Best Performing Model**: {best_model} with F1 score of {best_f1:.3f}")
            else:
                insights.append("🏆 **Model Performance**: Analysis available in performance tab")
        except Exception as e:
            insights.append("🏆 **Model Performance**: Analysis available in performance tab")
    
    # Feature insights
    if analysis_type == "SHAP":
        try:
            # Handle different data structures for SHAP
            if hasattr(feature_data, 'mean') and callable(feature_data.mean):
                # DataFrame case
                numeric_data = feature_data.select_dtypes(include=[np.number])
                if not numeric_data.empty:
                    top_features = numeric_data.mean(axis=1).nlargest(3)
                    feature_names = [str(name) for name in top_features.index]
                    insights.append(f"🎯 **Top {analysis_type} Features**: {', '.join(feature_names[:3])}")
                    insights.append(f"💡 **Key Insight**: {feature_names[0]} shows the strongest global impact on predictions")
                else:
                    insights.append(f"🎯 **SHAP Analysis**: Feature importance data available in SHAP tab")
            else:
                insights.append(f"🎯 **SHAP Analysis**: Feature importance data available in SHAP tab")
        except Exception as e:
            insights.append(f"🎯 **SHAP Analysis**: Feature importance data available in SHAP tab")
    elif analysis_type == "LIME":
        insights.append(f"🍋 **LIME Analysis**: Shows local feature importance patterns that complement global SHAP insights")
    
    return "\n\n".join(insights) if insights else "Analysis in progress..."

def create_performance_plot(df_perf):
    """Create interactive performance comparison plot"""
    if df_perf is None or df_perf.empty:
        return None
    
    metrics = ['accuracy', 'f1', 'precision', 'recall']
    available_metrics = [m for m in metrics if m in df_perf.columns]
    
    if not available_metrics:
        return None
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[m.capitalize() for m in available_metrics[:4]],
        specs=[[{"type": "bar"}, {"type": "bar"}],
               [{"type": "bar"}, {"type": "bar"}]]
    )
    
    colors = ['#00ff88', '#00aaff', '#ff8800', '#ff0088']
    
    for i, metric in enumerate(available_metrics[:4]):
        row = (i // 2) + 1
        col = (i % 2) + 1
        
        fig.add_trace(
            go.Bar(
                x=df_perf.index,
                y=df_perf[metric],
                name=metric.capitalize(),
                marker_color=colors[i],
                showlegend=False
            ),
            row=row, col=col
        )
    
    fig.update_layout(
        height=600,
        title_text="🏆 Model Performance Comparison",
        title_x=0.5,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font_color='white',
        title_font_color='white'
    )
    
    # Update subplot titles to white
    for annotation in fig['layout']['annotations']:
        annotation['font'] = dict(color='white')
    
    return fig

# =====================
# 📂 SIDEBAR - FILE UPLOADS
# =====================
st.sidebar.markdown("## 📂 Upload Pipeline Results")
st.sidebar.markdown("Upload your CSV files from the ML pipeline or use default data")

# File uploaders
st.sidebar.markdown("### 📁 Upload Your CSV Files")
perf_file = st.sidebar.file_uploader("📈 Model Performance CSV", 
                                    type=["csv"], 
                                    key="perf",
                                    help="Upload CSV with columns: accuracy, f1, precision, recall")

shap_file = st.sidebar.file_uploader("🎯 SHAP Feature Importance CSV", 
                                    type=["csv"], 
                                    key="shap",
                                    help="Upload CSV with features as rows, models as columns")

lime_file = st.sidebar.file_uploader("🍋 LIME Feature Importance CSV", 
                                    type=["csv"], 
                                    key="lime",
                                    help="Upload CSV with feature rules and weights")

overlap_file = st.sidebar.file_uploader("🔗 Feature Overlap Analysis CSV", 
                                       type=["csv"], 
                                       key="overlap",
                                       help="Upload CSV with overlap percentages by model")

# Load default data checkbox
use_default = st.sidebar.checkbox("📊 Use Default Pipeline Results", value=True)

# Load data
if use_default and not any([perf_file, shap_file, lime_file, overlap_file]):
    df_perf, df_shap, df_lime, df_overlap = load_default_data()
else:
    df_perf = load_csv_file(perf_file)
    df_shap = load_csv_file(shap_file)
    df_lime = load_csv_file(lime_file)
    df_overlap = load_csv_file(overlap_file)

# Load GenAI explanations
genai_explanations = load_genai_explanations()

# Prepare data context for chatbot
st.session_state.data_context = prepare_data_context(df_perf, df_shap, df_lime, df_overlap, genai_explanations)

# Data status indicators
st.sidebar.markdown("### 📊 Data Status")
data_status = {
    "Model Performance": "✅" if df_perf is not None else "❌",
    "SHAP Analysis": "✅" if df_shap is not None else "❌",
    "LIME Analysis": "✅" if df_lime is not None else "❌",
    "Feature Overlap": "✅" if df_overlap is not None else "❌",
    "GenAI Explanations": "✅" if genai_explanations else "❌"
}

for name, status in data_status.items():
    st.sidebar.markdown(f"{status} {name}")

# =====================
# 🤖 AI INSIGHTS SECTION
# =====================
if any([df_perf is not None, df_shap is not None, df_lime is not None]):
    st.markdown("## 🤖 AI-Powered Analysis Insights")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Generate comprehensive insights
        insights_text = ""
        
        if df_perf is not None:
            performance_insights = generate_ai_insights(df_shap, df_perf, "Performance")
            insights_text += performance_insights + "\n\n"
        
        if df_shap is not None:
            shap_insights = generate_ai_insights(df_shap, df_perf, "SHAP")
            insights_text += shap_insights + "\n\n"
        
        if df_lime is not None:
            lime_insights = generate_ai_insights(df_lime, df_perf, "LIME")
            insights_text += lime_insights
        
        st.markdown(f'<div class="insight-box">{insights_text}</div>', unsafe_allow_html=True)
    
    with col2:
        # Key metrics summary
        if df_perf is not None:
            st.markdown("### 📊 Quick Metrics")
            try:
                best_acc = df_perf['accuracy'].max() if 'accuracy' in df_perf.columns else 0
                best_f1 = df_perf['f1'].max() if 'f1' in df_perf.columns else 0
                avg_precision = df_perf['precision'].mean() if 'precision' in df_perf.columns else 0
                
                st.metric("🎯 Best Accuracy", f"{best_acc:.3f}")
                st.metric("🏆 Best F1 Score", f"{best_f1:.3f}")
                st.metric("⚖️ Avg Precision", f"{avg_precision:.3f}")
            except:
                st.info("Metrics calculation unavailable")

# =====================
# 📋 MAIN DASHBOARD TABS
# =====================
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "📈 Model Performance", 
    "🎯 SHAP Analysis", 
    "🍋 LIME Analysis", 
    "🔗 Feature Overlap",
    "🤖 GenAI Explanations",
    "💬 AI Chatbot",
    "🔍 Detailed Insights"
])

# =====================
# 📈 TAB 1: MODEL PERFORMANCE
# =====================
with tab1:
    st.header("🏆 Model Performance Analysis")
    
    if df_perf is not None:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Interactive performance plot
            perf_fig = create_performance_plot(df_perf)
            if perf_fig:
                st.plotly_chart(perf_fig, use_container_width=True)
        
        with col2:
            st.subheader("📊 Performance Table")
            st.dataframe(df_perf.round(3), use_container_width=True)
        
        # Model comparison insights
        st.subheader("📋 Model Analysis")
        if 'f1' in df_perf.columns:
            best_model = df_perf['f1'].idxmax()
            worst_model = df_perf['f1'].idxmin()
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("🥇 Best Model (F1)", best_model, f"{df_perf.loc[best_model, 'f1']:.3f}")
            with col2:
                st.metric("📊 Models Analyzed", len(df_perf), "")
            with col3:
                st.metric("📈 Avg F1 Score", f"{df_perf['f1'].mean():.3f}", "")
    else:
        st.warning("📤 Upload model_performance.csv to see results or enable default data.")
        st.info("💡 Expected format: CSV with columns like 'accuracy', 'f1', 'precision', 'recall'")

# =====================
# 🎯 TAB 2: SHAP ANALYSIS
# =====================
with tab2:
    st.header("🎯 SHAP Feature Importance Analysis")
    
    if df_shap is not None:
        # Model selection for SHAP
        available_models = [col for col in df_shap.columns if col not in ['Unnamed: 0', 'index']]
        
        if not available_models:
            st.warning("No model columns found in SHAP data")
            st.dataframe(df_shap.head())
        else:
            selected_model = st.selectbox("Select Model for SHAP Analysis", available_models)
            
            if selected_model in df_shap.columns:
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    # SHAP importance plot
                    try:
                        # Get the SHAP values for the selected model
                        shap_values = df_shap[selected_model]
                        
                        # Remove any non-numeric or zero values
                        numeric_shap = pd.to_numeric(shap_values, errors='coerce')
                        numeric_shap = numeric_shap.dropna()
                        numeric_shap = numeric_shap[numeric_shap != 0]  # Remove zero values
                        
                        if len(numeric_shap) > 0:
                            top_shap = numeric_shap.nlargest(15)
                            
                            fig = px.bar(
                                x=top_shap.values,
                                y=top_shap.index,
                                orientation='h',
                                title=f"🎯 Top 15 SHAP Features - {selected_model}",
                                labels={'x': 'SHAP Importance', 'y': 'Features'},
                                color=top_shap.values,
                                color_continuous_scale='Viridis'
                            )
                            fig.update_layout(
                                height=600,
                                plot_bgcolor='rgba(0,0,0,0)',
                                paper_bgcolor='rgba(0,0,0,0)',
                                font_color='white',
                                title_font_color='white'
                            )
                            st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.warning(f"No non-zero SHAP values found for {selected_model}")
                            st.info("This might indicate that the model has no feature importance data.")
                    except Exception as e:
                        st.error(f"Error creating SHAP plot: {e}")
                        st.info("Raw SHAP data:")
                        st.dataframe(df_shap[selected_model].head(20))
            
                with col2:
                    st.subheader(f"📊 {selected_model} SHAP Data")
                    try:
                        # Get SHAP values and remove zeros
                        shap_values = df_shap[selected_model]
                        numeric_shap = pd.to_numeric(shap_values, errors='coerce')
                        numeric_shap = numeric_shap.dropna()
                        numeric_shap = numeric_shap[numeric_shap != 0]  # Remove zero values
                        
                        if len(numeric_shap) > 0:
                            shap_data = numeric_shap.nlargest(20)
                            st.dataframe(shap_data, use_container_width=True)
                            
                            # Feature statistics
                            st.subheader("📈 SHAP Statistics")
                            st.metric("🔝 Top Feature", str(shap_data.index[0]))
                            st.metric("📊 Max Importance", f"{shap_data.iloc[0]:.4f}")
                            st.metric("📉 Features w/ Importance", len(shap_data))
                        else:
                            st.warning(f"No non-zero SHAP values for {selected_model}")
                            # Show raw data for debugging
                            st.dataframe(df_shap[selected_model].head(20))
                    except Exception as e:
                        st.error(f"Error processing SHAP data: {e}")
                        st.dataframe(df_shap[selected_model].head(20))
        
        # Cross-model SHAP comparison
        if len(available_models) > 1:
            st.subheader("🔄 Cross-Model SHAP Comparison")
            
            # Get top 10 features from each model
            comparison_data = {}
            for model in available_models:
                try:
                    # Convert to numeric and handle data type issues
                    shap_values = df_shap[model]
                    numeric_shap = pd.to_numeric(shap_values, errors='coerce')
                    numeric_shap = numeric_shap.dropna()
                    numeric_shap = numeric_shap[numeric_shap != 0]  # Remove zero values
                    
                    if len(numeric_shap) > 0:
                        top_features = numeric_shap.nlargest(10)
                        comparison_data[model] = top_features
                    else:
                        st.warning(f"No numeric data available for {model}")
                except Exception as e:
                    st.warning(f"Error processing {model}: {e}")
            
            if comparison_data:
                comparison_df = pd.DataFrame(comparison_data).fillna(0)
                
                # Create heatmap
                fig = px.imshow(
                    comparison_df.T,
                    aspect="auto",
                    title="🔥 SHAP Feature Importance Heatmap (All Models)",
                    color_continuous_scale='Viridis'
                )
                fig.update_layout(
                    height=400,
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font_color='white',
                    title_font_color='white'
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("No data available for cross-model comparison")
            
        else:
            st.info("Need at least 2 models for cross-model comparison")
    else:
        st.warning("📤 Upload shap_feature_importance.csv to see SHAP analysis.")
        st.info("💡 Expected format: CSV with features as rows and models as columns")

# =====================
# 🍋 TAB 3: LIME ANALYSIS
# =====================
with tab3:
    st.header("🍋 LIME Feature Importance Analysis")
    
    if df_lime is not None:
        # Display LIME results
        st.subheader("📊 LIME Feature Analysis")
        
        # Check if data has the expected structure
        try:
            # Get available models (exclude index columns)
            lime_models = [col for col in df_lime.columns if col not in ['Unnamed: 0', 'index']]
            
            if lime_models:
                selected_lime_model = st.selectbox("Select Model for LIME Analysis", lime_models, key="lime_model")
                
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    # LIME importance plot
                    try:
                        if selected_lime_model in df_lime.columns:
                            # Get LIME values and remove zeros
                            lime_values = df_lime[selected_lime_model]
                            numeric_lime = pd.to_numeric(lime_values, errors='coerce')
                            numeric_lime = numeric_lime.dropna()
                            numeric_lime = numeric_lime[numeric_lime != 0]  # Remove zero values
                            
                            if len(numeric_lime) > 0:
                                lime_data = numeric_lime.nlargest(15)
                                
                                fig = px.bar(
                                    x=lime_data.values,
                                    y=lime_data.index,
                                    orientation='h',
                                    title=f"🍋 Top 15 LIME Features - {selected_lime_model}",
                                    labels={'x': 'LIME Weight', 'y': 'Feature Rules'},
                                    color=lime_data.values,
                                    color_continuous_scale='Plasma'
                                )
                                fig.update_layout(
                                    height=600,
                                    plot_bgcolor='rgba(0,0,0,0)',
                                    paper_bgcolor='rgba(0,0,0,0)',
                                    font_color='white',
                                    title_font_color='white'
                                )
                                st.plotly_chart(fig, use_container_width=True)
                            else:
                                st.warning(f"No non-zero LIME values found for {selected_lime_model}")
                                st.info("This might indicate limited local explanations for this model.")
                        else:
                            st.error(f"Model {selected_lime_model} not found in LIME data")
                    except Exception as e:
                        st.error(f"Error creating LIME plot: {e}")
                        st.info("Raw LIME data structure:")
                        st.dataframe(df_lime.head(10))
                
                with col2:
                    st.subheader(f"📋 {selected_lime_model} LIME Data")
                    try:
                        if selected_lime_model in df_lime.columns:
                            # Get LIME values and remove zeros
                            lime_values = df_lime[selected_lime_model]
                            numeric_lime = pd.to_numeric(lime_values, errors='coerce')
                            numeric_lime = numeric_lime.dropna()
                            numeric_lime = numeric_lime[numeric_lime != 0]  # Remove zero values
                            
                            if len(numeric_lime) > 0:
                                lime_display = numeric_lime.nlargest(20)
                                st.dataframe(lime_display, use_container_width=True)
                                
                                # LIME statistics
                                st.subheader("📈 LIME Statistics")
                                st.metric("🎯 Top Feature", str(lime_display.index[0]))
                                st.metric("⚖️ Max Weight", f"{lime_display.iloc[0]:.4f}")
                                st.metric("📊 Features w/ Weight", len(lime_display))
                            else:
                                st.warning(f"No non-zero LIME values for {selected_lime_model}")
                                st.dataframe(df_lime[selected_lime_model].head(20))
                        else:
                            st.error(f"Model {selected_lime_model} not found")
                    except Exception as e:
                        st.error(f"Error processing LIME data: {e}")
                        st.dataframe(df_lime.head(20))
        except Exception as e:
            st.error(f"Error processing LIME data: {e}")
            st.dataframe(df_lime, use_container_width=True)
    else:
        st.warning("📤 Upload lime_feature_importance.csv to see LIME analysis.")
        st.info("💡 Expected format: CSV with feature rules and importance weights")

# =====================
# 🔗 TAB 4: FEATURE OVERLAP
# =====================
with tab4:
    st.header("🔗 SHAP-LIME Feature Overlap Analysis")
    
    if df_overlap is not None:
        st.subheader("📊 Feature Overlap Results")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            # Display overlap table
            st.dataframe(df_overlap, use_container_width=True)
        
        with col2:
            # Overlap percentage visualization
            if 'overlap_percentage' in df_overlap.columns:
                # Create a proper chart with model names
                model_names = df_overlap.index.tolist()
                overlap_values = df_overlap['overlap_percentage'].tolist()
                
                fig = px.bar(
                    x=model_names,
                    y=overlap_values,
                    title="🔄 SHAP-LIME Overlap Percentage by Model",
                    labels={'x': 'Models', 'y': 'Overlap Percentage (%)'},
                    color=overlap_values,
                    color_continuous_scale='Viridis'
                )
                fig.update_layout(
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font_color='white',
                    title_font_color='white'
                )
                st.plotly_chart(fig, use_container_width=True)
        
        # Overlap insights
        if 'overlap_percentage' in df_overlap.columns:
            avg_overlap = df_overlap['overlap_percentage'].mean()
            max_overlap = df_overlap['overlap_percentage'].max()
            max_overlap_model = df_overlap['overlap_percentage'].idxmax()
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("📊 Average Overlap", f"{avg_overlap:.1f}%")
            with col2:
                st.metric("🏆 Highest Overlap", f"{max_overlap:.1f}%")
            with col3:
                st.metric("🎯 Best Agreement", max_overlap_model)
            
            # Interpretation based on your actual data
            if avg_overlap > 15:
                interpretation = "🟢 **Moderate Agreement**: SHAP and LIME show some consistent feature importance patterns, indicating reliable insights"
            elif avg_overlap > 5:
                interpretation = "🟡 **Low Agreement**: SHAP and LIME capture different aspects of model behavior - both perspectives provide complementary insights"
            else:
                interpretation = "🔴 **Minimal Agreement**: SHAP and LIME focus on very different features - consider both for comprehensive understanding"
            
            st.markdown(f'<div class="insight-box">{interpretation}</div>', unsafe_allow_html=True)
            
            # Show overlap details
            st.subheader("🔍 Detailed Overlap Analysis")
            if 'overlap_features' in df_overlap.columns:
                for model in df_overlap.index:
                    overlap_features = df_overlap.loc[model, 'overlap_features']
                    overlap_count = df_overlap.loc[model, 'overlap_count']
                    st.write(f"**{model}**: {overlap_count} overlapping features - {overlap_features}")
            
        else:
            st.warning("Overlap percentage data not available in the expected format")
    else:
        st.warning("📤 Upload feature_overlap_analysis.csv to see overlap analysis.")

# =====================
# 🤖 TAB 5: GENAI EXPLANATIONS
# =====================
with tab5:
    st.header("🤖 GenAI Model Explanations")
    
    if genai_explanations:
        st.subheader("� AI-Powered Model Insights")
        
        # Model selection for GenAI explanations
        available_models = [key for key in genai_explanations.keys() if key != 'model_comparison' and key != 'json_data']
        
        if available_models:
            selected_genai_model = st.selectbox(
                "Select Model for GenAI Explanation", 
                available_models,
                key="genai_model_select"
            )
            
            if selected_genai_model in genai_explanations:
                # Display the GenAI explanation
                explanation_text = genai_explanations[selected_genai_model]
                
                # Parse and format the explanation
                st.markdown(f'<div class="insight-box">{explanation_text}</div>', unsafe_allow_html=True)
        
        # Model comparison section
        if 'model_comparison' in genai_explanations:
            st.subheader("🏆 AI-Generated Model Comparison")
            comparison_text = genai_explanations['model_comparison']
            st.markdown(f'<div class="insight-box">{comparison_text}</div>', unsafe_allow_html=True)
        
        # JSON insights
        if 'json_data' in genai_explanations:
            st.subheader("📊 Structured GenAI Insights")
            with st.expander("View Detailed JSON Data"):
                st.json(genai_explanations['json_data'])
        
        # Generate custom explanation
        st.subheader("🎯 Custom AI Explanation")
        
        custom_prompt = st.text_area(
            "Ask for a specific explanation or insight:",
            placeholder="E.g., 'Explain why XGBoost performed better than Random Forest for loan default prediction'",
            key="custom_genai_prompt"
        )
        
        if st.button("🚀 Generate Custom Explanation") and custom_prompt:
            if gemini_model:
                with st.spinner("🤖 Generating AI explanation..."):
                    try:
                        # Create context for custom explanation
                        context_for_prompt = f"""
                        You are explaining a loan default prediction ML project. 
                        Available models: Random Forest, XGBoost, Logistic Regression
                        Best model: {st.session_state.data_context['data_summary']['best_model']}
                        
                        Context: {str(st.session_state.data_context)[:2000]}
                        """
                        
                        full_prompt = f"{context_for_prompt}\n\nUser Question: {custom_prompt}\n\nProvide a detailed explanation:"
                        
                        response = gemini_model.generate_content(full_prompt)
                        st.markdown(f'<div class="insight-box">{response.text}</div>', unsafe_allow_html=True)
                        
                    except Exception as e:
                        st.error(f"Error generating explanation: {e}")
            else:
                st.error("GenAI model not configured. Please check API key.")
    
    else:
        st.warning("📤 GenAI explanations not found. Please run the GenAI explainer notebook first.")
        st.info("💡 Run `genai_explainer.ipynb` to generate AI-powered model explanations")

# =====================
# 💬 TAB 6: AI CHATBOT
# =====================
with tab6:
    st.header("💬 Interactive AI Assistant")
    st.subheader("🤖 Ask me anything about your ML pipeline results!")
    
    # Chat interface
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # Chat history display
        st.markdown("### 💬 Conversation")
        
        # Display chat history
        for i, (role, message) in enumerate(st.session_state.chat_history):
            if role == "user":
                st.markdown(f'<div style="background-color: #1a1a1a; padding: 10px; margin: 5px 0; border-radius: 10px; border-left: 3px solid #00ff88;"><strong>You:</strong> {message}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div style="background-color: #0a2a0a; padding: 10px; margin: 5px 0; border-radius: 10px; border-left: 3px solid #00aaff;"><strong>AI Assistant:</strong> {message}</div>', unsafe_allow_html=True)
        
        # User input
        user_input = st.text_input(
            "Ask your question:",
            placeholder="E.g., 'Why did XGBoost perform better?', 'Show SHAP importance for top 10 features', 'What are the key insights?'",
            key="chat_input"
        )
        
        col_send, col_plot, col_clear = st.columns([1, 1, 1])
        
        with col_send:
            send_button = st.button("📤 Send", use_container_width=True)
        
        with col_plot:
            generate_plot_button = st.button("📊 Generate Plot", use_container_width=True)
        
        with col_clear:
            clear_button = st.button("🗑️ Clear Chat", use_container_width=True)
        
        # Additional plot generation button for last question
        plot_from_last_button = st.button("� Generate Plot from Last Question", use_container_width=True)
        
        # Handle user input
        if send_button and user_input:
            # Add user message to history
            st.session_state.chat_history.append(("user", user_input))
            
            # Generate AI response
            if gemini_model and st.session_state.data_context:
                with st.spinner("🤖 AI is thinking..."):
                    ai_response = ai_chatbot_response(user_input, st.session_state.data_context)
                    st.session_state.chat_history.append(("assistant", ai_response))
                    
                    # Auto-generate plot if request seems plot-related
                    plot_keywords = ["plot", "chart", "graph", "show", "display", "visualiz", "comparison", "shap", "lime", "performance", "f1", "compare"]
                    if any(keyword in user_input.lower() for keyword in plot_keywords):
                        try:
                            plot_fig = generate_ai_dynamic_plot(user_input, st.session_state.data_context)
                            if plot_fig:
                                st.plotly_chart(plot_fig, use_container_width=True)
                        except Exception as e:
                            st.error(f"Error auto-generating plot: {e}")
            else:
                error_msg = "🔴 AI assistant not available. Please check configuration and data."
                st.session_state.chat_history.append(("assistant", error_msg))
            
            st.rerun()
        
        # Handle direct plot generation
        if generate_plot_button and user_input:
            try:
                with st.spinner("🤖 AI is analyzing your plot request..."):
                    plot_fig = generate_ai_dynamic_plot(user_input, st.session_state.data_context)
                    if plot_fig:
                        st.session_state.current_plot = plot_fig
                        st.session_state.plot_title = f"Plot: {user_input}"
                        st.success("✅ Plot generated successfully!")
                        st.rerun()
                    else:
                        st.warning("⚠️ Could not generate plot from that request. Try being more specific.")
            except Exception as e:
                st.error(f"Error generating plot: {e}")
        
        # Clear chat history
        if clear_button:
            st.session_state.chat_history = []
            st.rerun()
        
        # Generate plot from last question
        if plot_from_last_button and st.session_state.chat_history:
            last_user_message = None
            for role, message in reversed(st.session_state.chat_history):
                if role == "user":
                    last_user_message = message
                    break
            
            if last_user_message:
                try:
                    st.subheader("📊 Generated Plot")
                    plot_fig = generate_ai_dynamic_plot(last_user_message, st.session_state.data_context)
                    if plot_fig:
                        st.plotly_chart(plot_fig, use_container_width=True)
                        st.success("✅ AI-powered plot generated successfully!")
                    else:
                        st.warning("⚠️ Could not generate plot from that request. Try being more specific.")
                except Exception as e:
                    st.error(f"Error generating AI plot: {e}")
            else:
                st.warning("⚠️ No previous question found to generate plot from.")
    
    with col2:
        # Quick action buttons
        st.markdown("### 🚀 Quick Questions")
        
        quick_questions = [
            "What is the best performing model?",
            "Why did XGBoost perform better?",
            "What are the top 5 most important features?",
            "How do SHAP and LIME compare?",
            "What are the key business insights?",
            "Show me performance comparison",
            "Explain the model overlap analysis",
            "What recommendations do you have?"
        ]
        
        for question in quick_questions:
            if st.button(f"💭 {question}", key=f"quick_{hash(question)}", use_container_width=True):
                # Add to chat and process
                st.session_state.chat_history.append(("user", question))
                
                if gemini_model and st.session_state.data_context:
                    with st.spinner("🤖 AI is thinking..."):
                        ai_response = ai_chatbot_response(question, st.session_state.data_context)
                        st.session_state.chat_history.append(("assistant", ai_response))
                else:
                    error_msg = "🔴 AI assistant not available."
                    st.session_state.chat_history.append(("assistant", error_msg))
                
                st.rerun()
        
        # Plot generation buttons
        st.markdown("### 📊 Quick Plots")
        
        plot_buttons = [
            "Show F1 scores for all models",
            "Compare XGBoost and Random Forest",
            "Show SHAP importance for top 10 features",
            "Generate performance comparison",
            "Show LIME analysis for XGBoost",
            "Display overlap analysis"
        ]
        
        for plot_request in plot_buttons:
            if st.button(f"📈 {plot_request}", key=f"plot_{hash(plot_request)}", use_container_width=True):
                try:
                    plot_fig = generate_ai_dynamic_plot(plot_request, st.session_state.data_context)
                    if plot_fig:
                        # Display plot in main area by adding to session state
                        st.session_state.current_plot = plot_fig
                        st.session_state.plot_title = plot_request
                        st.rerun()
                except Exception as e:
                    st.error(f"Error generating AI plot: {e}")
        
        # Custom plot request
        st.markdown("### 🎨 Custom Plot Request")
        custom_plot_request = st.text_input(
            "Describe your custom plot:",
            placeholder="e.g., 'Compare XGBoost and Random Forest F1 scores'",
            key="custom_plot_input"
        )
        
        if st.button("🚀 Generate Custom Plot", use_container_width=True) and custom_plot_request:
            try:
                with st.spinner("🤖 AI is analyzing your request..."):
                    plot_fig = generate_ai_dynamic_plot(custom_plot_request, st.session_state.data_context)
                    if plot_fig:
                        st.session_state.current_plot = plot_fig
                        st.session_state.plot_title = f"Custom: {custom_plot_request}"
                        st.rerun()
            except Exception as e:
                st.error(f"Error generating custom plot: {e}")
        
        # Data context info
        st.markdown("### 📊 Available Data")
        st.markdown(f"**Models:** {st.session_state.data_context.get('data_summary', {}).get('models_count', 0)}")
        st.markdown(f"**Features:** {st.session_state.data_context.get('data_summary', {}).get('features_count', 0)}")
        st.markdown(f"**Best Model:** {st.session_state.data_context.get('data_summary', {}).get('best_model', 'Unknown')}")
        st.markdown(f"**GenAI Explanations:** {'✅' if genai_explanations else '❌'}")
    
    # Display current plot if available
    if hasattr(st.session_state, 'current_plot') and st.session_state.current_plot:
        st.subheader(f"📊 {st.session_state.get('plot_title', 'Generated Plot')}")
        st.plotly_chart(st.session_state.current_plot, use_container_width=True)
        
        if st.button("🗑️ Clear Plot"):
            del st.session_state.current_plot
            if hasattr(st.session_state, 'plot_title'):
                del st.session_state.plot_title
            st.rerun()

# =====================
# �🔍 TAB 7: DETAILED INSIGHTS
# =====================
with tab7:
    st.header("🔍 Comprehensive Analysis & Recommendations")
    
    if any([df_perf is not None, df_shap is not None, df_lime is not None, df_overlap is not None]):
        
        # Summary statistics
        st.subheader("📊 Analysis Summary")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            models_count = len(df_perf) if df_perf is not None else 0
            st.metric("🤖 Models Analyzed", models_count)
        
        with col2:
            shap_features = len(df_shap) if df_shap is not None else 0
            st.metric("🎯 SHAP Features", shap_features)
        
        with col3:
            lime_features = len(df_lime) if df_lime is not None else 0
            st.metric("🍋 LIME Features", lime_features)
        
        with col4:
            overlap_models = len(df_overlap) if df_overlap is not None else 0
            st.metric("🔗 Overlap Analysis", overlap_models)
        
        # Detailed recommendations
        st.subheader("💡 Strategic Recommendations")
        
        recommendations = []
        
        if df_perf is not None:
            try:
                best_f1_model = df_perf['f1'].idxmax()
                best_acc_model = df_perf['accuracy'].idxmax()
                
                if best_f1_model == best_acc_model:
                    recommendations.append(f"🎯 **Primary Recommendation**: Deploy **{best_f1_model}** - shows best performance across metrics")
                else:
                    recommendations.append(f"⚖️ **Trade-off Decision**: **{best_f1_model}** (best F1) vs **{best_acc_model}** (best accuracy)")
            except:
                pass
        
        if df_overlap is not None:
            try:
                avg_overlap = df_overlap['overlap_percentage'].mean()
                if avg_overlap < 30:
                    recommendations.append("🔄 **Explainability Strategy**: Use both SHAP and LIME for comprehensive understanding")
                else:
                    recommendations.append("✅ **Explainability Strategy**: SHAP and LIME show good agreement - either method reliable")
            except:
                pass
        
        recommendations.extend([
            "🔍 **Feature Engineering**: Investigate top features identified by both SHAP and LIME",
            "📊 **Model Monitoring**: Track feature importance drift over time",
            "🎯 **Business Integration**: Translate feature insights into actionable business rules",
            "🔄 **Continuous Improvement**: Regular re-analysis with new data"
        ])
        
        for rec in recommendations:
            st.markdown(f'<div class="insight-box">{rec}</div>', unsafe_allow_html=True)
        
        # Export functionality
        st.subheader("💾 Export Analysis")
        
        if st.button("📊 Generate Comprehensive Report"):
            # Create summary report
            report_data = {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "models_analyzed": models_count,
                "shap_features": shap_features,
                "lime_features": lime_features,
                "recommendations": recommendations
            }
            
            st.success("📄 Report generated successfully!")
            st.json(report_data)
    
    else:
        st.info("📤 Upload analysis files to see detailed insights and recommendations.")

# =====================
# 📱 FOOTER
# =====================
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 2rem;'>
    🔍 <strong>Explainable AI Dashboard</strong> | 
    Built with Streamlit + Google Gemini | 
    🤖 AI-Powered Insights & Interactive Chatbot | 
    📊 Complete ML Explainability Pipeline
</div>
""", unsafe_allow_html=True)
