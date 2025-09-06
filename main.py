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
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        # Use a model that is likely to be available, e.g., gemini-pro
        gemini_model = genai.GenerativeModel('gemini-pro')
    except Exception as e:
        gemini_model = None
        st.error(f"🔴 Failed to configure Gemini API: {e}")
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
            except Exception:
                # Suppress the warning message for missing files
                pass
        
        # Load model comparison
        try:
            with open(f"{genai_path}model_comparison_analysis.txt", 'r', encoding='utf-8') as f:
                explanations['model_comparison'] = f.read()
        except Exception:
            # Suppress the warning message for missing file
            pass
        
        # Load JSON data
        try:
            with open(f"{genai_path}genai_explanations.json", 'r', encoding='utf-8') as f:
                explanations['json_data'] = json.load(f)
        except Exception:
            # Suppress the warning message for missing file
            pass
        
        return explanations
        
    except Exception as e:
        # Keep the main error message if the entire process fails
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
            "best_model": df_perf['f1'].idxmax() if df_perf is not None and 'f1' in df_perf.columns and not df_perf.empty else "Unknown",
            "best_f1": df_perf['f1'].max() if df_perf is not None and 'f1' in df_perf.columns and not df_perf.empty else 0
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
        try:
            plot_specs = json.loads(response.text.strip())
            return plot_specs
        except json.JSONDecodeError:
            # If JSON parsing fails, try to extract JSON from the response
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
            df_plot = df_perf.reset_index().melt(id_vars='index', var_name='Metric', value_name='Score')
            fig = px.bar(
                df_plot,
                x='index',
                y='Score',
                color='Metric',
                title=title,
                labels={'index': 'Models'},
                barmode='group'
            )
    elif chart_type == "line":
        df_plot = df_perf.reset_index().melt(id_vars='index', var_name='Metric', value_name='Score')
        fig = px.line(
            df_plot,
            x='index',
            y='Score',
            color='Metric',
            title=title,
            labels={'index': 'Models'}
        )
    else:
        # Default to bar
        df_plot = df_perf.reset_index().melt(id_vars='index', var_name='Metric', value_name='Score')
        fig = px.bar(
            df_plot,
            x='index',
            y='Score',
            color='Metric',
            title=title,
            labels={'index': 'Models'},
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
st.sidebar.markdown("Upload your CSV files from the ML pipeline")

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

# Load data
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
    
    if df_perf is not None and not df_perf.empty:
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
        if 'f1' in df_perf.columns and not df_perf.empty:
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
        st.warning("📤 Upload model_performance.csv to see results.")
        st.info("💡 Expected format: CSV with columns like 'accuracy', 'f1', 'precision', 'recall'")

# =====================
# 🎯 TAB 2: SHAP ANALYSIS
# =====================
with tab2:
    st.header("🎯 SHAP Feature Importance Analysis")
    
    if df_shap is not None and not df_shap.empty:
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
                            st.info("No non-zero SHAP values to plot for this model.")
                    except Exception as e:
                        st.error(f"Error generating SHAP plot: {e}")
                
                with col2:
                    st.subheader("Raw SHAP Data")
                    st.dataframe(df_shap[selected_model].sort_values(ascending=False).round(5))
            else:
                st.warning(f"Column '{selected_model}' not found in SHAP data.")
    else:
        st.warning("📤 Upload shap_importance.csv to see SHAP analysis.")
        st.info("💡 Expected format: CSV with features as the first column and models as subsequent columns.")

# =====================
# 🍋 TAB 3: LIME ANALYSIS
# =====================
with tab3:
    st.header("🍋 LIME Feature Importance Analysis")
    
    if df_lime is not None and not df_lime.empty:
        available_models = [col for col in df_lime.columns if col not in ['Unnamed: 0', 'index']]
        
        if not available_models:
            st.warning("No model columns found in LIME data")
            st.dataframe(df_lime.head())
        else:
            selected_model = st.selectbox("Select Model for LIME Analysis", available_models)
            
            if selected_model in df_lime.columns:
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    try:
                        lime_values = df_lime[selected_model]
                        numeric_lime = pd.to_numeric(lime_values, errors='coerce')
                        numeric_lime = numeric_lime.dropna()
                        numeric_lime = numeric_lime[numeric_lime != 0]
                        
                        if len(numeric_lime) > 0:
                            top_lime = numeric_lime.nlargest(15)
                            
                            fig = px.bar(
                                x=top_lime.values,
                                y=top_lime.index,
                                orientation='h',
                                title=f"🍋 Top 15 LIME Features - {selected_model}",
                                labels={'x': 'LIME Weight', 'y': 'Feature Rules'},
                                color=top_lime.values,
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
                            st.info("No non-zero LIME values to plot for this model.")
                    except Exception as e:
                        st.error(f"Error generating LIME plot: {e}")
                
                with col2:
                    st.subheader("Raw LIME Data")
                    st.dataframe(df_lime[selected_model].sort_values(ascending=False).round(5))
            else:
                st.warning(f"Column '{selected_model}' not found in LIME data.")
    else:
        st.warning("📤 Upload lime_importance.csv to see LIME analysis.")
        st.info("💡 Expected format: CSV with feature rules as the first column and models as subsequent columns.")

# =====================
# 🔗 TAB 4: FEATURE OVERLAP
# =====================
with tab4:
    st.header("🔗 SHAP vs. LIME Feature Overlap Analysis")
    
    if df_overlap is not None and not df_overlap.empty:
        st.markdown("### 📊 Overlap Percentage by Model")
        st.dataframe(df_overlap.round(2))
        
        try:
            fig = px.bar(df_overlap, 
                         x=df_overlap.index, 
                         y='overlap_percentage', 
                         title='SHAP vs. LIME Overlap Percentage by Model',
                         labels={'overlap_percentage': 'Overlap %', 'index': 'Model'},
                         color='overlap_percentage',
                         color_continuous_scale='Inferno')
            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font_color='white',
                title_font_color='white'
            )
            st.plotly_chart(fig, use_container_width=True)
            
            st.subheader("📋 Overlap Insights")
            max_overlap_model = df_overlap['overlap_percentage'].idxmax()
            min_overlap_model = df_overlap['overlap_percentage'].idxmin()
            
            st.markdown(f"**Highest Overlap**: {max_overlap_model} with {df_overlap.loc[max_overlap_model, 'overlap_percentage']:.2f}%")
            st.markdown(f"**Lowest Overlap**: {min_overlap_model} with {df_overlap.loc[min_overlap_model, 'overlap_percentage']:.2f}%")
            st.info("A high overlap suggests that the model's global and local explanations are consistent, indicating a more stable and reliable model.")
        except Exception as e:
            st.error(f"Error generating overlap plot: {e}")
    else:
        st.warning("📤 Upload feature_overlap.csv to see the overlap analysis.")
        st.info("💡 Expected format: CSV with models as rows and a column 'overlap_percentage'.")

# =====================
# 🤖 TAB 5: GENAI EXPLANATIONS
# =====================
with tab5:
    st.header("🤖 GenAI Explanations")
    
    if genai_explanations:
        st.markdown("These insights were generated automatically by a GenAI model to provide high-level, human-readable explanations of the pipeline results.")
        st.markdown("---")
        
        # Display model comparison
        if 'model_comparison' in genai_explanations:
            st.subheader("Model Comparison Analysis")
            st.info(genai_explanations['model_comparison'])
            st.markdown("---")
        
        # Display individual model explanations
        st.subheader("Individual Model Explanations")
        model_exp_tabs = st.tabs([m.replace('_', ' ').title() for m in ['random_forest', 'xgboost', 'logistic_regression'] if m in genai_explanations])
        for i, model in enumerate(['random_forest', 'xgboost', 'logistic_regression']):
            if model in genai_explanations:
                with model_exp_tabs[i]:
                    st.markdown(f"#### Explanation for {model.replace('_', ' ').title()}")
                    st.info(genai_explanations[model])
                    
    else:
        st.warning("GenAI explanations not found. Please check your `results/genai_explanations/` folder.")

# =====================
# 💬 TAB 6: AI CHATBOT
# =====================
with tab6:
    st.header("💬 AI Explainability Chatbot")
    
    st.info("Ask questions about the model performance, SHAP/LIME features, or overall insights. For example: 'Which features are most important for the XGBoost model?'")
    
    # Display chat messages from history
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            
    # React to user input
    if prompt := st.chat_input("Ask a question about the dashboard..."):
        # Display user message
        st.chat_message("user").markdown(prompt)
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        
        # Generate and display AI response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = ai_chatbot_response(prompt, st.session_state.data_context)
                st.markdown(response)
                st.session_state.chat_history.append({"role": "assistant", "content": response})

# =====================
# 🔍 TAB 7: DETAILED INSIGHTS
# =====================
with tab7:
    st.header("🔍 Dynamic Plot Generation Assistant")
    
    st.info("Describe the plot you want to see, and the AI will try to generate it. For example: 'Show me the F1 scores for all models' or 'Compare accuracy and precision for Random Forest and Logistic Regression'")
    
    # Text input for dynamic plot request
    plot_request = st.text_input("Enter your plot request:", key="plot_request")
    
    if plot_request:
        with st.spinner("Generating plot based on your request..."):
            dynamic_fig = generate_ai_dynamic_plot(plot_request, st.session_state.data_context)
            if dynamic_fig:
                st.plotly_chart(dynamic_fig, use_container_width=True)
            else:
                st.error("Failed to generate plot. Please check the data and your request.")
