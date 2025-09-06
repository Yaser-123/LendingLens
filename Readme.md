# 🔍 LendingLens - Explainable AI Dashboard
## Interactive Streamlit Application for P2P Lending Risk Assessment

[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Python](https://img.shields.io/badge/Python-3.8+-blue?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![MIT License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)

**Live Dashboard for Explainable AI in Peer-to-Peer Lending Risk Assessment**

---

## 🚀 **Quick Start**

### **1. Launch the Dashboard**
```bash
# Navigate to Streamlit directory
cd Streamlit

# Install dependencies
pip install -r requirements.txt

# Run the dashboard
streamlit run app.py
```

### **2. Access the Application**
- **Local URL:** http://localhost:8501
- **Network URL:** Will be displayed in terminal for sharing

---

## 🎯 **Dashboard Features**

### **📊 Model Performance Analysis**
- **Interactive Performance Metrics** - Compare Accuracy, Precision, Recall, F1-scores
- **Cross-Validation Results** - 5-fold CV performance with confidence intervals
- **Model Comparison Charts** - Side-by-side performance visualization
- **Financial Risk Metrics** - Business-focused performance indicators

### **🔍 SHAP Explainability**
- **Global Feature Importance** - Dataset-wide feature impact analysis
- **Local Explanations** - Individual prediction breakdowns
- **Interactive Visualizations** - Waterfall plots, force plots, summary plots
- **Cross-Model Comparison** - SHAP values across different algorithms

### **🍋 LIME Analysis**
- **Local Interpretations** - Neighborhood-based explanations
- **Feature Perturbation** - Understanding prediction sensitivity
- **Confidence Intervals** - Uncertainty quantification in explanations
- **Interactive Exploration** - Dynamic local explanation generation

### **🔄 Feature Overlap Analysis**
- **SHAP vs LIME Comparison** - Method agreement visualization
- **Consensus Features** - Universally important risk factors
- **Complementarity Analysis** - Unique insights from each method
- **Visual Overlap Metrics** - Venn diagrams and overlap percentages

### **🤖 GenAI Interpretations**
- **Business-Friendly Explanations** - Technical to stakeholder translation
- **Risk Assessment Summaries** - Automated insight generation
- **Model Comparison Analysis** - AI-powered model evaluation
- **Regulatory Compliance** - Explanation documentation for audit trails

### **💬 AI-Powered Chatbot**
- **Interactive Q&A** - Ask questions about model results
- **Context-Aware Responses** - Answers based on your specific data
- **Technical Support** - Get help understanding complex concepts
- **Business Translation** - Convert technical insights to business language

---

## 🛠️ **Installation & Setup**

### **Prerequisites**
- **Python 3.8+** (Python 3.9+ recommended)
- **Google Gemini API Key** (for GenAI features)
- **8GB+ RAM** (for large dataset processing)

### **Environment Setup**

#### **1. Clone the Repository**
```bash
git clone https://github.com/Yaser-123/explainable-ai-p2p-lending.git
cd explainable-ai-p2p-lending/Streamlit
```

#### **2. Create Virtual Environment**
```bash
# Create virtual environment
python -m venv streamlit_env

# Activate environment
# Windows:
streamlit_env\Scripts\activate
# Linux/Mac:
source streamlit_env/bin/activate
```

#### **3. Install Dependencies**
```bash
pip install -r requirements.txt
```

#### **4. Environment Configuration**
Create a `.env` file in the parent directory:
```env
# Google Gemini API Configuration
GEMINI_API_KEY=your_gemini_api_key_here

# Dashboard Configuration
DASHBOARD_TITLE=LendingLens - Explainable AI Dashboard

# Optional: Custom model paths
MODEL_PATH=../models/
DATA_PATH=../results/pipeline_results/
```

### **5. Launch Dashboard**
```bash
streamlit run app.py
```

---

## 📁 **Required Data Files**

The dashboard expects the following files in `../results/pipeline_results/`:

### **Performance Data**
- **`model_performance.csv`** - Model accuracy, precision, recall, F1-scores
- **`../reports/ALL_cv_metrics.csv`** - Cross-validation results with std deviations

### **Explainability Data**
- **`shap_feature_importance.csv`** - SHAP values for feature importance
- **`lime_feature_importance.csv`** - LIME local explanations
- **`feature_overlap_analysis.csv`** - SHAP vs LIME overlap analysis

### **GenAI Explanations**
- **`../results/genai_explanations/*.txt`** - AI-generated model interpretations
- **Model-specific explanation files** for detailed insights

---

## 🎨 **Dashboard Interface**

### **Navigation Structure**

#### **📋 Main Dashboard**
- **Overview Cards** - Key metrics and performance summaries
- **Model Selection** - Choose between Logistic Regression, Random Forest, XGBoost
- **Quick Insights** - Highlight critical findings and recommendations

#### **📊 Performance Analysis**
- **Metrics Comparison** - Interactive charts comparing all models
- **Financial Perspective** - Business-focused performance evaluation
- **Cross-Validation** - Robust performance assessment with error bars

#### **🔍 Explainability Hub**
- **SHAP Analysis** - Global and local SHAP explanations
- **LIME Insights** - Local interpretable model explanations
- **Method Comparison** - Side-by-side SHAP vs LIME analysis

#### **🤖 AI Interpretations**
- **GenAI Summaries** - Business-friendly model explanations
- **Risk Assessments** - Automated risk factor analysis
- **Compliance Reports** - Regulatory-ready documentation

#### **💬 Interactive Chat**
- **AI Assistant** - Ask questions about your models and data
- **Context-Aware** - Responses based on your specific results
- **Help & Support** - Technical assistance and guidance

### **Interactive Features**

#### **Dynamic Filtering**
- **Model Selection** - Filter results by specific algorithms
- **Feature Focus** - Drill down into specific risk factors
- **Performance Metrics** - Customize chart displays

#### **Export Capabilities**
- **Chart Downloads** - Save visualizations as PNG/PDF
- **Data Export** - Download filtered datasets as CSV
- **Report Generation** - Create summary reports for stakeholders

#### **Real-time Updates**
- **Data Refresh** - Update with new model results
- **Interactive Plots** - Hover, zoom, and explore visualizations
- **Responsive Design** - Optimal viewing on desktop and tablet

---

## 🔧 **Configuration Options**

### **Dashboard Customization**

#### **Visual Themes**
```python
# Custom color schemes
COLORS = {
    'primary': '#1f77b4',
    'secondary': '#ff7f0e', 
    'success': '#2ca02c',
    'warning': '#d62728'
}

# Chart configurations
PLOT_CONFIG = {
    'height': 500,
    'template': 'plotly_white',
    'font_size': 12
}
```

#### **Data Sources**
```python
# Custom data paths
DATA_PATHS = {
    'performance': '../results/pipeline_results/model_performance.csv',
    'shap': '../results/pipeline_results/shap_feature_importance.csv',
    'lime': '../results/pipeline_results/lime_feature_importance.csv',
    'overlap': '../results/pipeline_results/feature_overlap_analysis.csv'
}
```

### **API Configuration**

#### **Google Gemini Setup**
1. **Get API Key:** Visit [Google AI Studio](https://makersuite.google.com/app/apikey)
2. **Create .env file:** Add your API key to environment configuration
3. **Configure Model:** Optionally customize the Gemini model version

#### **Rate Limiting**
```python
# API usage optimization
GEMINI_CONFIG = {
    'model': 'gemini-2.5-flash',
    'max_tokens': 1000,
    'temperature': 0.7,
    'timeout': 30
}
```

---

## 🚀 **Deployment Options**

### **Local Development**
```bash
# Basic local deployment
streamlit run app.py --server.port 8501

# Custom configuration
streamlit run app.py --server.port 8080 --server.address 0.0.0.0
```

### **Streamlit Cloud**
1. **Connect Repository** to Streamlit Cloud
2. **Set Environment Variables** in Streamlit dashboard
3. **Deploy Automatically** from GitHub

### **Docker Deployment**
```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.address", "0.0.0.0"]
```

### **Production Deployment**
- **Heroku:** Use `setup.sh` and `Procfile` for Heroku deployment
- **AWS/GCP:** Deploy using container services
- **Azure:** Use Azure Container Instances

---

## 📱 **Usage Examples**

### **Business Users**
1. **Open Dashboard** → Navigate to Model Performance
2. **Select Model** → Choose Random Forest (recommended)
3. **Review Metrics** → Focus on Precision and F1-score
4. **Check Explanations** → View SHAP global importance
5. **Ask AI** → "Why is Random Forest recommended for lending?"

### **Technical Users**
1. **Performance Analysis** → Compare cross-validation results
2. **SHAP Deep Dive** → Analyze local explanations
3. **LIME Validation** → Cross-check with SHAP findings
4. **Feature Engineering** → Identify top risk factors
5. **Model Debugging** → Use overlap analysis for validation

### **Regulatory Teams**
1. **Compliance Check** → Review GenAI explanations
2. **Audit Trail** → Export explanation reports
3. **Documentation** → Generate compliance summaries
4. **Transparency** → Use dual explainability methods

---

## 🔍 **Troubleshooting**

### **Common Issues**

#### **Dashboard Won't Start**
```bash
# Check Python version
python --version  # Should be 3.8+

# Verify dependencies
pip install -r requirements.txt --upgrade

# Check file paths
ls ../results/pipeline_results/
```

#### **Missing Data Files**
- **Ensure** all required CSV files are in correct directories
- **Run** the main pipeline first to generate results
- **Check** file permissions and accessibility

#### **API Errors**
- **Verify** Gemini API key in `.env` file
- **Check** API quota and rate limits
- **Test** internet connectivity

#### **Performance Issues**
- **Reduce** dataset size for large files
- **Limit** SHAP/LIME sample sizes
- **Close** other memory-intensive applications

### **Performance Optimization**

#### **Memory Management**
```python
# Optimize for large datasets
@st.cache_data
def load_data(file_path):
    return pd.read_csv(file_path)

# Limit sample sizes
SHAP_SAMPLES = 100
LIME_SAMPLES = 20
```

#### **Caching Strategy**
```python
# Cache expensive computations
@st.cache_data(ttl=3600)  # Cache for 1 hour
def compute_explanations(model_data):
    # Expensive SHAP/LIME calculations
    return results
```

---

## 🤝 **Contributing**

### **Development Setup**
1. **Fork** the repository
2. **Create** feature branch: `git checkout -b feature/new-dashboard-feature`
3. **Install** development dependencies: `pip install -r requirements-dev.txt`
4. **Make** changes and test thoroughly
5. **Submit** pull request with clear description

### **Code Standards**
- **Follow** PEP 8 style guidelines
- **Add** docstrings for new functions
- **Include** error handling for user inputs
- **Test** with different data configurations

---

## 📊 **Technical Specifications**

### **System Requirements**
- **Python:** 3.8+ (3.9+ recommended)
- **RAM:** 8GB minimum, 16GB recommended
- **Storage:** 2GB for dependencies and cache
- **Network:** Internet connection for GenAI features

### **Dependencies**
```
streamlit>=1.28.0    # Dashboard framework
pandas>=1.5.0        # Data manipulation
numpy>=1.21.0        # Numerical computing
matplotlib>=3.5.0    # Static plotting
seaborn>=0.11.0      # Statistical visualization
plotly>=5.10.0       # Interactive plotting
google-generativeai>=0.3.0  # GenAI integration
scikit-learn>=1.1.0  # ML utilities
python-dotenv>=1.0.0 # Environment management
```

### **Architecture**
- **Frontend:** Streamlit web interface
- **Backend:** Python data processing
- **AI Integration:** Google Gemini API
- **Data Storage:** CSV files and cached results
- **Visualization:** Plotly interactive charts

---

## 📄 **License & Support**

### **License**
This project is licensed under the MIT License - see the [LICENSE](../LICENSE) file for details.

### **Support**
- **Issues:** Report bugs via [GitHub Issues](https://github.com/Yaser-123/explainable-ai-p2p-lending/issues)
- **Documentation:** Check the main [README](../README.md) for project overview
- **Discussions:** Join [GitHub Discussions](https://github.com/Yaser-123/explainable-ai-p2p-lending/discussions)

### **Citation**
If you use this dashboard in your research, please cite:
```bibtex
@software{explainable_ai_p2p_lending,
  title={LendingLens: Explainable AI Dashboard for P2P Lending Risk Assessment},
  author={Research Team},
  year={2025},
  url={https://github.com/Yaser-123/explainable-ai-p2p-lending}
}
```

---

## 🎯 **Roadmap**

### **Current Version (v1.0)**
- ✅ Model performance comparison
- ✅ SHAP/LIME explainability
- ✅ GenAI interpretations
- ✅ Interactive chatbot
- ✅ Export capabilities

### **Upcoming Features (v1.1)**
- 🔄 Real-time model retraining
- 📱 Mobile-responsive design
- 🔐 User authentication
- 📊 Advanced visualizations
- 🌐 Multi-language support

### **Future Enhancements (v2.0)**
- 🤖 Advanced AI explanations
- 🔗 API endpoints for integration
- 📈 Time-series analysis
- 🎛️ Model parameter tuning
- 🔄 Automated reporting

---

**🔍 LendingLens - Making AI Decisions Transparent and Trustworthy**

*Built with ❤️ using Streamlit • Powered by Explainable AI • Designed for Financial Excellence*
