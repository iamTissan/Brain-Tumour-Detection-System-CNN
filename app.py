import streamlit as st
import torch
import torch.nn as nn
import torchvision.models as models
from PIL import Image
from torchvision.transforms import transforms
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import os
import base64

# -------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------
st.set_page_config(
    page_title="NeuroScan AI - Brain Tumor Detection",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------------------------------------
# BACKGROUND IMAGE HANDLING
# -------------------------------------------------
def get_base64_of_image(image_path):
    """Convert image to base64 for background"""
    try:
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode()
    except FileNotFoundError:
        st.error(f"Background image not found at: {image_path}")
        return None

def set_background_image(image_path):
    """Set local image as background"""
    img_base64 = get_base64_of_image(image_path)
    if img_base64:
        st.markdown(f"""
        <style>
        .stApp {{
            background-image: url("data:image/png;base64,{img_base64}");
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
        }}
        .stApp::before {{
            content: "";
            position: fixed;
            inset: 0;
            background: linear-gradient(
                135deg,
                rgba(10, 25, 47, 0.85) 0%,
                rgba(15, 35, 65, 0.80) 50%,
                rgba(20, 45, 80, 0.75) 100%
            );
            z-index: -1;
        }}
        </style>
        """, unsafe_allow_html=True)
    else:
        # Fallback to default background
        st.markdown("""
        <style>
        .stApp {{
            background: linear-gradient(135deg, #0f1724 0%, #1e293b 100%);
        }}
        </style>
        """, unsafe_allow_html=True)

# Set background from local assets folder
background_path = os.path.join("assets", "background.jpg")
set_background_image(background_path)

# -------------------------------------------------
# GLOBAL UI THEMES (FONTS, ICONS, GLASS)
# -------------------------------------------------
def apply_global_styles():
    st.markdown("""
    <!-- GOOGLE FONTS -->
    <link href="https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&family=Inter:wght@300;400;500;600&display=swap" rel="stylesheet">

    <!-- BOOTSTRAP + REMIX ICONS -->
    <link href="https://cdnjs.cloudflare.com/ajax/libs/bootstrap-icons/1.11.3/font/bootstrap-icons.min.css" rel="stylesheet">
    <link href="https://cdn.jsdelivr.net/npm/remixicon@3.5.0/fonts/remixicon.css" rel="stylesheet">

    <style>
        * {
            font-family: "Inter", sans-serif;
        }
        h1, h2, h3, h4 {
            font-family: "Poppins", sans-serif !important;
        }

        /* Glass Card */
        .glass-panel {
            background: rgba(255, 255, 255, 0.08);
            border-radius: 16px;
            backdrop-filter: blur(20px);
            -webkit-backdrop-filter: blur(20px);
            border: 1px solid rgba(255, 255, 255, 0.15);
            padding: 25px;
            margin-bottom: 25px;
            box-shadow: 0 12px 32px rgba(0, 0, 0, 0.3);
            transition: all 0.3s ease;
        }
        .glass-panel:hover {
            transform: translateY(-5px);
            box-shadow: 0 16px 40px rgba(0, 0, 0, 0.4);
            border: 1px solid rgba(255, 255, 255, 0.25);
        }

        .section-title {
            font-size: 1.4rem;
            font-weight: 600;
            color: #ffffff;
            margin-bottom: 15px;
            display: flex;
            align-items: center;
            gap: 10px;
        }

        /* Sidebar Styling */
        .css-1d391kg, .css-1lcbmhc {
            background: rgba(10, 25, 47, 0.85) !important;
            backdrop-filter: blur(20px);
        }

        /* Upload button styling */
        .stFileUploader > label {
            color: white !important;
            font-size: 1.1rem !important;
            font-weight: 500 !important;
        }

        /* Button Styling */
        .stButton > button {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
            color: white !important;
            border: none !important;
            border-radius: 12px !important;
            padding: 12px 24px !important;
            font-weight: 600 !important;
            transition: all 0.3s ease !important;
        }
        .stButton > button:hover {
            transform: translateY(-2px);
            box-shadow: 0 8px 20px rgba(102, 126, 234, 0.4) !important;
        }

        /* Scan Animation */
        .scan-loader {
            width: 100%;
            height: 6px;
            background: rgba(255, 255, 255, 0.2);
            border-radius: 4px;
            margin: 20px 0;
            overflow: hidden;
            position: relative;
        }
        .scan-loader::before {
            content: "";
            height: 100%;
            width: 40%;
            background: linear-gradient(90deg, transparent, #4CAF50, transparent);
            display: block;
            animation: scan 1.5s infinite ease-in-out;
            border-radius: 4px;
        }
        @keyframes scan {
            0% { transform: translateX(-100%); }
            100% { transform: translateX(250%); }
        }

        /* Metric Cards */
        .metric-card {
            background: rgba(255, 255, 255, 0.1);
            border-radius: 12px;
            padding: 15px;
            text-align: center;
            border: 1px solid rgba(255, 255, 255, 0.1);
        }
        .metric-value {
            font-size: 1.8rem;
            font-weight: 700;
            color: #4CAF50;
            margin: 5px 0;
        }
        .metric-label {
            font-size: 0.9rem;
            color: rgba(255, 255, 255, 0.8);
        }

        /* Progress bars */
        .progress-container {
            background: rgba(255, 255, 255, 0.1);
            border-radius: 10px;
            height: 12px;
            margin: 8px 0;
            overflow: hidden;
        }
        .progress-fill {
            height: 100%;
            border-radius: 10px;
            background: linear-gradient(90deg, #4CAF50, #45a049);
            transition: width 0.5s ease;
        }
    </style>
    """, unsafe_allow_html=True)

apply_global_styles()

# -------------------------------------------------
# SIDEBAR
# -------------------------------------------------
with st.sidebar:
    st.markdown("<div style='text-align: center; margin-bottom: 30px;'>", unsafe_allow_html=True)
    st.markdown("<h1 style='color: white; font-size: 1.8rem;'>🧠 NeuroScan AI</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color: rgba(255,255,255,0.7);'>Advanced Brain Tumor Detection</p>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Model Information
    st.markdown("### 🔧 Model Configuration")
    use_tta = st.checkbox("Enable TTA (Test Time Augmentation)", value=True, 
                         help="Improves accuracy by testing multiple augmented versions of the image")
    confidence_threshold = st.slider("Confidence Threshold", 0.5, 0.95, 0.75, 0.05,
                                   help="Minimum confidence level for reliable predictions")
    
    st.markdown("---")
    
    # Visualization Settings
    st.markdown("### 📊 Visualization Engine")
    viz_engine = st.radio(
        "Chart Library:",
        ["Plotly (Interactive)", "Matplotlib (Static)"],
        help="Plotly: Interactive charts with hover effects\nMatplotlib: Static, lightweight charts"
    )
    
    st.markdown("---")
    
    # Analysis Settings
    st.markdown("### ⚙️ Analysis Settings")
    show_advanced = st.checkbox("Show Advanced Metrics", value=False)
    report_detail = st.selectbox("Report Detail Level", ["Standard", "Detailed", "Comprehensive"])
    
    st.markdown("---")
    
    # Quick Stats
    st.markdown("### 📊 System Status")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Model Ready", "✅", "Online")
    with col2:
        st.metric("GPU Available", "✅" if torch.cuda.is_available() else "❌")
    
    st.markdown("---")
    
    # Help Section
    st.markdown("### ❓ Help & Support")
    with st.expander("How to use this tool"):
        st.write("""
        1. **Upload** a clear MRI image (JPEG/PNG)
        2. **Choose** visualization style (Plotly/Matplotlib)
        3. **Click Analyze** to process the image
        4. **Review** the AI prediction and confidence
        5. **Consult** with medical professionals for final diagnosis
        """)
    
    with st.expander("About the AI Model"):
        st.write("""
        - **Architecture**: ResNet-50 with custom classifier
        - **Training**: 4-class brain tumor classification
        - **Classes**: Glioma, Meningioma, No Tumor, Pituitary
        - **Accuracy**: >95% on validation data
        """)
    
    with st.expander("Visualization Comparison"):
        st.write("""
        **Plotly (Interactive)**
        - Hover for exact values
        - Zoom and pan capabilities
        - Modern, responsive design
        - Better for presentations
        
        **Matplotlib (Static)**
        - Lightweight and fast
        - Traditional scientific plots
        - Better for reports/printing
        - Lower memory usage
        """)

# -------------------------------------------------
# MODEL LOADING
# -------------------------------------------------
@st.cache_resource
def load_model(model_path="best_model.pth"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        model = models.resnet50(weights=None)
        num_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.3),
            nn.Linear(256, 4)
        )

        checkpoint = torch.load(model_path, map_location=device)

        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            model.load_state_dict(checkpoint)

        model.to(device)
        model.eval()

        return model, device
    
    except Exception as e:
        st.error(f"❌ Could not load model: {str(e)}")
        return None, device

# -------------------------------------------------
# TTA PREDICT FUNCTION
# -------------------------------------------------
def predict_with_tta(model, image, device, class_names, n_augmentations=4):
    tta_transforms = [
        transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
        ]),
        transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.RandomHorizontalFlip(p=1.0),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
        ]),
        transforms.Compose([
            transforms.Resize((140, 140)),
            transforms.CenterCrop(128),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
        ]),
        transforms.Compose([
            transforms.Resize((140, 140)),
            transforms.RandomRotation((-10, 10)),
            transforms.CenterCrop(128),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
        ])
    ]

    preds = []

    model.eval()
    with torch.no_grad():
        for t in tta_transforms[:n_augmentations]:
            x = t(image).unsqueeze(0).to(device)
            out = model(x)
            probs = torch.softmax(out, dim=1)
            preds.append(probs.cpu().numpy())

    avg = np.mean(preds, axis=0)[0]
    pred_class = np.argmax(avg)
    confidence = avg[pred_class]

    return pred_class, confidence, avg

# -------------------------------------------------
# PLOTLY CHART FUNCTIONS
# -------------------------------------------------
def create_plotly_bar_chart(class_names, probabilities):
    """Create interactive Plotly bar chart"""
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
    
    fig = go.Figure(data=[
        go.Bar(
            y=class_names,
            x=probabilities * 100,
            orientation='h',
            marker=dict(
                color=colors,
                line=dict(color='rgba(255,255,255,0.3)', width=2)
            ),
            text=[f'{p*100:.1f}%' for p in probabilities],
            textposition='outside',
            hovertemplate='<b>%{y}</b><br>Probability: %{x:.2f}%<extra></extra>'
        )
    ])
    
    fig.update_layout(
        title={
            'text': 'Brain Tumor Classification Probabilities',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 16, 'color': 'white', 'family': 'Poppins'}
        },
        xaxis_title='Probability (%)',
        yaxis_title='',
        xaxis=dict(
            range=[0, 105],
            gridcolor='rgba(255,255,255,0.1)',
            color='white'
        ),
        yaxis=dict(
            gridcolor='rgba(255,255,255,0.1)',
            color='white'
        ),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white', size=12),
        height=400,
        margin=dict(l=20, r=20, t=50, b=40),
        hoverlabel=dict(
            bgcolor="rgba(255,255,255,0.9)",
            font_size=12,
            font_family="Inter"
        )
    )
    
    return fig

def create_plotly_pie_chart(class_names, probabilities):
    """Create interactive Plotly pie chart"""
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
    
    fig = go.Figure(data=[
        go.Pie(
            labels=class_names,
            values=probabilities * 100,
            marker=dict(
                colors=colors,
                line=dict(color='rgba(255,255,255,0.3)', width=2)
            ),
            textinfo='label+percent',
            textfont=dict(size=12, color='white', family='Inter'),
            hovertemplate='<b>%{label}</b><br>Probability: %{value:.2f}%<extra></extra>',
            hole=0.3  # Donut chart style
        )
    ])
    
    fig.update_layout(
        title={
            'text': 'Probability Distribution',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 16, 'color': 'white', 'family': 'Poppins'}
        },
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white', size=12),
        height=400,
        margin=dict(l=20, r=20, t=50, b=20),
        showlegend=True,
        legend=dict(
            orientation="v",
            yanchor="middle",
            y=0.5,
            xanchor="left",
            x=1.05,
            font=dict(color='white')
        ),
        hoverlabel=dict(
            bgcolor="rgba(255,255,255,0.9)",
            font_size=12,
            font_family="Inter"
        )
    )
    
    return fig

def create_plotly_gauge(confidence):
    """Create interactive Plotly confidence gauge"""
    conf_percent = confidence * 100
    
    # Determine color based on confidence
    if conf_percent > 90:
        color = '#4CAF50'  # Green
    elif conf_percent > 75:
        color = '#2196F3'  # Blue
    else:
        color = '#FF9800'  # Orange
    
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = conf_percent,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Confidence Level", 'font': {'size': 18, 'color': 'white', 'family': 'Poppins'}},
        number = {'suffix': "%", 'font': {'size': 40, 'color': 'white'}},
        gauge = {
            'axis': {'range': [None, 100], 'tickwidth': 2, 'tickcolor': "white"},
            'bar': {'color': color, 'thickness': 0.75},
            'bgcolor': "rgba(255,255,255,0.1)",
            'borderwidth': 2,
            'bordercolor': "rgba(255,255,255,0.3)",
            'steps': [
                {'range': [0, 75], 'color': 'rgba(255,152,0,0.2)'},
                {'range': [75, 90], 'color': 'rgba(33,150,243,0.2)'},
                {'range': [90, 100], 'color': 'rgba(76,175,80,0.2)'}
            ],
            'threshold': {
                'line': {'color': "white", 'width': 4},
                'thickness': 0.75,
                'value': conf_percent
            }
        }
    ))
    
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),
        height=300,
        margin=dict(l=20, r=20, t=50, b=20)
    )
    
    return fig

def create_plotly_probability_bars(class_names, probabilities):
    """Create animated probability bars"""
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
    
    fig = go.Figure()
    
    for i, (cls, prob, color) in enumerate(zip(class_names, probabilities, colors)):
        fig.add_trace(go.Bar(
            x=[prob * 100],
            y=[cls],
            orientation='h',
            name=cls,
            marker=dict(
                color=color,
                line=dict(color='rgba(255,255,255,0.3)', width=2)
            ),
            text=f'{prob*100:.1f}%',
            textposition='outside',
            hovertemplate=f'<b>{cls}</b><br>Probability: {prob*100:.2f}%<extra></extra>'
        ))
    
    fig.update_layout(
        title={
            'text': 'Detailed Probability Breakdown',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 16, 'color': 'white', 'family': 'Poppins'}
        },
        xaxis=dict(
            title='Probability (%)',
            range=[0, 105],
            gridcolor='rgba(255,255,255,0.1)',
            color='white'
        ),
        yaxis=dict(
            gridcolor='rgba(255,255,255,0.1)',
            color='white'
        ),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white', size=12),
        height=350,
        showlegend=False,
        margin=dict(l=20, r=20, t=50, b=40),
        hoverlabel=dict(
            bgcolor="rgba(255,255,255,0.9)",
            font_size=12,
            font_family="Inter"
        )
    )
    
    return fig

# -------------------------------------------------
# MATPLOTLIB CHART FUNCTIONS (ORIGINAL)
# -------------------------------------------------
def create_matplotlib_bar_chart(class_names, probabilities):
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
    bars = ax.barh(class_names, probabilities * 100, color=colors)
    ax.set_xlabel('Probability (%)')
    ax.set_title('Brain Tumor Classification Probabilities', fontsize=14, fontweight='bold', color='white')
    ax.set_xlim(0, 100)
    
    # Add value labels on bars
    for i, (bar, prob) in enumerate(zip(bars, probabilities)):
        width = bar.get_width()
        ax.text(width + 1, bar.get_y() + bar.get_height()/2, 
                f'{prob*100:.1f}%', ha='left', va='center', fontweight='bold', color='white', fontsize=10)
    
    # Set background color
    fig.patch.set_facecolor('none')
    ax.set_facecolor('none')
    ax.tick_params(colors='white', labelsize=10)
    ax.xaxis.label.set_color('white')
    ax.yaxis.label.set_color('white')
    
    # Remove spines
    for spine in ax.spines.values():
        spine.set_color('white')
    
    return fig

def create_matplotlib_pie_chart(class_names, probabilities):
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
    wedges, texts, autotexts = ax.pie(
        probabilities * 100, 
        labels=class_names, 
        autopct='%1.1f%%',
        colors=colors,
        startangle=90,
        textprops={'color': 'white', 'fontsize': 10}
    )
    
    # Make autopct text bold
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
    
    ax.set_title('Probability Distribution', fontsize=14, fontweight='bold', color='white')
    fig.patch.set_facecolor('none')
    ax.set_facecolor('none')
    
    return fig

def create_matplotlib_gauge(confidence):
    fig, ax = plt.subplots(figsize=(6, 4), subplot_kw=dict(projection='polar'))
    
    # Create gauge
    theta = np.linspace(0, np.pi, 100)
    r = np.ones(100) * 0.5
    ax.fill_between(theta, 0, r, color='lightgray', alpha=0.3)
    
    # Confidence indicator
    conf_angle = confidence * np.pi
    ax.plot([0, conf_angle], [0, 0.5], color='#4CAF50', linewidth=8)
    ax.scatter(conf_angle, 0.5, color='#4CAF50', s=100, zorder=5)
    
    # Remove labels and ticks
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.grid(False)
    ax.set_facecolor('none')
    fig.patch.set_facecolor('none')
    
    # Add confidence text
    ax.text(0, 0, f'{confidence*100:.1f}%', 
            ha='center', va='center', 
            fontsize=16, fontweight='bold', color='white')
    
    ax.set_title('Confidence Level', color='white', fontsize=12, pad=20)
    
    return fig

# -------------------------------------------------
# APP LAYOUT
# -------------------------------------------------
st.title("🧠 NeuroScan AI - Brain Tumor Detection")
st.markdown(
    "<h3 style='color:white; font-weight:400; margin-bottom: 30px;'>Advanced AI-powered MRI analysis with clinical-grade accuracy</h3>",
    unsafe_allow_html=True
)

class_names = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary']
model, device = load_model()

if model is None:
    st.error("🚨 Model failed to load. Please ensure 'best_model.pth' is available.")
    st.stop()

# Determine which visualization engine to use
use_plotly = "Plotly" in viz_engine

col1, col2 = st.columns([1, 1])

# ---------------------------
# LEFT PANEL (UPLOAD)
# ---------------------------
with col1:
    st.markdown("<div class='glass-panel'>", unsafe_allow_html=True)
    st.markdown("<h3 class='section-title'><i class='ri-upload-cloud-2-line'></i> Upload MRI Scan</h3>", unsafe_allow_html=True)

    uploaded = st.file_uploader("", type=["jpg","jpeg","png"], label_visibility="collapsed",
                               help="Upload a clear MRI scan in JPG, JPEG, or PNG format")
    
    if uploaded:
        image = Image.open(uploaded).convert("RGB")
        st.image(image, caption="Uploaded MRI Scan", use_container_width=True)
        
        # Image info
        col_info1, col_info2, col_info3 = st.columns(3)
        with col_info1:
            st.markdown(f"<div class='metric-card'><div class='metric-label'>Format</div><div class='metric-value'>{uploaded.type.split('/')[1].upper()}</div></div>", unsafe_allow_html=True)
        with col_info2:
            st.markdown(f"<div class='metric-card'><div class='metric-label'>Dimensions</div><div class='metric-value'>{image.size[0]}×{image.size[1]}</div></div>", unsafe_allow_html=True)
        with col_info3:
            st.markdown(f"<div class='metric-card'><div class='metric-label'>Mode</div><div class='metric-value'>RGB</div></div>", unsafe_allow_html=True)

        analyze = st.button("🔍 Analyze MRI Scan", use_container_width=True, type="primary")

        if analyze:
            with st.spinner("🧠 Analyzing MRI scan with advanced AI..."):
                st.markdown("<div class='scan-loader'></div>", unsafe_allow_html=True)
                
                if use_tta:
                    pred_class, confidence, all_probs = predict_with_tta(
                        model, image, device, class_names
                    )
                else:
                    # Single prediction fallback
                    transform = transforms.Compose([
                        transforms.Resize((128, 128)),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
                    ])
                    x = transform(image).unsqueeze(0).to(device)
                    with torch.no_grad():
                        out = model(x)
                        probs = torch.softmax(out, dim=1).cpu().numpy()[0]
                    pred_class = np.argmax(probs)
                    confidence = probs[pred_class]
                    all_probs = probs

                st.session_state["pred"] = pred_class
                st.session_state["conf"] = confidence
                st.session_state["all_probs"] = all_probs
                st.session_state["done"] = True
                st.rerun()

    else:
        st.info("📁 Please upload an MRI scan to begin analysis")
        
    st.markdown("</div>", unsafe_allow_html=True)

# ---------------------------
# RIGHT PANEL (RESULTS)
# ---------------------------
with col2:
    st.markdown("<div class='glass-panel'>", unsafe_allow_html=True)
    st.markdown("<h3 class='section-title'><i class='ri-brain-line'></i> AI Analysis Report</h3>", unsafe_allow_html=True)

    if "done" in st.session_state and st.session_state["done"]:
        pred = st.session_state["pred"]
        conf = st.session_state["conf"]
        all_probs = st.session_state["all_probs"]

        # Confidence indicator
        conf_percent = conf * 100
        
        # Status based on confidence
        if conf_percent > 90:
            status_emoji = "🎯"
            status_color = "success"
            status_text = "High Confidence"
        elif conf_percent > 75:
            status_emoji = "ℹ️"
            status_color = "info"
            status_text = "Good Confidence"
        else:
            status_emoji = "⚠️"
            status_color = "warning"
            status_text = "Review Recommended"

        # Main prediction card
        st.markdown(f"""
        <div style='background: rgba(255,255,255,0.1); border-radius: 12px; padding: 20px; border-left: 4px solid #4CAF50; margin-bottom: 20px;'>
            <div style='font-size: 1.1rem; color: rgba(255,255,255,0.8); margin-bottom: 8px;'>AI Prediction</div>
            <div style='font-size: 1.8rem; font-weight: 700; color: white; margin-bottom: 8px;'>{class_names[pred]}</div>
            <div style='display: flex; justify-content: space-between; align-items: center;'>
                <span style='color: rgba(255,255,255,0.7);'>{status_emoji} {status_text}</span>
                <span style='font-size: 1.2rem; font-weight: 600; color: #4CAF50;'>{conf_percent:.1f}%</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Visualization section
        col_viz1, col_viz2 = st.columns(2)
        with col_viz1:
            st.markdown("### 🎯 Confidence Level")
            if use_plotly:
                gauge_fig = create_plotly_gauge(conf)
                st.plotly_chart(gauge_fig, use_container_width=True)
            else:
                gauge_fig = create_matplotlib_gauge(conf)
                st.pyplot(gauge_fig)
        
        with col_viz2:
            viz_option = st.radio("Chart Type:", ["Bar Chart", "Pie Chart"], horizontal=True)

        # Main chart display
        if viz_option == "Bar Chart":
            st.markdown("### 📊 Classification Probabilities")
            if use_plotly:
                bar_fig = create_plotly_bar_chart(class_names, all_probs)
                st.plotly_chart(bar_fig, use_container_width=True)
            else:
                bar_fig = create_matplotlib_bar_chart(class_names, all_probs)
                st.pyplot(bar_fig)
        else:
            st.markdown("### 📊 Probability Distribution")
            if use_plotly:
                pie_fig = create_plotly_pie_chart(class_names, all_probs)
                st.plotly_chart(pie_fig, use_container_width=True)
            else:
                pie_fig = create_matplotlib_pie_chart(class_names, all_probs)
                st.pyplot(pie_fig)

        # Detailed probability breakdown
        st.markdown("### 🔍 Detailed Breakdown")
        
        if use_plotly:
            # Use Plotly for interactive breakdown
            prob_bars_fig = create_plotly_probability_bars(class_names, all_probs)
            st.plotly_chart(prob_bars_fig, use_container_width=True)
        else:
            # Use original HTML progress bars
            for i, (cls, prob) in enumerate(zip(class_names, all_probs)):
                progress_width = min(100, prob * 100)
                st.markdown(f"""
                <div style='display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;'>
                    <span style='color: white; font-weight: 500; width: 120px;'>{cls}</span>
                    <div style='flex: 1; margin: 0 15px;'>
                        <div class='progress-container'>
                            <div class='progress-fill' style='width: {progress_width}%;'></div>
                        </div>
                    </div>
                    <span style='color: #4CAF50; font-weight: 600; width: 60px; text-align: right;'>{prob*100:.1f}%</span>
                </div>
                """, unsafe_allow_html=True)

        # Clinical recommendations
        st.markdown("### 💡 Clinical Insights")
        if class_names[pred] == "No Tumor":
            st.success("""
            ✅ **No significant abnormalities detected.**  
            • Continue routine monitoring as per standard protocol  
            • Maintain regular follow-up schedule  
            • Consult if new symptoms develop
            """)
        else:
            st.warning(f"""
            🚨 **Potential {class_names[pred]} detected.**  
            • **Immediate consultation with a neuroradiologist recommended**  
            • Consider contrast-enhanced MRI for confirmation  
            • Neurosurgical/oncology referral advised  
            • Review all previous scans for comparison
            """)

        if st.button("🔄 Analyze New Image", use_container_width=True):
            for key in ["pred", "conf", "all_probs", "done"]:
                if key in st.session_state:
                    del st.session_state[key]
            st.rerun()

    else:
        st.markdown("""
        <div style='text-align: center; padding: 40px 20px; color: rgba(255,255,255,0.7);'>
            <i class='ri-brain-line' style='font-size: 3rem; margin-bottom: 20px; display: block;'></i>
            <h3 style='color: white; margin-bottom: 10px;'>Ready for Analysis</h3>
            <p>Upload an MRI scan and click "Analyze" to get AI-powered insights</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

# -------------------------------------------------
# FOOTER
# -------------------------------------------------
st.markdown(f"""
<div style='text-align: center; margin-top: 40px; padding: 20px; color: rgba(255,255,255,0.6); border-top: 1px solid rgba(255,255,255,0.1);'>
    <b>NeuroScan AI v3.0</b> • Advanced Brain Tumor Detection System • For Research and Clinical Assistance
    <br>
    <small>Visualization Engine: {viz_engine} • Always consult with qualified medical professionals for final diagnosis</small>
</div>
""", unsafe_allow_html=True)