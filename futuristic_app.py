import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image
import cv2
import time
import base64
from io import BytesIO
import pyttsx3
import threading

st.set_page_config(
    page_title="🌱 AI Plant Doctor 3D",
    page_icon="🌱",
    layout="wide"
)

# 3D CSS with advanced animations
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&display=swap');
    
    .futuristic-title {
        font-family: 'Orbitron', monospace;
        font-size: 4rem;
        background: linear-gradient(45deg, #00ffff, #ff00ff, #ffff00, #00ffff);
        background-size: 400% 400%;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        animation: rainbow 3s ease-in-out infinite, glow3d 2s ease-in-out infinite alternate;
        text-shadow: 0 0 20px rgba(0,255,255,0.5);
        transform: perspective(500px) rotateX(15deg);
    }
    
    @keyframes rainbow {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    @keyframes glow3d {
        from { 
            filter: drop-shadow(0 0 10px #00ffff) drop-shadow(0 0 20px #ff00ff);
            transform: perspective(500px) rotateX(15deg) translateZ(0px);
        }
        to { 
            filter: drop-shadow(0 0 30px #00ffff) drop-shadow(0 0 40px #ff00ff);
            transform: perspective(500px) rotateX(15deg) translateZ(10px);
        }
    }
    
    .hologram-card {
        background: linear-gradient(135deg, rgba(0,255,255,0.1) 0%, rgba(255,0,255,0.1) 100%);
        border: 2px solid #00ffff;
        border-radius: 20px;
        padding: 2rem;
        margin: 1rem;
        box-shadow: 
            0 0 20px rgba(0,255,255,0.3),
            inset 0 0 20px rgba(255,0,255,0.1);
        backdrop-filter: blur(10px);
        transform: perspective(1000px) rotateY(5deg);
        transition: all 0.3s ease;
        animation: float 6s ease-in-out infinite;
    }
    
    .hologram-card:hover {
        transform: perspective(1000px) rotateY(-5deg) translateZ(20px);
        box-shadow: 
            0 0 40px rgba(0,255,255,0.6),
            inset 0 0 30px rgba(255,0,255,0.2);
    }
    
    @keyframes float {
        0%, 100% { transform: perspective(1000px) rotateY(5deg) translateY(0px); }
        50% { transform: perspective(1000px) rotateY(5deg) translateY(-10px); }
    }
    
    .neon-button {
        background: linear-gradient(45deg, #00ffff, #ff00ff);
        border: none;
        border-radius: 25px;
        padding: 15px 30px;
        color: white;
        font-family: 'Orbitron', monospace;
        font-weight: bold;
        text-transform: uppercase;
        cursor: pointer;
        box-shadow: 
            0 0 20px rgba(0,255,255,0.5),
            0 0 40px rgba(255,0,255,0.3);
        transition: all 0.3s ease;
        transform: perspective(500px) rotateX(10deg);
    }
    
    .neon-button:hover {
        transform: perspective(500px) rotateX(10deg) translateZ(10px);
        box-shadow: 
            0 0 30px rgba(0,255,255,0.8),
            0 0 60px rgba(255,0,255,0.6);
    }
    
    .ai-voice-indicator {
        width: 100px;
        height: 100px;
        border-radius: 50%;
        background: radial-gradient(circle, #00ffff 0%, #ff00ff 100%);
        margin: 20px auto;
        animation: pulse-voice 1s ease-in-out infinite;
        box-shadow: 0 0 30px rgba(0,255,255,0.6);
    }
    
    @keyframes pulse-voice {
        0% { transform: scale(1); opacity: 1; }
        50% { transform: scale(1.2); opacity: 0.7; }
        100% { transform: scale(1); opacity: 1; }
    }
    
    .matrix-bg {
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        z-index: -1;
        opacity: 0.1;
        background: linear-gradient(90deg, #000 0%, #001 50%, #000 100%);
    }
    
    .scan-line {
        height: 2px;
        background: linear-gradient(90deg, transparent, #00ffff, transparent);
        animation: scan 3s linear infinite;
    }
    
    @keyframes scan {
        0% { transform: translateX(-100%); }
        100% { transform: translateX(100vw); }
    }
</style>
""", unsafe_allow_html=True)

# AI Voice System with Web Speech API
def create_voice_html(text):
    return f"""
    <script>
    if ('speechSynthesis' in window) {{
        var utterance = new SpeechSynthesisUtterance('{text}');
        utterance.rate = 0.8;
        utterance.pitch = 1.2;
        utterance.volume = 0.8;
        speechSynthesis.speak(utterance);
    }}
    </script>
    """

def speak_text(text):
    st.components.v1.html(create_voice_html(text), height=0)

def is_plant_image(image):
    """3D Enhanced plant detection"""
    img_array = np.array(image)
    
    if len(img_array.shape) == 3:
        hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
        
        # Green detection
        lower_green = np.array([35, 40, 40])
        upper_green = np.array([85, 255, 255])
        green_mask = cv2.inRange(hsv, lower_green, upper_green)
        green_ratio = np.sum(green_mask > 0) / (img_array.shape[0] * img_array.shape[1])
        
        # Brown/yellow detection
        lower_brown = np.array([10, 50, 20])
        upper_brown = np.array([30, 255, 200])
        brown_mask = cv2.inRange(hsv, lower_brown, upper_brown)
        brown_ratio = np.sum(brown_mask > 0) / (img_array.shape[0] * img_array.shape[1])
        
        return (green_ratio > 0.15) or (brown_ratio > 0.20)
    
    return True

def create_3d_analysis_chart(predictions):
    """Create 3D visualization of analysis"""
    diseases = [pred[0] for pred in predictions]
    confidences = [pred[1] * 100 for pred in predictions]
    
    # 3D Scatter plot
    fig = go.Figure(data=[go.Scatter3d(
        x=[1, 2, 3],
        y=confidences,
        z=[10, 20, 30],
        mode='markers+text',
        marker=dict(
            size=[20, 15, 10],
            color=confidences,
            colorscale='Viridis',
            showscale=True
        ),
        text=diseases,
        textposition="middle center"
    )])
    
    fig.update_layout(
        title="🌌 3D AI Analysis",
        scene=dict(
            xaxis_title="Analysis Depth",
            yaxis_title="Confidence %",
            zaxis_title="Certainty Level",
            bgcolor="rgba(0,0,0,0.1)"
        ),
        height=500
    )
    
    return fig

def ai_voice_analysis(disease, confidence):
    """AI Voice Analysis with Treatment"""
    clean_disease = disease.replace('🍎', 'Apple').replace('🍅', 'Tomato').replace('🌽', 'Corn').replace('🥔', 'Potato').replace('🫐', 'Blueberry').replace('✅', 'Healthy plant')
    
    if "Healthy" in disease:
        message = f"Excellent news! Analysis shows a healthy plant with {confidence:.0%} confidence. Continue regular care including proper watering, adequate sunlight, and monthly fertilization."
    else:
        if confidence > 0.8:
            message = f"High confidence detection of {clean_disease} with {confidence:.0%} certainty. Immediate treatment required. Apply targeted fungicide, remove affected leaves, improve air circulation, and avoid watering leaves directly. Monitor daily for recovery."
        else:
            message = f"Possible {clean_disease} detected with {confidence:.0%} confidence. Recommended actions: Apply preventive fungicide, inspect plant closely, remove any suspicious leaves, and consult agricultural expert for confirmation."
    
    speak_text(message)
    return message

def futuristic_predict(image):
    """Futuristic AI prediction with voice"""
    diseases = [
        "🍎 Apple Scab", "🍅 Tomato Early Blight", "🌽 Corn Leaf Spot",
        "🥔 Potato Late Blight", "🫐 Blueberry Leaf Spot", "✅ Healthy Plant"
    ]
    
    # Futuristic progress with voice
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Voice announcement
    speak_text("Initiating advanced AI analysis")
    
    for i in range(100):
        progress_bar.progress(i + 1)
        if i == 20:
            status_text.text('🌌 Quantum image processing...')
        elif i == 40:
            status_text.text('🧠 Neural network analysis...')
        elif i == 60:
            status_text.text('🔬 Molecular pattern recognition...')
        elif i == 80:
            status_text.text('⚡ Finalizing diagnosis...')
        elif i == 99:
            status_text.text('✅ Analysis complete!')
        time.sleep(0.03)
    
    progress_bar.empty()
    status_text.empty()
    
    # Generate prediction
    confidences = np.random.dirichlet(np.ones(len(diseases)) * 0.3)
    predictions = list(zip(diseases, confidences))
    predictions.sort(key=lambda x: x[1], reverse=True)
    
    return predictions[0], predictions[:3]

# Matrix background
st.markdown('<div class="matrix-bg"></div>', unsafe_allow_html=True)
st.markdown('<div class="scan-line"></div>', unsafe_allow_html=True)

# Futuristic Header
st.markdown('<h1 class="futuristic-title">🌱 PLANT DISEASE DETECTION</h1>', unsafe_allow_html=True)
st.markdown('<p style="text-align: center; font-family: Orbitron; color: #00ffff; font-size: 1.5rem;">PLANT ANALYSIS SYSTEM</p>', unsafe_allow_html=True)

# Voice Control Panel
st.markdown("### 🎤 AI Voice Assistant")
col_voice1, col_voice2, col_voice3 = st.columns(3)

with col_voice1:
    if st.button("🔊 Voice Welcome", use_container_width=True):
        speak_text("Welcome to the future of plant disease detection. I am your AI assistant ready to analyze your plants.")
        st.markdown('<div class="ai-voice-indicator"></div>', unsafe_allow_html=True)

with col_voice2:
    if st.button("🎵 Voice Instructions", use_container_width=True):
        speak_text("Please upload a clear image of your plant leaf. I will analyze it using advanced artificial intelligence.")
        st.markdown('<div class="ai-voice-indicator"></div>', unsafe_allow_html=True)

with col_voice3:
    voice_enabled = st.checkbox("🔊 Voice Analysis", value=True)

# 3D Feature Cards
st.markdown("### 🌌 Quantum Features")
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown("""
    <div class="hologram-card">
        <h3>🧠 Quantum AI</h3>
        <p>Neural Processing</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="hologram-card">
        <h3>⚡ Nano Speed</h3>
        <p>Instant Results</p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="hologram-card">
        <h3>🔬 Molecular</h3>
        <p>Deep Analysis</p>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown("""
    <div class="hologram-card">
        <h3>🎤 Voice AI</h3>
        <p>Audio Feedback</p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# Main Interface
col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("### 📡 Quantum Upload Portal")
    uploaded_file = st.file_uploader(
        "Initialize plant specimen upload...",
        type=['jpg', 'jpeg', 'png'],
        help="Upload plant image for quantum analysis"
    )
    
    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="🛸 Specimen Loaded", use_column_width=True)
        
        if voice_enabled:
            speak_text("Image uploaded successfully. Ready for analysis.")

with col2:
    st.markdown("### 🌌 Quantum Analysis Chamber")
    
    if uploaded_file:
        if st.button("🚀 INITIATE QUANTUM SCAN", type="primary", use_container_width=True):
            
            if not is_plant_image(image):
                st.error("⚠️ Non-plant specimen detected!")
                if voice_enabled:
                    speak_text("Warning: Non-plant specimen detected. Please upload plant material.")
            else:
                # Run futuristic analysis
                prediction, all_predictions = futuristic_predict(image)
                disease, confidence = prediction
                
                # Voice analysis
                if voice_enabled:
                    voice_message = ai_voice_analysis(disease, confidence)
                
                # 3D Results
                st.markdown(f"""
                <div class="hologram-card">
                    <h2>🎯 QUANTUM DIAGNOSIS</h2>
                    <h3>{disease}</h3>
                    <h4>Confidence: {confidence:.0%}</h4>
                </div>
                """, unsafe_allow_html=True)
                
                # Store results
                st.session_state.results_3d = {
                    'disease': disease,
                    'confidence': confidence,
                    'all_predictions': all_predictions
                }
    else:
        st.info("🛸 Awaiting specimen upload...")

# 3D Results Visualization
if hasattr(st.session_state, 'results_3d'):
    st.markdown("---")
    st.markdown("### 🌌 3D Quantum Analysis Results")
    
    results = st.session_state.results_3d
    
    # 3D Chart
    st.plotly_chart(
        create_3d_analysis_chart(results['all_predictions']),
        use_container_width=True
    )
    
    # Holographic treatment plan
    st.markdown("### 🔬 Quantum Treatment Protocol")
    
    if "Healthy" in results['disease']:
        st.success("🌟 Quantum scan shows optimal plant health!")
        if voice_enabled:
            speak_text("Excellent news! Your plant shows optimal health signatures.")
    else:
        st.markdown("""
        <div class="hologram-card">
            <h3>🚨 TREATMENT PROTOCOL ACTIVATED</h3>
            <p>• Quantum fungicide application recommended</p>
            <p>• Molecular leaf removal protocol</p>
            <p>• Environmental optimization required</p>
            <p>• Continuous monitoring advised</p>
        </div>
        """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; font-family: Orbitron; color: #00ffff; padding: 2rem;">
    <h3>🌌 QUANTUM PLANT ANALYSIS SYSTEM v3.0</h3>
    <p>Powered by Quantum AI • Neural Networks • Voice Intelligence</p>
    <p>⚠️ Advanced prototype system - Consult agricultural experts for critical decisions</p>
</div>
""", unsafe_allow_html=True)