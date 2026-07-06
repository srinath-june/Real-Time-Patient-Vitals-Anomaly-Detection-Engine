import streamlit as st
import numpy as np
import pandas as pd
import time
from sklearn.ensemble import IsolationForest

# 1. Page Configuration for ICU Dashboard
st.set_page_config(
    page_title="ICU Real-Time Vitals Monitor",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🏥 Real-Time Patient Vitals Anomaly Detection Engine")
st.markdown("---")

# 2. Simulate Baseline "Normal" ICU Patient Data for Training
@st.cache_resource
def train_anomaly_detector():
    """Trains an Isolation Forest model on normal physiological data."""
    np.random.seed(42)
    # Normal vitals distribution
    normal_hr = np.random.normal(loc=75, scale=8, size=1000)       # 60-90 bpm
    normal_spo2 = np.random.normal(loc=98, scale=1, size=1000)     # 95-100%
    normal_bp = np.random.normal(loc=120, scale=10, size=1000)     # 100-140 mmHg
    
    X_train = pd.DataFrame({
        'Heart_Rate': normal_hr,
        'SpO2': normal_spo2,
        'Blood_Pressure': normal_bp
    })
    # Clip SpO2 to a realistic maximum of 100%
    X_train['SpO2'] = X_train['SpO2'].clip(upper=100)
    
    # Contamination set low because training data is mostly clean/normal
    model = IsolationForest(contamination=0.02, random_state=42)
    model.fit(X_train)
    return model

detector = train_anomaly_detector()

# 3. Sidebar Controls
st.sidebar.header("🎛️ ICU Unit Controls")
patient_id = st.sidebar.selectbox("Select Patient Bed:", ["Bed-04 (ICU East)", "Bed-09 (ICU East)", "Bed-12 (ICU West)"])
anomaly_trigger = st.sidebar.button("🚨 Simulate Patient Distress Event")

st.sidebar.markdown("---")
st.sidebar.info(
    "**System Note:** This system utilizes an unsupervised **Isolation Forest** ML algorithm "
    "to detect multi-variable anomalies in real-time streams before traditional single-threshold alarms trigger."
)

# 4. Streamlit Placeholders for Live Updates
metric_row = st.columns(3)
chart_placeholder = st.empty()
alert_placeholder = st.empty()

# Initialize session state for data history
if 'history' not in st.session_state:
    st.session_state.history = pd.DataFrame(columns=['Timestamp', 'Heart_Rate', 'SpO2', 'Blood_Pressure', 'Is_Anomaly'])

# 5. Live Streaming Loop
# Simple counter loop to mimic continuous edge data stream
for i in range(100):
    current_time = time.strftime("%H:%M:%S")
    
    # Generate live stream data
    if anomaly_trigger and i in range(5, 15):
        # Inject an anomaly: High Heart Rate + Crashing SpO2 (Hypoxia/Distress)
        live_hr = float(np.random.normal(135, 5))
        live_spo2 = float(np.random.normal(84, 2))
        live_bp = float(np.random.normal(155, 8))
    else:
        # Standard fluctuating patient data
        live_hr = float(np.random.normal(76, 4))
        live_spo2 = float(np.random.normal(97.5, 0.8))
        live_bp = float(np.random.normal(118, 5))
        
    live_spo2 = min(live_spo2, 100.0)
    
    # Format for model prediction
    current_vitals = pd.DataFrame([{
        'Heart_Rate': live_hr,
        'SpO2': live_spo2,
        'Blood_Pressure': live_bp
    }])
    
    # Predict Anomaly (-1 standard for anomaly in sklearn, 1 for normal)
    prediction = detector.predict(current_vitals)[0]
    is_anomaly = 1 if prediction == -1 else 0
    
    # Append to session state history
    new_entry = pd.DataFrame([{
        'Timestamp': current_time,
        'Heart_Rate': round(live_hr, 1),
        'SpO2': round(live_spo2, 1),
        'Blood_Pressure': round(live_bp, 1),
        'Is_Anomaly': is_anomaly
    }])
    st.session_state.history = pd.concat([st.session_state.history, new_entry], ignore_index=True).tail(30)
    
    # 6. Update Dashboard UI Components
    with alert_placeholder:
        if is_anomaly == 1:
            st.error(f"🚨 CRITICAL ANOMALY DETECTED AT {current_time} FOR {patient_id}! Multi-variable vitals cross-matching indicates physiological distress.")
        else:
            st.success(f"✅ {patient_id} Vitals stable. ML Engine monitoring stream...")

    # Update metric cards dynamically
    with metric_row[0]:
        st.metric(label="Heart Rate (BPM)", value=f"{round(live_hr, 1)}", delta="- Normal" if is_anomaly==0 else "⚠️ STRESS", delta_color="inverse" if is_anomaly==1 else "normal")
    with metric_row[1]:
        st.metric(label="SpO2 (Oxygen %)", value=f"{round(live_spo2, 1)}%", delta="- Stable" if is_anomaly==0 else "⚠️ DROPPING", delta_color="inverse" if is_anomaly==1 else "normal")
    with metric_row[2]:
        st.metric(label="Systolic BP (mmHg)", value=f"{round(live_bp, 1)}", delta="- Normal" if is_anomaly==0 else "⚠️ FLUID CHANGE", delta_color="inverse" if is_anomaly==1 else "normal")

    # Display rolling real-time chart
    with chart_placeholder:
        chart_data = st.session_state.history.set_index('Timestamp')
        st.line_chart(chart_data[['Heart_Rate', 'SpO2', 'Blood_Pressure']])
        
    time.sleep(1) # Frequency of simulated sensor data packet (1Hz)
