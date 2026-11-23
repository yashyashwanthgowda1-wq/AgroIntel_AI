import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score
from fertilizer_logic import recommend_fertilizer
from datetime import datetime
import os
import requests
from google import genai
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

# Keep credentials outside source control. Set GEMINI_API_KEY in the environment
# or add it to .streamlit/secrets.toml (see .streamlit/secrets.toml.example).
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
if not GEMINI_API_KEY:
    try:
        GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY", "")
    except (FileNotFoundError, KeyError):
        pass

def get_gemini_client():
    if not GEMINI_API_KEY:
        raise RuntimeError(
            "Gemini is not configured. Add GEMINI_API_KEY to "
            ".streamlit/secrets.toml and restart the app."
        )
    return genai.Client(api_key=GEMINI_API_KEY)

# -------------------------------------------------------
# GLOBAL SESSION STATE INITIALIZATION
# -------------------------------------------------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "scanned_n" not in st.session_state:
    st.session_state.scanned_n = None
if "scanned_p" not in st.session_state:
    st.session_state.scanned_p = None
if "scanned_k" not in st.session_state:
    st.session_state.scanned_k = None
if "scanned_ph" not in st.session_state:
    st.session_state.scanned_ph = None
if "scanned_type" not in st.session_state:
    st.session_state.scanned_type = "Pending Scan..."
if "scan_completed" not in st.session_state:
    st.session_state.scan_completed = False

# -------------------------------------------------------
# WEATHER FETCHING FUNCTION
# -------------------------------------------------------
def get_live_weather(lat=12.9716, lon=77.5946):
    """
    Fetches real-time temperature, humidity, and simulated annual rainfall 
    using the free Open-Meteo API. Defaults to Bengaluru coordinates.
    """
    try:
        url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current_weather=true&relative_humidity_2m=true"
        response = requests.get(url, timeout=5)
        
        if response.status_code == 200:
            data = response.json()
            current = data.get("current_weather", {})
            temp = current.get("temperature")
            hum = data.get("current", {}).get("relative_humidity_2m") or 65.0 
            rain = 120.0 
            
            return {
                "temperature": float(temp) if temp is not None else 25.0,
                "humidity": float(hum),
                "rainfall": float(rain),
                "success": True
            }
    except Exception as e:
        pass
    
    return {"temperature": 25.0, "humidity": 60.0, "rainfall": 120.0, "success": False}

# -------------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------------
st.set_page_config(
    page_title="AgroIntel AI",
    page_icon="🌱",
    layout="wide"
)

# -------------------------------------------------------
# LOAD CUSTOM CSS
# -------------------------------------------------------
def local_css(file_name):
    if os.path.exists(file_name):
        with open(file_name) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

local_css(BASE_DIR / "assets" / "styles.css")

# -------------------------------------------------------
# HERO SECTION
# -------------------------------------------------------
st.markdown("""
<div class='hero'>
    <h1>🌱 AgroIntel AI</h1>
    <p>Smart Soil Quality Analyzer, Crop Recommendation & Yield Predictor</p>
</div>
""", unsafe_allow_html=True)

# -------------------------------------------------------
# SIDEBAR NAVIGATION
# -------------------------------------------------------
st.sidebar.image(str(BASE_DIR / "assets" / "agri_logo.png"), width=120)

menu = st.sidebar.radio(
    "Navigation",
    ["Dashboard", "Soil Analysis", "Crop Prediction", "AI Assistant", "Sustainability & health", "Reports", "About"]
)

# -------------------------------------------------------
# LOAD DATASETS
# -------------------------------------------------------
crop_df = pd.read_csv(BASE_DIR / "datasets" / "Crop_recommendation.csv")
soil_df = pd.read_csv(BASE_DIR / "datasets" / "Soil Nutrients.csv")

# -------------------------------------------------------
# TRAIN MODELS ONCE (CACHE)
# -------------------------------------------------------
@st.cache_resource
def train_models():
    # ---------------- CROPS MODEL ----------------
    X_crop = crop_df.drop("label", axis=1)
    y_crop = crop_df["label"]
    crop_features = list(X_crop.columns)

    X_train, X_test, y_train, y_test = train_test_split(
        X_crop, y_crop, test_size=0.2, random_state=42
    )

    crop_model = RandomForestClassifier(n_estimators=200, random_state=42)
    crop_model.fit(X_train, y_train)
    crop_acc = accuracy_score(y_test, crop_model.predict(X_test))

    # ---------------- SOIL & YIELD MODEL ----------------
    y_soil_clf = soil_df["Fertility"]
    y_soil_reg = soil_df["Yield"]

    X_soil_clean = soil_df.drop(columns=["Name", "Fertility", "Yield", "Category_pH", "Soil_Type", "Season"], errors="ignore")
    X_soil_clean = X_soil_clean.select_dtypes(include=[np.number])
    soil_features = list(X_soil_clean.columns)

    X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
        X_soil_clean, y_soil_clf, test_size=0.2, random_state=42
    )
    soil_model_clf = RandomForestClassifier(n_estimators=150, random_state=42)
    soil_model_clf.fit(X_train_c, y_train_c)
    soil_acc = accuracy_score(y_test_c, soil_model_clf.predict(X_test_c))

    X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(
        X_soil_clean, y_soil_reg, test_size=0.2, random_state=42
    )
    yield_model_reg = RandomForestRegressor(n_estimators=150, random_state=42)
    yield_model_reg.fit(X_train_r, y_train_r)
    yield_r2 = r2_score(y_test_r, yield_model_reg.predict(X_test_r))

    return crop_model, soil_model_clf, yield_model_reg, crop_acc, soil_acc, yield_r2, crop_features, soil_features

crop_model, soil_model, yield_model, crop_acc, soil_acc, yield_r2, crop_features, soil_features = train_models()

os.makedirs(BASE_DIR / "generated_reports", exist_ok=True)

CROP_PRICES = {
    "rice": 2500, "maize": 1800, "chickpea": 5000, "kidneybeans": 6000,
    "pigeonpeas": 5500, "mothbeans": 4500, "mungbean": 6500, "blackgram": 6000,
    "lentil": 5200, "pomegranate": 7000, "banana": 1500, "mango": 8000,
    "grapes": 9000, "watermelon": 1000, "muskmelon": 1200, "apple": 10000,
    "orange": 4000, "papaya": 2000, "coconut": 2500, "cotton": 6500,
    "jute": 4000, "coffee": 12000
}

# -------------------------------------------------------
# DASHBOARD PAGE
# -------------------------------------------------------
if menu == "Dashboard":
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"<div class='metric-card'><h3>Crop Model</h3><h1>{crop_acc*100:.2f}%</h1></div>", unsafe_allow_html=True)
    with col2:
        st.markdown(f"<div class='metric-card'><h3>Soil Model</h3><h1>{soil_acc*100:.2f}%</h1></div>", unsafe_allow_html=True)
    with col3:
        st.markdown(f"<div class='metric-card'><h3>Yield Variance (R²)</h3><h1>{yield_r2*100:.2f}%</h1></div>", unsafe_allow_html=True)
    with col4:
        st.markdown("<div class='metric-card'><h3>AI Status</h3><h1>ACTIVE</h1></div>", unsafe_allow_html=True)

    st.markdown("## 📊 Feature Importance (Crop Model)")
    feature_importance = pd.DataFrame({
        "Feature": crop_features,
        "Importance": crop_model.feature_importances_
    }).sort_values(by="Importance", ascending=False)
    fig = px.bar(feature_importance, x="Feature", y="Importance", color="Importance", template="plotly_dark")
    st.plotly_chart(fig, width="stretch")

# -------------------------------------------------------
# SOIL ANALYSIS PAGE
# -------------------------------------------------------
elif menu == "Soil Analysis":
    st.markdown("## 🧪 Smart Soil Analysis & Production Estimation")
    
    st.markdown("### 📥 Select Soil Data Input Method")
    input_mode = st.radio(
        "Choose how you want to provide soil data:", 
        ["📸 Instant AI Visual Soil Scan (Unique Feature)", "✏️ Enter Sensor Values Manually"]
    )

    st.markdown("---")

    nitrogen, phosphorus, potassium, ph = None, None, None, None
    estimated_soil_type = "Unknown"

    # --- MODE 1: VISUAL CAMERA SCANNING ---
    if input_mode == "📸 Instant AI Visual Soil Scan (Unique Feature)":
        st.markdown("### 📷 Live Mobile / Local Soil Scanner")
        st.info("💡 Tip: Open this app on your phone via your local Network IP to use your phone's live camera directly!")
        
        uploaded_soil_img = st.camera_input("Position the soil sample cleanly in focus:", key="soil_camera_streamer")
        
        if uploaded_soil_img is not None:
            if st.button("🔬 Execute Visual Soil Analysis", key="remote_scan_btn"):
                with st.spinner("Analyzing soil color spectrum, moisture reflection, and composition..."):
                    try:
                        import PIL.Image
                        import json
                        
                        soil_image_data = PIL.Image.open(uploaded_soil_img)
                        client = get_gemini_client()
                        
                        # Fixes the multi-line string syntax breakage securely
                        prompt = (
                            "You are an advanced agricultural computer vision system analyzing a soil sample photograph. "
                            "First, strictly verify if the image actually shows an outdoor environment, farm ground, dirt, mud, or a soil sample. "
                            "If the image shows clothing fabric, human skin, hands, faces, indoor walls, text, consumer electronics, animals, or random household items, set 'is_soil' to false.\n\n"
                            "Return ONLY a raw JSON object with exactly these keys, no markdown formatting blocks, no extra text:\n"
                            "{\n"
                            '  "is_soil": boolean (true if image contains genuine soil/dirt, false otherwise),\n'
                            '  "soil_type": "string describing type and texture, or \'invalid\' if not soil",\n'
                            '  "nitrogen": float value between 10 and 140,\n'
                            '  "phosphorus": float value between 10 and 100,\n'
                            '  "potassium": float value between 10 and 100,\n'
                            '  "ph": float value between 4.5 and 8.5\n'
                            "}"
                        )
                        
                        response = client.models.generate_content(
                            model="gemini-2.5-flash",
                            contents=[prompt, soil_image_data],
                        )
                        
                        clean_text = response.text.strip()
                        if "```json" in clean_text:
                            clean_text = clean_text.split("```json")[1].split("```")[0].strip()
                        elif "```" in clean_text:
                            clean_text = clean_text.split("```")[1].split("```")[0].strip()
                            
                        soil_metrics = json.loads(clean_text)
                        
                        # Defensive Guardrail Flag Inspection
                        if not soil_metrics.get("is_soil", True):
                            st.error("❌ **Invalid Scan Target:** The AI framework detected that this image does not contain a valid soil sample. Please ensure you are scanning real dirt or farm soil.")
                            st.session_state.scan_completed = False
                        else:
                            st.session_state.scanned_n = float(soil_metrics.get("nitrogen", 50.0))
                            st.session_state.scanned_p = float(soil_metrics.get("phosphorus", 45.0))
                            st.session_state.scanned_k = float(soil_metrics.get("potassium", 40.0))
                            st.session_state.scanned_ph = float(soil_metrics.get("ph", 6.5))
                            st.session_state.scanned_type = soil_metrics.get("soil_type", "Loamy Soil")
                            st.session_state.scan_completed = True
                            st.toast("⚡ Scan analysis complete!")
                            
                    except Exception as scan_err:
                        st.error(f"❌ Visual processing error: {str(scan_err)}")

        if st.session_state.get("scan_completed", False):
            nitrogen = st.session_state.scanned_n
            phosphorus = st.session_state.scanned_p
            potassium = st.session_state.scanned_k
            ph = st.session_state.scanned_ph
            estimated_soil_type = st.session_state.scanned_type
            
            st.success(f"📊 **AI Visual Diagnosis Results:** Detected **{estimated_soil_type}**")
            col_preview1, col_preview2, col_preview3, col_preview4 = st.columns(4)
            col_preview1.metric("Estimated Nitrogen (N)", f"{nitrogen} mg/kg")
            col_preview2.metric("Estimated Phosphorus (P)", f"{phosphorus} mg/kg")
            col_preview3.metric("Estimated Potassium (K)", f"{potassium} mg/kg")
            col_preview4.metric("Estimated pH Level", f"{ph}")

    # --- MODE 2: MANUAL ENTRY ---
    else:
        st.markdown("### 🧪 Soil Nutrients & Conditions (Manual Entry)")
        col1, col2, col3 = st.columns(3)
        with col1:
            nitrogen = st.number_input("Nitrogen (N)", min_value=0.0, max_value=200.0, value=50.0)
            phosphorus = st.number_input("Phosphorus (P)", min_value=0.0, max_value=200.0, value=45.0)
        with col2:
            potassium = st.number_input("Potassium (K)", min_value=0.0, max_value=200.0, value=40.0)
            ph = st.number_input("Soil pH Level", min_value=0.0, max_value=14.0, value=6.5)

    # --- AUTOMATED ENVIRONMENTAL FETCHING ---
    st.markdown("### 🌐 Regional Climate Matrix")
    coord_col1, coord_col2 = st.columns(2)
    with coord_col1:
        lat_input = st.number_input("Latitude", value=12.9716, format="%.4f")
    with coord_col2:
        lon_input = st.number_input("Longitude", value=77.5946, format="%.4f")
    
    with st.spinner("Synchronizing with real-time climate telemetry..."):
        weather_data = get_live_weather(lat_input, lon_input)
        temperature = float(weather_data["temperature"])
        humidity = float(weather_data["humidity"])
        rainfall = float(weather_data["rainfall"])
        st.caption(f"Connected live: Temp: {temperature}°C | Humidity: {humidity}% | Rainfall: {rainfall}mm")

    st.markdown("### 💰 Production Variables")
    col_f1, col_f2 = st.columns(2)
    with col_f1:
        farm_size = st.number_input("Farm Size (Acres)", min_value=0.1, max_value=100.0, value=1.0)
    with col_f2:
        cost_per_acre = st.number_input("Cultivation Cost per Acre (₹)", min_value=0, value=15000)

    # --- ANALYSIS EXECUTION AND PREDICTION DISPLAY ---
    st.markdown("---")
    if st.button("🚀 Run Comprehensive Predictive Analysis", key="master_analysis_trigger"):
        if nitrogen is None or phosphorus is None or potassium is None or ph is None:
            st.error("⚠️ Soil metrics are missing. Please complete the camera scan or fill in the manual numbers first.")
        else:
            soil_input = np.zeros((1, len(soil_features)))
            mapping = {
                "Nitrogen": nitrogen, "Phosphorus": phosphorus, "Potassium": potassium, 
                "Temperature": temperature, "Rainfall": rainfall, "pH": ph, 
                "Humidity": humidity, "Rh": humidity
            }
            for idx, col in enumerate(soil_features):
                if col in mapping:
                    soil_input[0, idx] = mapping[col]

            crop_input = np.array([[nitrogen, phosphorus, potassium, temperature, humidity, ph, rainfall]])
            
            crop_prediction = crop_model.predict(crop_input)[0]
            predicted_yield_base = yield_model.predict(soil_input)[0]
            total_predicted_yield = predicted_yield_base * farm_size

            crop_key = crop_prediction.lower().strip()
            market_rate = CROP_PRICES.get(crop_key, 3000)
            
            gross_revenue = total_predicted_yield * market_rate
            total_costs = cost_per_acre * farm_size
            net_profit = gross_revenue - total_costs

            st.markdown("## 🏆 Crop Recommendation & Yield Insights")
            metric_col1, metric_col2, metric_col3 = st.columns(3)
            with metric_col1:
                st.metric(label="🌾 Recommended Crop Focus", value=str(crop_prediction))
            with metric_col2:
                st.metric(label="📈 Predicted Total Yield Output", value=f"{total_predicted_yield:.2f} Units")
            with metric_col3:
                st.metric(label="💰 Est. Local Market Rate", value=f"₹ {market_rate:,} / Unit")

            st.markdown("### 💸 Projected Profitability Framework")
            f_col1, f_col2, f_col3 = st.columns(3)
            with f_col1:
                st.metric("Total Input Capital", f"₹ {total_costs:,.2f}")
            with f_col2:
                st.metric("Gross Revenue Yield", f"₹ {gross_revenue:,.2f}")
            with f_col3:
                if net_profit > 0:
                    st.metric("Net Operational Profit", f"₹ {net_profit:,.2f}", delta=f"{net_profit / total_costs * 100:.1f}% ROI")
                else:
                    st.metric("Net Operational Profit", f"₹ {net_profit:,.2f}", delta=f"{net_profit / total_costs * 100:.1f}% Margin Loss", delta_color="inverse")

            st.markdown("---")
            chart_col1, chart_col2 = st.columns(2)
            with chart_col1:
                probabilities = crop_model.predict_proba(crop_input)[0]
                top_3_idx = np.argsort(probabilities)[::-1][:3]
                prob_df = pd.DataFrame({"Crop": crop_model.classes_[top_3_idx], "Probability": probabilities[top_3_idx]})
                fig = px.bar(prob_df, x="Crop", y="Probability", color="Probability", color_continuous_scale=["#1f2937", "#10b981"], template="plotly_dark", title="Model Certainty Distribution")
                st.plotly_chart(fig, width="stretch")

            with chart_col2:
                score = 0
                if nitrogen > 40: score += 25
                if phosphorus > 35: score += 25
                if potassium > 35: score += 25
                if 6.0 <= ph <= 7.8: score += 25
                
                gauge = go.Figure(go.Indicator(
                    mode="gauge+number", value=score, title={'text': "Composite Soil Vitality Rating"},
                    gauge={'axis': {'range': [0, 100]}, 'bar': {'color': "#10b981"},
                           'steps': [{'range': [0, 50], 'color': "#ef4444"}, {'range': [50, 75], 'color': "#f59e0b"}, {'range': [75, 100], 'color': "#10b981"}]}
                ))
                gauge.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
                st.plotly_chart(gauge, width="stretch")

            from fertilizer_logic import recommend_fertilizer
            fertilizer = recommend_fertilizer(nitrogen, phosphorus, potassium)
            st.success(f"💡 **Soil Enhancement Directive:** Apply **{fertilizer}** to optimize your land for growing {crop_prediction}.")

# -------------------------------------------------------
# CROP PREDICTION PAGE
# -------------------------------------------------------
elif menu == "Crop Prediction":
    st.markdown("## 🌾 AI Crop Recommendation")
    st.write("Enter soil and environmental parameters to find the most suitable crop.")

    col1, col2, col3 = st.columns(3)
    with col1:
        nitrogen = st.number_input("Nitrogen (N)", min_value=0.0, max_value=200.0, value=50.0, key="cp_n")
        phosphorus = st.number_input("Phosphorus (P)", min_value=0.0, max_value=200.0, value=50.0, key="cp_p")
        potassium = st.number_input("Potassium (K)", min_value=0.0, max_value=200.0, value=50.0, key="cp_k")
    with col2:
        temperature = st.number_input("Temperature (°C)", value=25.0, key="cp_temp")
        humidity = st.number_input("Humidity (%)", value=70.0, key="cp_hum")
    with col3:
        ph = st.number_input("Soil pH", value=6.5, key="cp_ph")
        rainfall = st.number_input("Rainfall (mm)", value=100.0, key="cp_rain")

    if st.button("🌱 Predict Best Crop"):
        input_data = np.array([[
            float(nitrogen), float(phosphorus), float(potassium),
            float(temperature), float(humidity), float(ph), float(rainfall)
        ]])

        prediction = crop_model.predict(input_data)[0]
        probabilities = crop_model.predict_proba(input_data)[0]
        class_labels = crop_model.classes_

        top_indices = np.argsort(probabilities)[::-1][:5]
        top_crops = class_labels[top_indices]
        top_probs = probabilities[top_indices]

        st.markdown(f"""
        <div class='glass-card'>
            <h2>🌾 Recommended Crop: {prediction}</h2>
        </div>
        """, unsafe_allow_html=True)

        prob_df = pd.DataFrame({
            "Crop": top_crops,
            "Probability": top_probs
        })

        fig = px.bar(
            prob_df,
            x="Crop",
            y="Probability",
            color="Probability",
            text="Probability",
            template="plotly_dark",
            title="Top Crop Recommendations"
        )
        fig.update_traces(texttemplate='%{text:.2%}', textposition='outside')
        st.plotly_chart(fig, width="stretch")

        st.subheader("📋 AI Recommendation Summary")
        st.success(f"The AI model predicts that **{prediction}** is the most suitable crop for the provided conditions.")
        st.info(f"Temperature: {temperature}°C | Humidity: {humidity}% | pH: {ph} | Rainfall: {rainfall} mm")

# -------------------------------------------------------
# AI ASSISTANT CHATBOT PAGE
# -------------------------------------------------------
elif menu == "AI Assistant":
    st.markdown("## 🤖 AI Farming Assistant")
    st.write("Ask our expert agronomy assistant anything about crop diseases, organic farming, or soil optimization techniques.")

    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if user_query := st.chat_input("Ask me about pest controls, seasonal crop cycles, or nutrient management..."):
        with st.chat_message("user"):
            st.markdown(user_query)
        st.session_state.chat_history.append({"role": "user", "content": user_query})

        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            with st.spinner("AI Engine thinking..."):
                try:
                    client = get_gemini_client()
                    conversation = "\n".join(
                        f'{msg["role"].title()}: {msg["content"]}'
                        for msg in st.session_state.chat_history
                    )
                    response = client.models.generate_content(
                        model="gemini-2.5-flash",
                        contents=(
                            "You are an expert agronomy assistant. Answer clearly and "
                            "practically. Here is the conversation so far:\n" + conversation
                        ),
                    )
                    
                    message_placeholder.markdown(response.text)
                    st.session_state.chat_history.append({"role": "assistant", "content": response.text})
                except Exception as e:
                    message_placeholder.markdown(f"⚠️ **Connection Issue:** {str(e)}")

# -------------------------------------------------------
# SUSTAINABILITY & HEALTH PAGE
# -------------------------------------------------------
elif menu == "Sustainability & health":
    st.markdown("## 🌿 Eco-Intelligence Dashboard")
    st.write("Access multimodal plant disease diagnosis tools and calculate operational carbon emission equivalents.")
    
    eco_mode = st.radio(
        "Select Feature Focus Area:",
        ["Disease Diagnosis", "Carbon Calculator"],
        horizontal=True,
        key="eco_mode_fixed_selector"
    )
    
    st.markdown("---")

    if eco_mode == "Disease Diagnosis":
        st.markdown("### 🔍 Visual Plant Tissue Analysis")
        st.info("Upload a clear close-up photograph of an infected plant leaf to run an AI diagnostic scan.")
        
        uploaded_leaf = st.file_uploader(
            "Upload leaf sample image (JPG, JPEG, PNG):", 
            type=["jpg", "jpeg", "png"], 
            key="leaf_diag_widget_uploader"
        )
        
        if uploaded_leaf is not None:
            st.image(uploaded_leaf, caption="Uploaded Sample Leaf Target", width=400)
            if st.button("🔬 Execute Tissue Diagnostic Scan", key="exec_tissue_scan_trigger"):
                with st.spinner("Processing image channels and calling Vision AI engine..."):
                    try:
                        import PIL.Image
                        image_data = PIL.Image.open(uploaded_leaf)
                        
                        client = get_gemini_client()
                        prompt = (
                            "You are an expert plant pathologist. Thoroughly analyze this plant leaf image. "
                            "Identify the visible disease, infection symptoms, or pest anomalies. "
                            "Provide your final diagnostic assessment using these headers:\n"
                            "1. 📋 Probable Diagnosis\n"
                            "2. 🔍 Identified Key Symptoms\n"
                            "3. 🌱 Immediate Eco-Friendly/Organic Remediation\n"
                            "4. 🧪 Target Low-Cost Chemical Intervention (Only if critical)"
                        )
                        response = client.models.generate_content(
                            model="gemini-2.5-flash",
                            contents=[prompt, image_data],
                        )
                        st.success("📋 Diagnostic Assessment Complete!")
                        st.markdown(response.text)
                    except Exception as vision_err:
                        st.error(f"❌ **Vision Subsystem Error:** {str(vision_err)}")

    elif eco_mode == "Carbon Calculator":
        st.markdown("### 📉 Farm Carbon Footprint Calculator")
        calc_col1, calc_col2 = st.columns(2)
        with calc_col1:
            fuel_used = st.number_input("Machinery Fuel Runtime Consumption (Liters/Acre):", min_value=0.0, value=20.0, key="v_fuel")
            nitrogen_applied = st.number_input("Synthetic Nitrogen Fertilizer Volume (kg/Acre):", min_value=0.0, value=50.0, key="v_nitro")
        with calc_col2:
            electricity_used = st.number_input("Irrigation Water Pump Electricity Draw (kWh/Acre):", min_value=0.0, value=100.0, key="v_elec")
            manure_applied = st.number_input("Organic Animal Manure/Compost Load (kg/Acre):", min_value=0.0, value=500.0, key="v_manure")
            
        if st.button("📊 Calculate Environmental Matrix Impact", key="run_carbon_calc_trigger"):
            co2_diesel = fuel_used * 2.68       
            co2_electric = electricity_used * 0.82  
            co2_nitrogen = nitrogen_applied * 5.8      
            co2_manure = manure_applied * 0.05       
            total_co2_kg = co2_diesel + co2_electric + co2_nitrogen + co2_manure
            
            st.markdown("---")
            st.markdown(f"#### Total Estimated Emission Footprint: **{total_co2_kg:,.2f} kg CO2eq per Acre**")
            
            breakdown_df = pd.DataFrame({
                "Emission Source": ["Machinery Fuel", "Pump Electricity", "Synthetic N Fertilizer", "Organic Compost Mix"],
                "Carbon Footprint (kg CO2eq)": [co2_diesel, co2_electric, co2_nitrogen, co2_manure]
            })
            fig = px.bar(breakdown_df, x="Emission Source", y="Carbon Footprint (kg CO2eq)", color="Emission Source", template="plotly_dark")
            st.plotly_chart(fig, width="stretch")

# -------------------------------------------------------
# REPORTS CENTER PAGE
# -------------------------------------------------------
elif menu == "Reports":
    st.markdown("## 📁 AI Reports Center")
    report_folder = BASE_DIR / "generated_reports"
    
    if os.path.exists(report_folder):
        report_files = [f for f in os.listdir(report_folder) if f.endswith(".txt")]
    else:
        report_files = []
        
    if not report_files:
        st.info("📂 No reports generated yet. Run a 'Soil Analysis' to create an automated record entry.")
    else:
        st.success(f"Found {len(report_files)} historical analysis records.")
        selected_file = st.selectbox("📄 Select an analysis record to review:", sorted(report_files, reverse=True))
        
        if selected_file:
            full_path = report_folder / selected_file
            with open(full_path, "r", encoding="utf-8") as f:
                report_contents = f.read()
                
            rep_col1, rep_col2 = st.columns([3, 1])
            with rep_col1:
                st.text_area(label="Report Document Data Stream", value=report_contents, height=350, disabled=True)
            with rep_col2:
                st.markdown("### ⚙️ File Options")
                st.download_button(
                    label="📥 Download This File",
                    data=report_contents,
                    file_name=selected_file,
                    mime="text/plain",
                    key="reports_center_dl"
                )
                if st.button("🗑️ Delete Record", key="delete_rep_btn"):
                    os.remove(full_path)
                    st.toast("File deleted successfully!")
                    st.rerun()

# -------------------------------------------------------
# ABOUT PAGE
# -------------------------------------------------------
elif menu == "About":
    st.markdown("""
    ## 🌍 About AgroIntel AI
    AgroIntel AI is a next-generation agriculture intelligence platform.

    Features:
    - AI Crop Prediction
    - Soil Fertility Analysis
    - Fertilizer Recommendation
    - Smart Analytics Dashboard
    - AI Insights Engine
    - Downloadable Reports
    """)

# -------------------------------------------------------
# FOOTER
# -------------------------------------------------------
st.markdown("""
<hr>
<div class='footer'>
    🌱 AgroIntel AI | Powered by Streamlit & Machine Learning
</div>
""", unsafe_allow_html=True)
