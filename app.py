import streamlit as st
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier

# ==========================================
# 1. Page Configuration & Styling
# ==========================================
st.set_page_config(layout="wide", page_title="Eco-Mind Forest AI")

st.markdown("""
<style>
    .main-header { font-size: 24px; font-weight: bold; color: #1B5E20; margin-bottom: 10px; }
    .card { background-color: #f8f9fa; padding: 15px; border-radius: 10px; border: 1px solid #ddd; margin-bottom: 15px; }
    .highlight { color: #2E7D32; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. Core Logic: Auto-Train Lightweight Model
# ==========================================
@st.cache_resource
def build_simulation_engine():
    """
    Simulates data and trains a model in memory.
    This avoids uploading large CSV files and prevents encoding errors.
    """
    # 1. Generate Simulation Data (2000 samples)
    np.random.seed(42)
    n = 2000
    
    # Simulate feature distributions based on real dataset statistics
    elevation = np.random.normal(2900, 400, n)
    slope = np.random.normal(15, 10, n)
    fire = np.random.normal(1500, 1000, n)
    road = np.random.normal(2000, 1500, n)
    soil = np.random.randint(0, 40, n)
    
    df = pd.DataFrame({
        'Elevation': elevation,
        'Slope': slope,
        'Horizontal_Distance_To_Fire_Points': fire,
        'Horizontal_Distance_To_Roadways': road,
        'Soil_Type': soil
    })
    
    # 2. Define Target Variable (Ecological Rules)
    # High Elevation -> Type 1 (Spruce/Fir)
    # Mid Elevation -> Type 2 (Lodgepole Pine)
    # Low Elevation -> Type 3 (Ponderosa Pine)
    conditions = [
        (df['Elevation'] > 3000),
        (df['Elevation'] > 2500) & (df['Elevation'] <= 3000),
        (df['Elevation'] <= 2500)
    ]
    choices = [1, 2, 3] 
    df['Cover_Type'] = np.select(conditions, choices, default=2)
    
    # 3. Train Model
    X = df.drop('Cover_Type', axis=1)
    y = df['Cover_Type']
    model = RandomForestClassifier(n_estimators=50, max_depth=8, random_state=42)
    model.fit(X, y)
    
    # 4. Initialize SHAP Explainer
    explainer = shap.TreeExplainer(model)
    
    return model, explainer, df

# Load system resources
model, explainer, reference_data = build_simulation_engine()

# ==========================================
# 3. Layout & Interface
# ==========================================

# Header
st.markdown('<div class="main-header">🌲 Eco-Mind: Forest Decision Support System</div>', unsafe_allow_html=True)
st.markdown("Context-Aware Navigation for Sustainable Forest Management")
st.markdown("---")

# Layout: Left (Controls) and Right (Results)
col_left, col_right = st.columns([1, 2], gap="large")

# --- Left Side: Control Panel ---
with col_left:
    st.markdown("### 📍 Feature Input")
    st.info("Adjust sliders to simulate different forest conditions.")
    
    with st.container():
        # Core Features
        input_elev = st.slider("Elevation (m)", 1800, 4000, 3210, help="Highest impact feature")
        input_fire = st.slider("Dist. to Fire (m)", 0, 6000, 339)
        input_road = st.slider("Dist. to Road (m)", 0, 6000, 3144)
        input_slope = st.slider("Slope (degrees)", 0, 60, 10)
        input_soil = st.selectbox("Soil Type", [f"Type {i}" for i in range(1,41)], index=28)
        
        # Construct Input Dataframe
        input_df = pd.DataFrame({
            'Elevation': [input_elev],
            'Slope': [input_slope],
            'Horizontal_Distance_To_Fire_Points': [input_fire],
            'Horizontal_Distance_To_Roadways': [input_road],
            'Soil_Type': [int(input_soil.split(" ")[1])] # Parse "Type 29" -> 29
        })

# --- Right Side: Real-time Analysis ---
with col_right:
    
    # 1. Real-time Prediction
    st.markdown("### 1. Real-time Prediction")
    
    # Inference
    pred_class = model.predict(input_df)[0]
    pred_prob = np.max(model.predict_proba(input_df))
    
    # Map Class Names
    names = {1: "Spruce/Fir (Type 1)", 2: "Lodgepole Pine (Type 2)", 3: "Ponderosa Pine (Type 3)"}
    pred_name = names.get(pred_class, "Unknown")
    
    # Display Result Cards
    c1, c2, c3 = st.columns(3)
    c1.metric("Forest Type", pred_name)
    c2.metric("Confidence", f"{pred_prob*100:.1f}%")
    
    if pred_class == 1:
        c3.error("High Elevation Zone")
    elif pred_class == 3:
        c3.success("Restoration Target")
    else:
        c3.warning("Mid-Elevation Zone")

    st.markdown("---")
    
    # 2. Dynamic SHAP Explanation
    st.markdown(f"### 2. Why {pred_name}? (Local Explanation)")
    
    # Calculate SHAP
    shap_values = explainer(input_df)
    
    # Get SHAP values for the predicted class
    class_idx = int(pred_class) - 1
    shap_explanation = shap.Explanation(
        values=shap_values.values[0, :, class_idx],
        base_values=shap_values.base_values[0, class_idx],
        data=input_df.iloc[0],
        feature_names=input_df.columns
    )
    
    # Plot Waterfall
    fig, ax = plt.subplots(figsize=(8, 3))
    shap.plots.waterfall(shap_explanation, show=False, max_display=6)
    st.pyplot(fig)
    
    st.markdown("---")
    
    # 3. Counterfactual Suggestions (Simplified DiCE)
    st.markdown("### 3. How to Restore Ponderosa Pine (Type 3)?")
    
    if pred_class == 3:
        st.success("✅ Current condition already matches Ponderosa Pine habitat.")
    else:
        st.write("Planner's Goal: Modify environment to support Type 3.")
        
        # Calculate Difference
        elev_diff = 2400 - input_elev # Assume target is 2400m
        road_diff = 1000 - input_road # Assume target is 1000m
        
        # Display Recommendation Table
        rec_data = {
            "Factor": ["Elevation", "Dist. to Road", "Dist. to Fire"],
            "Current": [f"{input_elev}m", f"{input_road}m", f"{input_fire}m"],
            "Target (Type 3)": ["~2400m", "<1500m", "Any"],
            "Action Required": [f"{elev_diff:+}m", f"{road_diff:+}m", "-"]
        }
        st.dataframe(pd.DataFrame(rec_data), use_container_width=True)
        
        if abs(elev_diff) > 500:
             st.error("❌ **Feasibility Alert:** Requires reducing elevation by >500m. Ecologically impossible.")
        else:
             st.success("✅ **Feasibility:** High. Modifications are realistic.")

# Footer
st.markdown("---")
st.caption("Powered by Eco-Mind XAI Engine")
