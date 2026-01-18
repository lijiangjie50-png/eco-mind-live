import streamlit as st
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
import dice_ml
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# ==========================================
# 1. Page Config & Caching
# ==========================================
st.set_page_config(layout="wide", page_title="Forest Cover XAI Dashboard")

st.markdown("""
<style>
    .main-header { font-size: 26px; font-weight: bold; color: #1B5E20; }
    .stButton>button { width: 100%; }
</style>
""", unsafe_allow_html=True)

# 缓存函数
@st.cache_resource
def load_and_train_model():
    # ---------------------------------------------------------
    # PART A: Load Data
    # ---------------------------------------------------------
    cols_continuous = [
        "Elevation", "Aspect", "Slope",
        "Horizontal_Distance_To_Hydrology", "Vertical_Distance_To_Hydrology",
        "Horizontal_Distance_To_Roadways", "Horizontal_Distance_To_Fire_Points",
        "Hillshade_9am", "Hillshade_Noon", "Hillshade_3pm"
    ]
    cols_wilderness = [f"Wilderness_Area{i}" for i in range(1, 5)]
    cols_soil = [f"Soil_Type{i}" for i in range(1, 41)]
    all_feature_names = cols_continuous + cols_wilderness + cols_soil

    try:
        df = pd.read_csv("covtype.csv", nrows=10000)
    except FileNotFoundError:
        st.warning("⚠️ 未找到 covtype.csv，正在使用【模拟数据】模式运行。")
        np.random.seed(42)
        n_samples = 2000
        data_sim = {}
        for col in cols_continuous:
            data_sim[col] = np.random.rand(n_samples) * 100 
        for col in cols_wilderness + cols_soil:
            data_sim[col] = np.random.randint(0, 2, n_samples)
        df = pd.DataFrame(data_sim)
        df = df[all_feature_names]
        df['Cover_Type'] = np.random.randint(1, 8, n_samples)
        
    X = df.drop(columns=["Cover_Type"], errors='ignore')
    y = df["Cover_Type"]
    X = X[all_feature_names]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # ---------------------------------------------------------
    # PART B: Train Model
    # ---------------------------------------------------------
    model = RandomForestClassifier(
        n_estimators=100, 
        random_state=42,
        n_jobs=-1,
        class_weight="balanced"
    )
    model.fit(X_train, y_train)

    # ---------------------------------------------------------
    # PART C: Setup Explainer & DiCE
    # ---------------------------------------------------------
    explainer = shap.TreeExplainer(model)
    valid_continuous = [c for c in cols_continuous if c in X_train.columns]

    d = dice_ml.Data(
        dataframe=pd.concat([X_train, y_train], axis=1),
        continuous_features=valid_continuous,
        outcome_name="Cover_Type"
    )
    
    m = dice_ml.Model(model=model, backend="sklearn", model_type="classifier")
    dice_exp = dice_ml.Dice(d, m, method="random")

    return model, explainer, dice_exp, X_train, valid_continuous

with st.spinner('System Initializing...'):
    model, explainer, dice_exp, X_train, continuous_cols = load_and_train_model()

# ==========================================
# 2. Sidebar: Inputs
# ==========================================
st.sidebar.header("📍 Feature Input")
feature_names_ref = X_train.columns.tolist()

def input_feature(label, default, min_v, max_v):
    return st.sidebar.slider(label, min_v, max_v, default)

elevation = input_feature("Elevation", 3210, 1800, 4000)
aspect = input_feature("Aspect", 192, 0, 360)
slope = input_feature("Slope", 10, 0, 60)
h_hydro = input_feature("Horz. Dist to Hydro", 90, 0, 1500)
v_hydro = input_feature("Vert. Dist to Hydro", 10, -200, 600)
road = input_feature("Horz. Dist to Road", 3144, 0, 7000)
fire = input_feature("Horz. Dist to Fire", 339, 0, 7000)
shade9 = input_feature("Hillshade 9am", 219, 0, 255)
shade12 = input_feature("Hillshade Noon", 248, 0, 255)
shade3 = input_feature("Hillshade 3pm", 162, 0, 255)

st.sidebar.markdown("---")
soil_options = [f"Soil_Type{i}" for i in range(1, 41)]
selected_soil = st.sidebar.selectbox("Soil Type", soil_options, index=28)
wilderness_options = [f"Wilderness_Area{i}" for i in range(1, 5)]
selected_wild = st.sidebar.selectbox("Wilderness Area", wilderness_options, index=0)

input_data = {}
for col in feature_names_ref: input_data[col] = 0
input_data['Elevation'] = elevation
input_data['Aspect'] = aspect
input_data['Slope'] = slope
input_data['Horizontal_Distance_To_Hydrology'] = h_hydro
input_data['Vertical_Distance_To_Hydrology'] = v_hydro
input_data['Horizontal_Distance_To_Roadways'] = road
input_data['Horizontal_Distance_To_Fire_Points'] = fire
input_data['Hillshade_9am'] = shade9
input_data['Hillshade_Noon'] = shade12
input_data['Hillshade_3pm'] = shade3
if selected_soil in input_data: input_data[selected_soil] = 1
if selected_wild in input_data: input_data[selected_wild] = 1

query_df = pd.DataFrame([input_data])
query_df = query_df[feature_names_ref] 

# ==========================================
# 3. Main Dashboard
# ==========================================
st.markdown('<div class="main-header">🌲 Forest Cover Type XAI Dashboard</div>', unsafe_allow_html=True)

# 定义所有类别名称
class_names = {
    1: "Spruce/Fir", 
    2: "Lodgepole Pine", 
    3: "Ponderosa Pine", 
    4: "Cottonwood/Willow", 
    5: "Aspen", 
    6: "Douglas-fir", 
    7: "Krummholz"
}

# Section 1: Prediction
col1, col2 = st.columns([1, 2])

with col1:
    st.markdown("### 1. Prediction")
    prediction = model.predict(query_df)[0]
    probs = model.predict_proba(query_df)[0]
    confidence = np.max(probs)
    
    pred_name = class_names.get(prediction, f"Type {prediction}")

    st.metric("Predicted Class", f"{pred_name}")
    st.metric("Confidence", f"{confidence*100:.1f}%")
    
    if prediction == 1:
        st.info("ℹ️ High Elevation Zone")
    elif prediction == 2:
        st.warning("ℹ️ Fire Risk: High")

# Section 2: SHAP
with col2:
    st.markdown("### 2. Explanation (SHAP)")
    shap_values = explainer(query_df)
    class_idx = int(prediction) - 1
    
    fig, ax = plt.subplots(figsize=(8, 4))
    shap.plots.waterfall(shap_values[0, :, class_idx], show=False, max_display=7)
    st.pyplot(fig)

st.markdown("---")

# ==========================================
# 3. Tactical Planning (REPLACEMENT BLOCK)
# ==========================================
st.markdown("### 3. Tactical Planning & Restoration Scenarios")

# 创建选项卡
tab1, tab2 = st.tabs(["🎯 Intelligent Fire Risk Assessment", "🧪 What-If Simulation"])

# ------------------------------------------
# Tab 1: 智能火险管理 (Tab 1) - 修正版
# ------------------------------------------
with tab1:
    st.markdown("**Module:** 🔥 Risk Assessment & Mitigation")
    st.caption("Based on predicted species and infrastructure.")

    col_risk1, col_risk2 = st.columns([1, 1], gap="large")

    # 定义火险等级
    fire_risk_map = {
        1: {"level": "High", "color": "inverse"},       # Spruce/Fir
        2: {"level": "Critical", "color": "inverse"},   # Lodgepole Pine
        3: {"level": "Medium", "color": "off"},         # Ponderosa Pine
        4: {"level": "Low", "color": "normal"},         # Cottonwood/Willow
        5: {"level": "Low", "color": "normal"},         # Aspen
        6: {"level": "Medium", "color": "off"},         # Douglas-fir
        7: {"level": "High", "color": "inverse"}        # Krummholz
    }

    # 获取风险信息
    current_risk_info = fire_risk_map.get(int(prediction), {"level": "Unknown", "color": "off"})
    risk_level = current_risk_info["level"]

    with col_risk1:
        st.write("#### ⚠️ Risk Diagnosis")
        
        # 修正：删除了不支持的 key 参数
        st.metric(
            label="Species Fire Susceptibility",
            value=f"{risk_level} Risk",
            delta=f"Species: {pred_name}",
            delta_color=current_risk_info["color"]
        )
        
        infra_status = []
        
        # 基础设施检查
        if h_hydro > 500:
            st.error(f"❌ **Water Access:** Poor ({h_hydro}m away)")
            infra_status.append("Water")
        else:
            st.success(f"✅ **Water Access:** Good ({h_hydro}m away)")
            
        if road > 1000:
            st.error(f"❌ **Emergency Road:** Poor ({road}m away)")
            infra_status.append("Road")
        else:
            st.success(f"✅ **Emergency Road:** Good ({road}m away)")

    with col_risk2:
        st.write("#### 🛡️ AI Recommendations")
        
        if risk_level in ["High", "Critical"]:
            st.warning(f"Detected **{pred_name}** (High Fuel Load). Mitigation recommended.")
            
            suggestions = []
            if "Water" in infra_status:
                suggestions.append(f"💧 **Construct Fire Canal:** Reduce distance to < 300m.")
            if "Road" in infra_status:
                suggestions.append(f"🛣️ **Extend Access Road:** Reduce distance to < 500m.")
            
            if not suggestions:
                st.info("✅ Infrastructure is adequate.")
            else:
                for s in suggestions:
                    st.markdown(s)
        else:
            st.success(f"**{pred_name}** is Low/Medium risk. Standard monitoring sufficient.")
            st.caption("No major interventions needed.")

# ------------------------------------------
# Tab 2: 正向模拟 (Tab 2)
# ------------------------------------------
with tab2:
    st.markdown("**Scenario:** Forecast ecological impact of infrastructure changes.")
    
    col_sim1, col_sim2 = st.columns([1, 1], gap="medium")
    
    with col_sim1:
        st.write("#### 🛠️ Intervention Settings")
        
        # 获取当前值作为默认值
        cur_h = int(query_df['Horizontal_Distance_To_Hydrology'].values[0])
        cur_r = int(query_df['Horizontal_Distance_To_Roadways'].values[0])
        cur_f = int(query_df['Horizontal_Distance_To_Fire_Points'].values[0])

        new_hydro = st.slider("New Dist. to Hydro (m)", 0, 1500, cur_h, key="sim_hydro")
        new_road = st.slider("New Dist. to Road (m)", 0, 7000, cur_r, key="sim_road")
        new_fire = st.slider("New Dist. to Fire (m)", 0, 7000, cur_f, key="sim_fire")
        
    with col_sim2:
        st.write("#### 🔮 Forecasted Outcome")
        
        sim_data = query_df.copy()
        sim_data['Horizontal_Distance_To_Hydrology'] = new_hydro
        sim_data['Horizontal_Distance_To_Roadways'] = new_road
        sim_data['Horizontal_Distance_To_Fire_Points'] = new_fire
        
        new_pred = model.predict(sim_data)[0]
        new_probs = model.predict_proba(sim_data)[0]
        
        delta_color = "normal" if new_pred == prediction else "inverse"
        
        st.metric(
            label="Projected Vegetation Type",
            value=f"{class_names[new_pred]}",
            delta=f"From: {pred_name}",
            delta_color=delta_color,
            key="metric_sim_outcome"
        )
        
        prob_df = pd.DataFrame({
            "Species": list(class_names.values()),
            "Probability": new_probs
        })
        st.bar_chart(prob_df.set_index("Species"), color="#2E7D32")

st.markdown("---")




