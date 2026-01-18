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

# 定义所有类别名称 (供全局使用)
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
# 3. Actionable Insights (Planner View) - REPLACEMENT BLOCK
# ==========================================
st.markdown("### 3. Tactical Planning & Restoration Scenarios(DiCE)")

# 创建两个选项卡：一个给 DiCE（逆向优化），一个给手动模拟（正向推演）
tab1, tab2 = st.tabs(["🎯 Goal-Driven Optimization (DiCE)", "🧪 What-If Simulation (Forward)"])

# ------------------------------------------
# Tab 1: 智能火险管理 (Fire Risk Management - 加固版)
# ------------------------------------------
with tab1:
    st.markdown("**Module:** 🔥 Intelligent Fire Risk Assessment & Mitigation")
    st.caption("Based on the predicted species and current infrastructure, suggest safety interventions.")

    col_risk1, col_risk2 = st.columns([1, 1], gap="large")

    # 1. 定义树种的易燃等级
    fire_risk_map = {
        1: {"level": "High", "color": "inverse"},       # Spruce/Fir
        2: {"level": "Critical", "color": "inverse"},   # Lodgepole Pine
        3: {"level": "Medium", "color": "off"},         # Ponderosa Pine (你现在的预测结果)
        4: {"level": "Low", "color": "normal"},         # Cottonwood/Willow
        5: {"level": "Low", "color": "normal"},         # Aspen
        6: {"level": "Medium", "color": "off"},         # Douglas-fir
        7: {"level": "High", "color": "inverse"}        # Krummholz
    }

    # 获取当前预测的风险信息
    # 注意：Prediction 必须是整数，如果报错 key error，可能是因为 prediction 还没计算出来
    current_risk_info = fire_risk_map.get(int(prediction), {"level": "Unknown", "color": "off"})
    risk_level = current_risk_info["level"]

    with col_risk1:
        st.write("#### ⚠️ Risk Diagnosis")
        
        # 显示当前树种的风险等级
        st.metric(
            label="Species Fire Susceptibility",
            value=f"{risk_level} Risk",
            delta=f"Species: {pred_name}",
            delta_color=current_risk_info["color"],
            key="metric_fire_risk"  # <--- 关键修复：添加唯一 Key
        )
        
        # 基础设施诊断
        infra_status = []
        
        # 检查水源
        if h_hydro > 500:
            st.error(f"❌ **Water Access:** Poor ({h_hydro}m away)")
            infra_status.append("Water")
        else:
            st.success(f"✅ **Water Access:** Good ({h_hydro}m away)")
            
        # 检查道路
        if road > 1000:
            st.error(f"❌ **Emergency Road:** Poor ({road}m away)")
            infra_status.append("Road")
        else:
            st.success(f"✅ **Emergency Road:** Good ({road}m away)")

    with col_risk2:
        st.write("#### 🛡️ AI Recommendations")
        
        # 只有高风险才建议修路
        if risk_level in ["High", "Critical"]:
            st.warning(f"Detected **{pred_name}** (High Fuel Load). Immediate mitigation recommended.")
            
            suggestions = []
            
            if "Water" in infra_status:
                suggestions.append(f"💧 **Construct Fire Canal:** Reduce distance to hydrology to < 300m.")
            
            if "Road" in infra_status:
                suggestions.append(f"🛣️ **Extend Access Road:** Reduce distance to roadways to < 500m.")
            
            if not suggestions:
                st.info("✅ Infrastructure is adequate. Maintain regular monitoring.")
            else:
                for i, s in enumerate(suggestions):
                    st.markdown(s)
        
        # Ponderosa Pine (Type 3) 属于 Medium，会走这里
        else:
            st.success(f"**{pred_name}** has manageable fire resistance. Standard monitoring is sufficient.")
            st.markdown("*No major infrastructure changes required.*")
            # 添加一个占位符，防止布局塌陷
            st.caption(f"Risk Level: {risk_level}")
# ------------------------------------------
# Tab 2: 正向模拟 (What-If 模拟器 - 你的核心需求)
# ------------------------------------------
with tab2:
    st.markdown("**Scenario:** Planner manually adjusts infrastructure to forecast ecological impact.")
    st.caption("Example: *'If I build a canal here (Distance to Hydro = 0), will the forest type change?'*")
    
    col_sim1, col_sim2 = st.columns([1, 1], gap="medium")
    
    with col_sim1:
        st.write("#### 🛠️ Intervention Settings")
        # 这里的滑块独立于左侧 Sidebar，只用于临时模拟
        # 默认值取当前 query_df 的值
        current_hydro = query_df['Horizontal_Distance_To_Hydrology'].values[0]
        current_road = query_df['Horizontal_Distance_To_Roadways'].values[0]
        current_fire = query_df['Horizontal_Distance_To_Fire_Points'].values[0]

        new_hydro = st.slider("New Dist. to Hydro (m)", 0, 1500, int(current_hydro), key="sim_hydro", help="Simulate building water sources")
        new_road = st.slider("New Dist. to Road (m)", 0, 7000, int(current_road), key="sim_road", help="Simulate building/removing roads")
        new_fire = st.slider("New Dist. to Fire (m)", 0, 7000, int(current_fire), key="sim_fire", help="Simulate fire breaks")
        
    with col_sim2:
        st.write("#### 🔮 Forecasted Outcome")
        
        # 1. 构造模拟数据
        sim_data = query_df.copy()
        sim_data['Horizontal_Distance_To_Hydrology'] = new_hydro
        sim_data['Horizontal_Distance_To_Roadways'] = new_road
        sim_data['Horizontal_Distance_To_Fire_Points'] = new_fire
        
        # 2. 重新预测
        new_pred = model.predict(sim_data)[0]
        new_probs = model.predict_proba(sim_data)[0]
        new_pred_name = class_names[new_pred]
        
        # 3. 显示对比结果 (Metirc)
        # 如果预测变了，显示绿色；没变显示灰色
        delta_color = "normal" if new_pred == prediction else "inverse"
        st.metric(
            label="Projected Vegetation Type",
            value=f"{new_pred_name}",
            delta=f"From: {pred_name}",
            delta_color=delta_color
        )
        
        # 4. 显示概率分布变化 (Bar Chart)
        # 用数据框展示概率，让规划师看到微小的概率波动
        prob_df = pd.DataFrame({
            "Species": list(class_names.values()),
            "Probability": new_probs
        })
        st.bar_chart(prob_df.set_index("Species"), color="#2E7D32")

st.markdown("---")


