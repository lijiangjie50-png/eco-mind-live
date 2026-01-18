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
    # 1. 定义真实的列名结构 (防止模拟数据列名对不上)
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
        # 尝试读取 CSV (只读前 10000 行以节省内存)
        df = pd.read_csv("covtype.csv", nrows=10000)
    except FileNotFoundError:
        # -----------------------------------------------------
        # 关键修复：模拟数据必须使用和真实数据一样的列名！
        # -----------------------------------------------------
        st.warning("⚠️ 未找到 covtype.csv，正在使用【模拟数据】模式运行。")
        np.random.seed(42)
        n_samples = 2000
        
        # 生成模拟数据字典
        data_sim = {}
        for col in cols_continuous:
            data_sim[col] = np.random.rand(n_samples) * 100 # 随机值
        for col in cols_wilderness + cols_soil:
            data_sim[col] = np.random.randint(0, 2, n_samples) # 0或1
            
        df = pd.DataFrame(data_sim)
        # 确保列顺序一致
        df = df[all_feature_names]
        # 生成目标变量
        df['Cover_Type'] = np.random.randint(1, 8, n_samples)
        
    X = df.drop(columns=["Cover_Type"], errors='ignore')
    y = df["Cover_Type"]
    
    # 确保 X 只包含我们定义的特征列
    X = X[all_feature_names]

    # Stratify split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # ---------------------------------------------------------
    # PART B: Train Model
    # ---------------------------------------------------------
    model = RandomForestClassifier(
        n_estimators=100, # 稍微减小一点以防超时
        random_state=42,
        n_jobs=-1,
        class_weight="balanced"
    )
    model.fit(X_train, y_train)

    # ---------------------------------------------------------
    # PART C: Setup Explainer & DiCE
    # ---------------------------------------------------------
    explainer = shap.TreeExplainer(model)

    # DiCE Setup
    valid_continuous = [c for c in cols_continuous if c in X_train.columns]

    d = dice_ml.Data(
        dataframe=pd.concat([X_train, y_train], axis=1),
        continuous_features=valid_continuous,
        outcome_name="Cover_Type"
    )
    
    m = dice_ml.Model(model=model, backend="sklearn", model_type="classifier")
    dice_exp = dice_ml.Dice(d, m, method="random")

    return model, explainer, dice_exp, X_train, valid_continuous

# 加载模型
with st.spinner('System Initializing... (Training Model & Loading XAI Engine)'):
    model, explainer, dice_exp, X_train, continuous_cols = load_and_train_model()

# ==========================================
# 2. Sidebar: Inputs
# ==========================================
st.sidebar.header("📍 Feature Input")

# 获取训练数据的列名结构 (这是标准答案)
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

# 构建输入向量
input_data = {}
# 先全部填 0
for col in feature_names_ref:
    input_data[col] = 0

# 填入滑块的值
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

# 填入下拉菜单的值 (One-Hot)
if selected_soil in input_data: input_data[selected_soil] = 1
if selected_wild in input_data: input_data[selected_wild] = 1

# 转为 DataFrame
query_df = pd.DataFrame([input_data])

# ==========================================
# 关键修复 (CRITICAL FIX)
# ==========================================
# 强制让 query_df 的列顺序和名字完全匹配训练时的 X_train
# 这一步会丢弃掉任何多余的列，并自动按照正确顺序排列
query_df = query_df[feature_names_ref] 

# ==========================================
# 3. Main Dashboard
# ==========================================
st.markdown('<div class="main-header">🌲 Forest Cover Type XAI Dashboard</div>', unsafe_allow_html=True)

# Section 1: Prediction
col1, col2 = st.columns([1, 2])

with col1:
    st.markdown("### 1. Prediction")
    prediction = model.predict(query_df)[0]
    probs = model.predict_proba(query_df)[0]
    confidence = np.max(probs)
    
    class_names = {1: "Spruce/Fir", 2: "Lodgepole Pine", 3: "Ponderosa Pine", 
                   4: "Cottonwood/Willow", 5: "Aspen", 6: "Douglas-fir", 7: "Krummholz"}
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
    
    # 画图
    fig, ax = plt.subplots(figsize=(8, 4))
    # 注意：使用当前预测类的 SHAP 值
    shap.plots.waterfall(shap_values[0, :, class_idx], show=False, max_display=7)
    st.pyplot(fig)

st.markdown("---")

# Section 3: DiCE
st.markdown("### 3. Actionable Insights (DiCE)")
st.write(f"Scenario: How to change from **{pred_name}** to another type?")

target_class = st.selectbox("Select Target Class:", list(class_names.keys()), index=2) # Default Type 3

if st.button("Generate Counterfactuals"):
    with st.spinner("Calculating..."):
        try:
            cf = dice_exp.generate_counterfactuals(
                query_df,
                total_CFs=3,
                desired_class=int(target_class),
                features_to_vary=continuous_cols 
            )
            cf_df = cf.visualize_as_dataframe(show_only_changes=False)
            st.dataframe(cf_df)
        except Exception as e:
            st.error(f"Could not generate counterfactuals: {e}")
