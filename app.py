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

# 缓存函数：防止每次刷新网页都重新训练模型
@st.cache_resource
def load_and_train_model():
    # ---------------------------------------------------------
    # PART A: Load Data (Matching your original code)
    # ---------------------------------------------------------
    try:
        # 为了演示速度，这里默认只读取前 15000 行。
        # 如果你想用全量数据 (58万行)，请把 nrows=15000 删掉
        # 注意：Streamlit Cloud 免费版内存可能跑不动 58万行
        df = pd.read_csv("covtype.csv", nrows=15000)
    except FileNotFoundError:
        # 如果没有csv，生成模拟数据防止报错 (备用方案)
        st.error("未找到 covtype.csv，正在使用模拟数据演示...")
        from sklearn.datasets import make_classification
        X_dummy, y_dummy = make_classification(n_samples=2000, n_features=54, n_classes=7, n_informative=10)
        df = pd.DataFrame(X_dummy, columns=[f"Feature_{i}" for i in range(54)])
        df['Cover_Type'] = y_dummy + 1
        
    X = df.drop(columns=["Cover_Type"])
    y = df["Cover_Type"]

    # Stratify split (Exactly as in your code)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # ---------------------------------------------------------
    # PART B: Train Model (Your exact hyperparameters)
    # ---------------------------------------------------------
    model = RandomForestClassifier(
        n_estimators=300,          # Your original param
        random_state=42,           # Your original param
        n_jobs=-1,                 # Your original param
        class_weight="balanced"    # Your original param
    )
    model.fit(X_train, y_train)

    # ---------------------------------------------------------
    # PART C: Setup Explainer & DiCE
    # ---------------------------------------------------------
    # SHAP Explainer
    explainer = shap.TreeExplainer(model)

    # DiCE Setup
    # Extract continuous features (matching your list)
    continuous_features = [
        "Elevation", "Aspect", "Slope",
        "Horizontal_Distance_To_Hydrology", "Vertical_Distance_To_Hydrology",
        "Horizontal_Distance_To_Roadways", "Horizontal_Distance_To_Fire_Points",
        "Hillshade_9am", "Hillshade_Noon", "Hillshade_3pm",
    ]
    
    # Check if these columns actually exist (in case of dummy data)
    valid_continuous = [c for c in continuous_features if c in X_train.columns]

    d = dice_ml.Data(
        dataframe=pd.concat([X_train, y_train], axis=1),
        continuous_features=valid_continuous,
        outcome_name="Cover_Type"
    )
    
    m = dice_ml.Model(model=model, backend="sklearn", model_type="classifier")
    dice_exp = dice_ml.Dice(d, m, method="random")

    return model, explainer, dice_exp, X_train, valid_continuous

# 加载模型 (显示加载条)
with st.spinner('Training Random Forest (n=300) & Initializing XAI engines...'):
    model, explainer, dice_exp, X_train, continuous_cols = load_and_train_model()

# ==========================================
# 2. Sidebar: Inputs (Smart Handling)
# ==========================================
st.sidebar.header("📍 Feature Input")

# 获取训练数据的列名结构
feature_names = X_train.columns.tolist()

# 1. 连续变量输入 (使用你提供的 "Query Instance" 值作为默认值)
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

# 2. 离散变量智能处理 (One-Hot Decoding)
# 将 40 个 Soil_Type 压缩为一个下拉菜单
soil_options = [f"Soil_Type{i}" for i in range(1, 41)]
selected_soil = st.sidebar.selectbox("Soil Type", soil_options, index=28) # Default Type 29 (index 28)

wilderness_options = [f"Wilderness_Area{i}" for i in range(1, 5)]
selected_wild = st.sidebar.selectbox("Wilderness Area", wilderness_options, index=0) # Default Area 1

# 3. 构建输入向量 (Reconstruct One-Hot Vector)
input_data = {}
# 先填满 0
for col in feature_names:
    input_data[col] = 0

# 填入连续值
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

# 填入离散值 (One-Hot)
if selected_soil in input_data: input_data[selected_soil] = 1
if selected_wild in input_data: input_data[selected_wild] = 1

# 转为 DataFrame
query_df = pd.DataFrame([input_data])

# ==========================================
# 3. Main Dashboard
# ==========================================
st.markdown('<div class="main-header">🌲 Forest Cover Type XAI Dashboard</div>', unsafe_allow_html=True)
st.markdown("Interactive analysis based on Random Forest & SHAP/DiCE")

# ----------------------------------
# Section 1: Prediction
# ----------------------------------
col1, col2 = st.columns([1, 2])

with col1:
    st.markdown("### 1. Prediction")
    prediction = model.predict(query_df)[0]
    probs = model.predict_proba(query_df)[0]
    confidence = np.max(probs)
    
    # 类别名称映射
    class_names = {1: "Spruce/Fir", 2: "Lodgepole Pine", 3: "Ponderosa Pine", 
                   4: "Cottonwood/Willow", 5: "Aspen", 6: "Douglas-fir", 7: "Krummholz"}
    pred_name = class_names.get(prediction, f"Type {prediction}")

    st.metric("Predicted Class", f"{pred_name} (Type {prediction})")
    st.metric("Confidence", f"{confidence*100:.1f}%")
    
    if prediction == 1:
        st.info("ℹ️ Note: High Elevation species.")
    elif prediction == 2:
        st.warning("ℹ️ Note: Fire-prone species.")

# ----------------------------------
# Section 2: SHAP Waterfall
# ----------------------------------
with col2:
    st.markdown("### 2. Explanation (SHAP Waterfall)")
    # 计算 SHAP
    shap_values = explainer(query_df)
    
    # 获取对应预测类别的 SHAP (和你的代码逻辑一致)
    # TreeExplainer 对多分类返回 list，需要取对应的 class index
    class_idx = int(prediction) - 1
    
    # 构造 Explanation 对象 (为了画图)
    shap_val_single = shap_values[0, :, class_idx]
    
    fig, ax = plt.subplots(figsize=(8, 4))
    shap.plots.waterfall(shap_val_single, show=False, max_display=7)
    st.pyplot(fig)

st.markdown("---")

# ----------------------------------
# Section 3: DiCE Counterfactuals
# ----------------------------------
st.markdown("### 3. Actionable Insights (DiCE Counterfactuals)")
st.write(f"Generating scenarios to flip prediction from **{pred_name}** to another class...")

# 让用户选择目标类别 (或者自动选择概率第二高的，如你的代码所示)
sorted_indices = np.argsort(probs)[::-1]
# 默认选概率第二高的作为目标
default_target = int(sorted_indices[1]) + 1
target_class = st.selectbox("Select Target Class for Restoration:", list(class_names.keys()), index=list(class_names.keys()).index(default_target))

if st.button("Generate Counterfactuals"):
    with st.spinner("DiCE is calculating minimal changes..."):
        try:
            # DiCE Generation (Matching your code params)
            cf = dice_exp.generate_counterfactuals(
                query_df,
                total_CFs=3,
                desired_class=int(target_class),
                features_to_vary=continuous_cols 
                # 注意：这里只允许改变你代码里定义的 continuous features，SoilType 不会变
            )
            
            # 显示结果
            cf_df = cf.visualize_as_dataframe(show_only_changes=False)
            
            # 高亮变化
            st.dataframe(cf_df.style.apply(lambda x: ['background-color: #d4edda' if x.name != 0 else '' for i in x], axis=1))
            
            st.success("✅ Changes generated! Compare the 'Original' row with the suggestions.")
            
        except Exception as e:
            st.error(f"DiCE calculation failed (possibly due to constraints): {e}")

st.markdown("---")
st.caption("Dashboard Logic mirrors original Jupyter Notebook: RF (n=300, balanced) | Stratified Split | DiCE Random Method")

