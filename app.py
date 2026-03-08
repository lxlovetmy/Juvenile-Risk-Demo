import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.font_manager as fm
import subprocess
import os

# 1. 页面配置
st.set_page_config(page_title="未成年人犯罪风险分级预警系统", layout="wide")

# ---------------------------------------------------------
# 2. 彻底击穿云端缓存的“底层物理寻址法”
# ---------------------------------------------------------
@st.cache_resource
def get_font_properties():
    # 路径 1: 精准狙击 Streamlit Cloud 上 Noto Sans CJK 的物理位置
    cloud_path = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
    if os.path.exists(cloud_path):
        return cloud_path
    
    # 路径 2: 如果路径不对，调用 Linux 终端命令暴力搜索中文字体
    try:
        result = subprocess.run(['fc-list', ':lang=zh', 'file'], capture_output=True, text=True)
        if result.stdout:
            # 拿到第一个找到的中文字体路径
            return result.stdout.strip().split('\n')[0].split(':')[0].strip()
    except:
        pass
        
    # 路径 3: 本地 Mac 调试时的备用路径
    mac_path = '/System/Library/Fonts/Supplemental/Songti.ttc'
    if os.path.exists(mac_path):
        return mac_path
        
    return None

# 获取物理路径并生成专用字体对象
font_path = get_font_properties()
if font_path:
    label_font = fm.FontProperties(fname=font_path, size=12, weight='bold')
    title_font = fm.FontProperties(fname=font_path, size=15, weight='bold')
else:
    label_font = fm.FontProperties(size=12, weight='bold')
    title_font = fm.FontProperties(size=15, weight='bold')

plt.rcParams['axes.unicode_minus'] = False 
plt.rcParams['figure.dpi'] = 300 

# ---------------------------------------------------------
# ---------------------------------------------------------
# 3. 真正科学的数据训练：剥离背景标签，基于行为与环境聚类
# ---------------------------------------------------------
@st.cache_data
def load_and_train():
    df = pd.read_csv('student-mat.csv')
    # 【核心修复】：把 Pstatus 从聚类特征中踢出去！
    # 风险只由家庭关系的恶劣程度和个人的越轨行为决定
    cluster_features = ['famrel', 'goout', 'Dalc', 'absences']
    
    df_selected = df[['Pstatus'] + cluster_features].copy()
    df_selected['Pstatus'] = df_selected['Pstatus'].map({'T': 1, 'A': 0})
    
    # 只对四个行为/关系特征进行标准化和聚类
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_selected[cluster_features])
    
    kmeans = KMeans(n_clusters=3, init='k-means++', random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X_scaled)
    
    # 依然用最硬核的破坏性指标（缺勤）来给 0, 1, 2 排序
    temp_df = pd.DataFrame({'cluster': clusters, 'absences': df_selected['absences']})
    order = temp_df.groupby('cluster')['absences'].mean().sort_values().index.tolist()
    rank_map = {original: new for new, original in enumerate(order)}
    
    df_selected['Risk_Cluster'] = pd.Series(clusters).map(rank_map)
    
    # 计算三个群落的常模（展示时依然带上 Pstatus，看看各群落的父母分居比例）
    cluster_centers = df_selected.groupby('Risk_Cluster').mean()
    
    return df_selected, scaler, cluster_centers, cluster_features
# ---------------------------------------------------------
# 4. 侧边栏：输入个体特征
# ---------------------------------------------------------
st.sidebar.header("🔍 个体风险因子输入")
st.sidebar.markdown("---")

st.sidebar.subheader("核心静态背景")
p_status = st.sidebar.selectbox("父母同居状态", ["同居 (T)", "分居 (A)"])
fam_rel = st.sidebar.slider("家庭关系质量 (1-5)", 1, 5, 4)

st.sidebar.subheader("动态行为表现")
go_out = st.sidebar.slider("课后会友频率 (1-5)", 1, 5, 3)
dalc = st.sidebar.slider("工作日饮酒严重度 (1-5)", 1, 5, 1)
absences = st.sidebar.number_input("近期缺勤/逃学次数", 0, 100, 5)

# ---------------------------------------------------------
# 5. 主界面布局
# ---------------------------------------------------------
st.title("⚖️ 未成年人犯罪风险“分级预防”辅助决策系统")
st.markdown("---")

# 执行预测
input_data = pd.DataFrame([[
    1 if p_status == "同居 (T)" else 0,
    fam_rel, go_out, dalc, absences
]], columns=['Pstatus', 'famrel', 'goout', 'Dalc', 'absences'])

input_scaled = scaler.transform(input_data)
centers_scaled = scaler.transform(cluster_centers)
distances = np.linalg.norm(centers_scaled - input_scaled, axis=1)
predicted_cluster = np.argmin(distances)

# 风险定义映射
risk_info = {
    0: {"name": "低风险：偶发冲动型", "color": "#3498db", "desc": "个体社会纽带稳固，建议以家庭教育和校内观察为主。"},
    1: {"name": "中风险：环境诱导脱轨型", "color": "#f39c12", "desc": "受不良同伴诱导产生实质越轨行为，建议切断负面同伴接触。"},
    2: {"name": "高风险：家庭结构失衡型", "color": "#e74c3c", "desc": "核心监护缺失导致的深层危机，建议立即引入司法社工多部门联动干预。"}
}
risk_map = {0: "低风险：偶发冲动型", 1: "中风险：环境诱导脱轨型", 2: "高风险：家庭结构失衡型"}

col1, col2 = st.columns([1, 2])

with col1:
    st.metric("综合评价结果", risk_info[predicted_cluster]["name"])
    st.error(f"**干预建议**：{risk_info[predicted_cluster]['desc']}")
    
    st.write("**当前个体特征向量：**")
    st.table(input_data.rename(columns={
        'Pstatus':'父母同居','famrel':'家庭关系','goout':'外出频率','Dalc':'饮酒量','absences':'缺勤'
    }))

with col2:
    # 动态生成雷达图
    labels = ['父母同居', '家庭关系', '外出频率', '饮酒量', '缺勤']
    plot_vals = input_data.values[0].tolist()
    plot_vals[4] = min(plot_vals[4], 15) # 缺勤截断保护
    
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    angles += angles[:1]
    plot_vals += plot_vals[:1]

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    ax.plot(angles, plot_vals, color=risk_info[predicted_cluster]["color"], linewidth=3, marker='o')
    ax.fill(angles, plot_vals, color=risk_info[predicted_cluster]["color"], alpha=0.3)
    
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    
    # 【核心：强制物理注入字体对象，避开缓存】
    ax.set_thetagrids(np.degrees(angles[:-1]), labels, fontproperties=label_font)
    ax.set_ylim(0, 16)
    plt.title("个体风险画像雷达图", fontproperties=title_font, pad=20)
    
    st.pyplot(fig)

st.markdown("---")
st.subheader("📊 算法决策依据：风险群落常模对比")
st.dataframe(cluster_centers.rename(index=risk_map).round(2))
