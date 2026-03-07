import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from matplotlib.font_manager import FontProperties
import os

# 1. 页面配置
st.set_page_config(page_title="未成年人犯罪风险分级预警系统", layout="wide")

# 2. 字体加载逻辑 (针对 M3 Mac 优化)
font_path = '/System/Library/Fonts/Supplemental/Songti.ttc'
if not os.path.exists(font_path):
    font_path = '/System/Library/Fonts/PingFang.ttc'
paper_font = FontProperties(fname=font_path)

# 3. 数据加载与模型预训练 (缓存处理)
@st.cache_data
def load_and_train():
    df = pd.read_csv('student-mat.csv')
    features = ['Pstatus', 'famrel', 'goout', 'Dalc', 'absences']
    df_selected = df[features].copy()
    df_selected['Pstatus'] = df_selected['Pstatus'].map({'T': 1, 'A': 0})
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_selected)
    
    kmeans = KMeans(n_clusters=3, init='k-means++', random_state=42)
    df_selected['Risk_Cluster'] = kmeans.fit_predict(X_scaled)
    
    # 计算三个簇的常模（平均值）
    cluster_centers = df_selected.groupby('Risk_Cluster').mean()
    return df_selected, scaler, kmeans, cluster_centers

df_final, scaler, kmeans, cluster_centers = load_and_train()

# 4. 侧边栏：输入个体特征
st.sidebar.header("🔍 个体风险因子输入")
st.sidebar.markdown("---")

# 静态风险
st.sidebar.subheader("核心静态背景")
p_status = st.sidebar.selectbox("父母同居状态", ["同居 (T)", "分居 (A)"])
fam_rel = st.sidebar.slider("家庭关系质量 (1-5)", 1, 5, 4)

# 动态风险
st.sidebar.subheader("动态行为表现")
go_out = st.sidebar.slider("课后会友频率 (1-5)", 1, 5, 3)
dalc = st.sidebar.slider("工作日饮酒严重度 (1-5)", 1, 5, 1)
absences = st.sidebar.number_input("近期缺勤/逃学次数", 0, 100, 5)

# 5. 主界面布局
st.title("⚖️ 未成年人犯罪风险“分级预防”辅助决策系统")
st.markdown("---")

# 执行预测
input_data = pd.DataFrame([[
    1 if p_status == "同居 (T)" else 0,
    fam_rel, go_out, dalc, absences
]], columns=['Pstatus', 'famrel', 'goout', 'Dalc', 'absences'])

input_scaled = scaler.transform(input_data)
predicted_cluster = kmeans.predict(input_scaled)[0]

# 风险定义映射
risk_info = {
    0: {"name": "低风险：偶发冲动型", "color": "#3498db", "desc": "个体社会纽带稳固，建议以家庭教育和校内观察为主。"},
    1: {"name": "中风险：家庭结构失稳型", "color": "#f39c12", "desc": "由于家庭结构变动导致监护效能下降，建议引入司法社工进行专业家庭教育指导。"},
    2: {"name": "高风险：环境诱导脱轨型", "color": "#e74c3c", "desc": "个体已产生实质性脱轨行为，建议立即触发多部门联动干预，切断负面同伴接触。"}
}
risk_map = {0: "低风险：偶发冲动型", 1: "中风险：家庭结构失稳型", 2: "高风险：环境诱导脱轨型"}
col1, col2 = st.columns([1, 2])

with col1:
    st.metric("综合评价结果", risk_info[predicted_cluster]["name"])
    st.error(f"**干预建议**：{risk_info[predicted_cluster]['desc']}")
    
    # 展示个体数据
    st.write("**当前个体特征向量：**")
    st.table(input_data.rename(columns={
        'Pstatus':'父母同居','famrel':'家庭关系','goout':'外出频率','Dalc':'饮酒量','absences':'缺勤'
    }))

with col2:
    # 动态生成雷达图
    labels = ['父母同居', '家庭关系', '外出频率', '饮酒量', '缺勤']
    # 将输入数据归一化到 0-10 范围以便展示
    plot_vals = input_data.values[0].tolist()
    plot_vals[4] = min(plot_vals[4], 15) # 缺勤次数截断
    
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    angles += angles[:1]
    plot_vals += plot_vals[:1]

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    ax.plot(angles, plot_vals, color=risk_info[predicted_cluster]["color"], linewidth=3, marker='o')
    ax.fill(angles, plot_vals, color=risk_info[predicted_cluster]["color"], alpha=0.3)
    
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_thetagrids(np.degrees(angles[:-1]), labels, fontproperties=paper_font, fontsize=12)
    ax.set_ylim(0, 15)
    plt.title("个体风险画像雷达图", fontproperties=paper_font, size=15)
    st.pyplot(fig)

st.markdown("---")
st.subheader("📊 算法决策依据：风险群落常模对比")
st.write("系统将当前个体数据与 395 例微观样本聚类后的三大“病理模板”进行结构化匹配：")
st.dataframe(cluster_centers.rename(index=risk_map).round(2))