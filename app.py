import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# 1. 页面配置
st.set_page_config(page_title="未成年人犯罪风险分级预警系统", layout="wide")

# 2. 全局字体配置：强制使用云端安装的思源黑体
plt.rcParams['font.sans-serif'] = ['Noto Sans CJK SC', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False 
plt.rcParams['figure.dpi'] = 300 

# 3. 数据加载与模型预训练
@st.cache_data
def load_and_train():
    df = pd.read_csv('student-mat.csv')
    features = ['Pstatus', 'famrel', 'goout', 'Dalc', 'absences']
    df_selected = df[features].copy()
    df_selected['Pstatus'] = df_selected['Pstatus'].map({'T': 1, 'A': 0})
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_selected)
    
    kmeans = KMeans(n_clusters=3, init='k-means++', random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X_scaled)
    
    # 自动排序：确保缺勤最多的组永远是红色高风险
    temp_df = pd.DataFrame({'cluster': clusters, 'absences': df_selected['absences']})
    order = temp_df.groupby('cluster')['absences'].mean().sort_values().index.tolist()
    rank_map = {original: new for new, original in enumerate(order)}
    df_selected['Risk_Cluster'] = pd.Series(clusters).map(rank_map)
    
    cluster_centers = df_selected.groupby('Risk_Cluster').mean()
    return df_selected, scaler, cluster_centers

df_final, scaler, cluster_centers = load_and_train()

# 4. 侧边栏
st.sidebar.header("🔍 个体风险因子输入")
p_status = st.sidebar.selectbox("父母同居状态", ["同居 (T)", "分居 (A)"])
fam_rel = st.sidebar.slider("家庭关系质量 (1-5)", 1, 5, 4)
go_out = st.sidebar.slider("课后会友频率 (1-5)", 1, 5, 3)
dalc = st.sidebar.slider("工作日饮酒严重度 (1-5)", 1, 5, 1)
absences = st.sidebar.number_input("近期缺勤/逃学次数", 0, 100, 5)

# 5. 主界面
st.title("⚖️ 未成年人犯罪风险“分级预防”辅助决策系统")
st.markdown("---")

# 预测逻辑
input_data = pd.DataFrame([[1 if p_status == "同居 (T)" else 0, fam_rel, go_out, dalc, absences]], 
                         columns=['Pstatus', 'famrel', 'goout', 'Dalc', 'absences'])
input_scaled = scaler.transform(input_data)
centers_scaled = scaler.transform(cluster_centers)
distances = np.linalg.norm(centers_scaled - input_scaled, axis=1)
predicted_cluster = np.argmin(distances)

risk_info = {
    0: {"name": "低风险：偶发冲动型", "color": "#3498db", "desc": "建议以家庭教育和校内观察为主。"},
    1: {"name": "中风险：家庭结构失稳型", "color": "#f39c12", "desc": "建议引入司法社工进行家庭教育指导。"},
    2: {"name": "高风险：环境诱导脱轨型", "color": "#e74c3c", "desc": "建议立即触发多部门联动干预。"}
}
risk_map = {0: "低风险：偶发冲动型", 1: "中风险：家庭结构失稳型", 2: "高风险：环境诱导脱轨型"}

col1, col2 = st.columns([1, 2])
with col1:
    st.metric("综合评价结果", risk_info[predicted_cluster]["name"])
    st.error(f"**干预建议**：{risk_info[predicted_cluster]['desc']}")
    st.table(input_data.rename(columns={'Pstatus':'父母同居','famrel':'家庭关系','goout':'外出频率','Dalc':'饮酒量','absences':'缺勤'}))

with col2:
    labels = ['父母同居', '家庭关系', '外出频率', '饮酒量', '缺勤']
    plot_vals = input_data.values[0].tolist()
    plot_vals[4] = min(plot_vals[4], 15) # 截断展示
    
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    angles += angles[:1]; plot_vals += plot_vals[:1]

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    ax.plot(angles, plot_vals, color=risk_info[predicted_cluster]["color"], linewidth=3, marker='o')
    ax.fill(angles, plot_vals, color=risk_info[predicted_cluster]["color"], alpha=0.3)
    ax.set_theta_offset(np.pi / 2); ax.set_theta_direction(-1)
    ax.set_thetagrids(np.degrees(angles[:-1]), labels, fontsize=12, fontweight='bold')
    ax.set_ylim(0, 16)
    plt.title("个体风险画像雷达图", fontsize=15, fontweight='bold', pad=20)
    st.pyplot(fig)

st.markdown("---")
st.subheader("📊 算法决策依据：风险群落常模对比")
st.dataframe(cluster_centers.rename(index=risk_map).round(2))
