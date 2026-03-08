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
    cloud_path = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
    if os.path.exists(cloud_path):
        return cloud_path
    
    try:
        result = subprocess.run(['fc-list', ':lang=zh', 'file'], capture_output=True, text=True)
        if result.stdout:
            return result.stdout.strip().split('\n')[0].split(':')[0].strip()
    except:
        pass
        
    mac_path = '/System/Library/Fonts/Supplemental/Songti.ttc'
    if os.path.exists(mac_path):
        return mac_path
        
    return None

font_path = get_font_properties()
if font_path:
    label_font = fm.FontProperties(fname=font_path, size=12, weight='bold')
    title_font = fm.FontProperties(fname=font_path, size=15, weight='bold')
    donut_label_font = fm.FontProperties(fname=font_path, size=10, weight='bold') # 甜甜圈图专用小字体
else:
    label_font = fm.FontProperties(size=12, weight='bold')
    title_font = fm.FontProperties(size=15, weight='bold')
    donut_label_font = fm.FontProperties(size=10, weight='bold')

plt.rcParams['axes.unicode_minus'] = False 
plt.rcParams['figure.dpi'] = 300 

# ---------------------------------------------------------
# 3. 数据训练：剥离偏见，基于行为聚类
# ---------------------------------------------------------
@st.cache_data
def load_and_train():
    df = pd.read_csv('student-mat.csv')
    
    cluster_features = ['famrel', 'goout', 'Dalc', 'absences']
    df_selected = df[['Pstatus'] + cluster_features].copy()
    df_selected['Pstatus'] = df_selected['Pstatus'].map({'T': 1, 'A': 0})
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_selected[cluster_features])
    
    kmeans = KMeans(n_clusters=3, init='k-means++', random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X_scaled)
    
    temp_df = pd.DataFrame({'cluster': clusters, 'absences': df_selected['absences']})
    order = temp_df.groupby('cluster')['absences'].mean().sort_values().index.tolist()
    rank_map = {original: new for new, original in enumerate(order)}
    
    df_selected['Risk_Cluster'] = pd.Series(clusters).map(rank_map)
    cluster_centers = df_selected.groupby('Risk_Cluster').mean()
    
    return df_selected, scaler, cluster_centers, cluster_features

df_final, scaler, cluster_centers, cluster_features = load_and_train()

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
# 5. 主界面布局 (微观个体分析)
# ---------------------------------------------------------
st.title("⚖️ 未成年人犯罪风险“分级预防”辅助决策系统")
st.markdown("---")

input_data = pd.DataFrame([[
    1 if p_status == "同居 (T)" else 0,
    fam_rel, go_out, dalc, absences
]], columns=['Pstatus', 'famrel', 'goout', 'Dalc', 'absences'])

input_cluster_data = input_data[cluster_features]
input_scaled = scaler.transform(input_cluster_data)
centers_scaled = scaler.transform(cluster_centers[cluster_features])

distances = np.linalg.norm(centers_scaled - input_scaled, axis=1)
predicted_cluster = np.argmin(distances)

risk_info = {
    0: {"name": "低脆弱度：原生支撑稳固群落", "color": "#3498db", "desc": "个体行为与家庭支撑稳健，建议保持常规普法教育。"},
    1: {"name": "中脆弱度：结构性失稳群落", "color": "#f39c12", "desc": "【系统提示】：该群落个体存在家庭关系受损或轻度行为越轨，建议社工介入提供『保护性关爱』，防范潜在环境诱导。"},
    2: {"name": "高脆弱度：环境诱导脱轨群落", "color": "#e74c3c", "desc": "【高危预警】：表现出严重的酗酒与逃学等实质脱轨行为！需立即核实个体行为轨迹，启动多部门联动干预。"}
}
risk_map = {0: "低脆弱度：原生支撑稳固群落", 1: "中脆弱度：结构性失稳群落", 2: "高脆弱度：环境诱导脱轨群落"}

col1, col2 = st.columns([1, 2])

with col1:
    st.metric("微观个体评价", risk_info[predicted_cluster]["name"])
    st.error(f"**干预建议**：{risk_info[predicted_cluster]['desc']}")
    
    if predicted_cluster == 0 and p_status == "分居 (A)":
        st.warning("💡 **辅助提示**：该个体当前行为表现良好，但存在【父母分居】的单亲背景。系统并未对其进行风险定罪，但建议学校给予常规的心理支持，防患于未然。")
    
    st.write("**当前个体特征向量：**")
    st.table(input_data.rename(columns={'Pstatus':'父母同居','famrel':'家庭关系','goout':'外出频率','Dalc':'饮酒量','absences':'缺勤'}))

with col2:
    labels = ['父母同居', '家庭关系', '外出频率', '饮酒量', '缺勤']
    plot_vals = input_data.values[0].tolist()
    plot_vals[4] = min(plot_vals[4], 16) 
    
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    angles += angles[:1]
    plot_vals += plot_vals[:1]

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    ax.plot(angles, plot_vals, color=risk_info[predicted_cluster]["color"], linewidth=3, marker='o')
    ax.fill(angles, plot_vals, color=risk_info[predicted_cluster]["color"], alpha=0.3)
    
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_thetagrids(np.degrees(angles[:-1]), labels, fontproperties=label_font)
    ax.set_ylim(0, 16)
    plt.title("个体综合风险画像雷达图", fontproperties=title_font, pad=20)
    st.pyplot(fig)

# ---------------------------------------------------------
# 6. 宏观大盘数据展示 (动态计算并生成甜甜圈图)
# ---------------------------------------------------------
st.markdown("---")
st.subheader("📊 宏观治理视角：大盘常模与司法资源配置")

col3, col4 = st.columns([1.2, 1])

with col3:
    st.write("**各群落核心特征常模数据（算法决策基准）：**")
    display_df = cluster_centers.rename(index=risk_map).rename(columns={
        'Pstatus':'父母同居概率','famrel':'家庭关系','goout':'外出频率','Dalc':'饮酒量','absences':'缺勤'
    }).round(2)
    st.dataframe(display_df, use_container_width=True)
    st.info("📌 **法理学洞察**：注意观察高脆弱度群落，其破坏性行为（缺勤与饮酒）极高，但家庭关系（famrel）却优于中脆弱度群体，完美印证了‘差异交往理论’中的环境诱导属性。")

with col4:
    # 动态获取当前全量样本的人数分布
    cluster_counts = df_final['Risk_Cluster'].value_counts().sort_index()
    sizes = cluster_counts.values.tolist()
    total = sum(sizes)
    
    donut_labels = [
        f'低脆弱度 ({sizes[0]/total*100:.1f}%)\n【一般预防】', 
        f'中脆弱度 ({sizes[1]/total*100:.1f}%)\n【不良干预】', 
        f'高脆弱度 ({sizes[2]/total*100:.1f}%)\n【严重矫治】'
    ]
    colors = ['#3498db', '#f39c12', '#e74c3c']
    explode = (0.02, 0.02, 0.08) # 突出高危群体

    fig_donut, ax_donut = plt.subplots(figsize=(6, 6))
    wedges, texts, autotexts = ax_donut.pie(
        sizes, explode=explode, labels=donut_labels, colors=colors,
        autopct='%1.1f%%', pctdistance=0.75, startangle=140,
        textprops={'fontproperties': donut_label_font}
    )
    
    # 挖空白心，做成高级甜甜圈图
    centre_circle = plt.Circle((0,0), 0.55, fc='white')
    fig_donut.gca().add_artist(centre_circle)
    
    plt.title('全量样本司法行政资源建议分配比', fontproperties=title_font, pad=15)
    st.pyplot(fig_donut)

