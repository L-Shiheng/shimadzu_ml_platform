import streamlit as st

# --- 头部标题区 ---
st.title("📊 预测模型训练与应用平台")
st.markdown("**(零代码机器学习与数据预测工作站)**")

st.write("---")

# --- 平台简介 (专业平实版) ---
st.markdown("""
### 💡 平台简介
本平台是一个专为结构化表格数据设计的预测建模工具，旨在协助您从多维变量中挖掘核心特征，并构建可靠的分类预测模型。

无论您的数据来源于科学研究、工业质控还是业务分析，只要拥有一份包含“特征指标”与“目标结果”的数据表（支持 Excel / CSV 格式），系统即可为您训练出专属的机器学习模型。

**您无需具备深厚的编程与算法基础。** 通过直观的界面交互，即可高效完成模型训练、效能验证与解释分析，并直接导出符合学术发表及专业报告标准的可视化图表。
""")

# --- 适用场景 (泛化版) ---
st.info("""
**🌐 典型应用场景：**
*   **科研探索**：挖掘复杂数据中的关键变量，为实验假设提供统计学与算法层面的客观支撑。
*   **工业与生产质控**：基于多维检测参数监控产品状态，实现异常预警与关键影响因素溯源。
*   **业务分析与决策**：通过历史数据构建分类模型，辅助自动化的风险评估与状态预测。
""")

st.write("---")

# --- 核心模块导航 (操作指南版) ---
st.markdown("### 🚦 它是如何工作的？(操作指南)")

col1, col2 = st.columns(2)

with col1:
    st.success("#### 🎓 模块一：【模型训练】")
    st.markdown("""
    **请在左侧菜单点击进入 `模型训练` 页面：**
    1.  **导入数据**：上传包含已知分类结果的历史数据表。
    2.  **配置参数**：选择算法引擎（系统已配置稳健的默认参数，您也可根据需求手动微调防过拟合策略）。
    3.  **效能评估**：系统将自动执行训练，并输出准确率、ROC 曲线等全方位的交叉验证评估报告。
    4.  **模型导出**：系统会提取核心特征权重，并允许您将训练完毕的“模型对象（.pkl 文件）”下载保存。
    """)

with col2:
    st.warning("#### 🔍 模块二：【模型应用】")
    st.markdown("""
    **请在左侧菜单点击进入 `模型应用` 页面：**
    1.  **载入模型**：上传您在第一步下载的“模型对象（.pkl 文件）”。
    2.  **导入新数据**：上传一份全新的、待预测的测试数据表。
    3.  **自动预测**：系统将立即输出每个新样本的分类结果及其对应的精确概率。
    4.  **特征归因 (SHAP)**：通过内置的解释算法，生成全局分析图与单样本瀑布图，清晰展示模型做出判断的具体依据。
    """)

st.write("---")

# --- 底部技术支持说明 ---
st.markdown("""
> **💡 底层技术说明**
> *   本平台核心算法基于主流的 `scikit-learn` 和 `XGBoost` 机器学习框架构建。
> *   内置的数据标准化、交叉验证（CV）以及 SHAP 特征归因分析流程，完全符合主流学术期刊对于机器学习方法学的严谨性要求。
""")
# ==========================================
# 👁️ 隐藏式后台：跨应用日志可视化看板
# ==========================================
# st.expander 默认 expanded=False，它就像一个折叠条，不点开根本不显眼
with st.expander("🛠️ 系统级运行日志看板 (仅管理员查阅)", expanded=False):
    try:
        import pandas as pd
        import io
        from github import Github
        import streamlit as st
        
        # 1. 安全获取密码并连接仓库
        # （这里用的是你刚刚存进 secrets 里的同一把钥匙）
        GITHUB_TOKEN = st.secrets["GITHUB_TOKEN"] 
        REPO_NAME = "L-Shiheng/Central_App_Logs"
        
        g = Github(GITHUB_TOKEN)
        repo = g.get_repo(REPO_NAME)
        
        st.caption("以下数据实时拉取自中心化 GitHub 云端仓库")
        
        # 2. 交互式选择要查看的 App (为了以后扩展多 App 准备)
        # 获取仓库里所有 csv 文件
        files = [f.path for f in repo.get_contents("") if f.path.endswith('.csv')]
        
        if not files:
            st.info("📭 中心仓库目前是空的，还没有任何 App 产生日志。")
        else:
            # 建立一个下拉菜单，如果你有多个 App，就可以随意切换看数据
            selected_file = st.selectbox("📂 选择要查看的应用日志:", files)
            
            # 3. 拉取选中文件的数据
            contents = repo.get_contents(selected_file)
            df_log = pd.read_csv(io.StringIO(contents.decoded_content.decode('utf-8')))
            
            # 4. 展示原始表格 (限制高度，防止占用太多屏幕)
            st.dataframe(df_log, use_container_width=True, height=200)
            
            # 5. 可视化分析
            if not df_log.empty:
                col1, col2 = st.columns(2)
                with col1:
                    st.caption(f"📊 {selected_file.split('_')[0]} - 模型调用频次")
                    # 统计模型使用次数并画柱状图
                    if '模型' in df_log.columns:
                        model_counts = df_log['模型'].value_counts()
                        st.bar_chart(model_counts)
                    else:
                        st.write("暂无模型调用数据")
                    
                with col2:
                    st.caption("📈 近期平台活跃趋势")
                    # 按天统计使用次数
                    if '时间' in df_log.columns:
                        df_log['日期'] = pd.to_datetime(df_log['时间']).dt.date
                        date_counts = df_log['日期'].value_counts().sort_index()
                        st.line_chart(date_counts)
                    else:
                        st.write("暂无时间分布数据")
                        
    except Exception as e:
        # 即使这里报错了，也只会显示在这个折叠面板里，绝不影响网页其他部分的正常运行
        st.warning(f"获取日志或渲染面板时发生异常，但不影响系统主功能。\n错误详情: {e}")
