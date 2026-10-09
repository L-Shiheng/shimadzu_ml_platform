import streamlit as st

# --- 页面配置 ---
st.set_page_config(
    page_title="多维数据预测建模与分析平台",
    page_icon="📊",
    layout="wide"
)

# --- 头部标题区 ---
st.title("📊 多维数据预测建模与解释分析平台")
st.markdown("**(Multi-dimensional Data Predictive Modeling & Interpretability Analysis Platform)**")

st.write("---")

# --- 平台简介 ---
st.markdown("""
### 💡 平台简介
本平台是一个专为**结构化特征矩阵（Structured Feature Matrix）**设计的端到端机器学习建模工作站。
无论您是处理**临床组学数据、食品风味色谱数据、还是工业理化检测指标**，平台都能为您提供从数据预处理、算法训练、效能评估到底层特征归因解释的完整科研工作流。

通过高度封装的算法接口，平台旨在帮助领域专家（Domain Experts）跨越编程门槛，直接生成可用于学术发表与实际业务评估的分析报告与可视化图表。
""")

# --- 适用场景 (强调多领域通用) ---
st.info("""
**🌐 典型应用场景：**
*   **临床与生命科学**：基于代谢物、基因表达量预测疾病表型，挖掘关键生物标志物 (Biomarkers)。
*   **食品与感官科学**：基于气相/液相色谱 (GC/LC-MS) 峰面积数据，对食品产地、风味等级或掺假情况进行鉴别与关键物质溯源。
*   **环境与理化分析**：基于多维检测指标预测水质等级或土壤异常。
*   **材料与质量控制**：通过生产过程中的多项仪器读数，预测最终产品的合格率。
""")

st.write("---")

# --- 核心模块导航 ---
st.markdown("### 🧩 核心功能模块")

col1, col2 = st.columns(2)

with col1:
    st.success("#### 📁 模块一：模型构建与验证")
    st.markdown("""
    **定位**：模型训练基建与泛化能力评估。
    *   **支持算法**：XGBoost, 随机森林 (Random Forest), 支持向量机 (SVM), 多层感知机 (MLP)。
    *   **核心功能**：
        *   内置数据标准化与缺失值自动插补。
        *   提供 L1/L2 正则化及样本防过拟合微调机制。
        *   严格的 5-Fold 交叉验证 (Cross-Validation) 评估。
        *   **多模型效能对比库 (Benchmark)**：一键聚合生成多算法 ROC 曲线对比图。
        *   导出算法对象 `.pkl` 文件以供下游部署。
    """)

with col2:
    st.warning("#### 📁 模块二：独立验证与解释分析")
    st.markdown("""
    **定位**：外部独立队列盲测与“黑盒”模型机制解析。
    *   **支持分析**：结合已训练的 `.pkl` 模型与全新外部数据执行验证。
    *   **核心功能**：
        *   数据特征字段一致性安全校验。
        *   外部测试集混淆矩阵与受试者工作特征 (ROC) 评估。
        *   连续分类概率输出，辅助边缘样本复核。
        *   **全局特征归因 (Global SHAP)**：生成蜂群图 (Beeswarm Plot)，洞察各特征的整体驱动方向。
        *   **局部特征归因 (Local SHAP)**：生成瀑布图 (Waterfall Plot)，解析单一样本分类的具体推导依据。
    """)

st.write("---")

# --- 底部技术支持说明 ---
st.markdown("""
> **🔧 技术说明与运行环境**
> *   底层算法引擎基于 `scikit-learn` 与 `xgboost` 框架构建。
> *   解释器模块基于 `shap` (SHapley Additive exPlanations) 合作博弈理论。
> *   *建议在执行高维数据（特征 > 5000）的全量 SHAP 解释时，合理控制单次运算的样本批次以保障系统响应速度。*
""")
