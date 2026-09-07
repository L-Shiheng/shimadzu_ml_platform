import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import cross_val_score
from sklearn.inspection import permutation_importance
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
import joblib
import io

# 尝试导入我们之前写的特征筛选类
try:
    from utils.data_processor import MassSpecFeatureSelector
except ImportError:
    st.error("⚠️ 警告：找不到 utils 文件夹。为了保证运行，已启用内置的特征筛选器。")
    from sklearn.base import BaseEstimator, TransformerMixin
    from sklearn.feature_selection import SelectKBest, SelectFdr, SelectFwe, f_classif
    class MassSpecFeatureSelector(BaseEstimator, TransformerMixin):
        def __init__(self, selection_method='fdr', n_features_to_select=100, alpha=0.05, remove_redundant=True, corr_threshold=0.9):
            self.selection_method = selection_method; self.n_features_to_select = n_features_to_select
            self.alpha = alpha; self.remove_redundant = remove_redundant; self.corr_threshold = corr_threshold
            self.final_indices_ = None
        def fit(self, X, y):
            X_array = X.values if hasattr(X, 'values') else X
            current_indices = np.arange(X_array.shape[1])
            if self.selection_method == 'kbest': selector = SelectKBest(f_classif, k=min(self.n_features_to_select, X_array.shape[1]))
            elif self.selection_method == 'fdr': selector = SelectFdr(f_classif, alpha=self.alpha)
            elif self.selection_method == 'fwe': selector = SelectFwe(f_classif, alpha=self.alpha)
            else:
                self.final_indices_ = current_indices
                return self
            try:
                X_selected = selector.fit_transform(X_array, y)
                current_indices = current_indices[selector.get_support()]
                if self.remove_redundant and X_selected.shape[1] > 1:
                    f_scores, _ = f_classif(X_selected, y)
                    sorted_idx = np.argsort(f_scores)[::-1]
                    selected_subset = []
                    for idx in sorted_idx:
                        if not selected_subset: selected_subset.append(idx)
                        else:
                            corr_vals = np.abs([np.corrcoef(X_selected[:, idx], X_selected[:, sel])[0, 1] for sel in selected_subset])
                            if np.max(corr_vals) <= self.corr_threshold: selected_subset.append(idx)
                    self.final_indices_ = current_indices[selected_subset]
                else: self.final_indices_ = current_indices
            except:
                self.final_indices_ = current_indices
            return self
        def transform(self, X):
            return X.values[:, self.final_indices_] if hasattr(X, 'values') else X[:, self.final_indices_]

# --- 页面初始化与状态管理 ---
st.set_page_config(page_title="模型训练工场", page_icon="🛠️", layout="wide")

for key in ['data_loaded', 'model_trained', 'trained_pipeline', 'df_raw']:
    if key not in st.session_state: st.session_state[key] = None if key in ['trained_pipeline', 'df_raw'] else False

st.title("🛠️ 多引擎组学模型训练工场")
st.markdown("支持多种经典与前沿机器学习引擎，提供专属超参数微调，自动解析核心生物标志物。")

# --- 🌟 新增：数据格式可视化引导指南 ---
with st.expander("📖 上传前必读：标准数据格式指南 (点击展开模板)", expanded=False):
    st.info("💡 **要求：** 数据必须是【行】代表样本，【列】代表代谢物特征。并且必须包含一列明确的【分类标签】（如健康/患病）。请参考下方示例表格格式：")
    
    # 构建一个直观的示例模板 DataFrame
    template_df = pd.DataFrame({
        "Sample_ID (样本编号)": ["S_001", "S_002", "S_003", "..."],
        "Group (临床标签)": ["Healthy", "Disease_A", "Healthy", "..."],
        "Metabolite_1 (代谢物1)": [1024.5, 453.2, 980.1, "..."],
        "Metabolite_2 (代谢物2)": [88.9, 1205.6, 95.3, "..."],
        "Metabolite_N (...)": ["...", "...", "...", "..."]
    })
    
    # 巧妙利用 html 高亮特定列，让用户一眼看懂
    st.write(template_df.style.apply(lambda x: ['background-color: #e6f7ff' if x.name == 'Sample_ID (样本编号)' 
                                            else 'background-color: #fff2e8' if x.name == 'Group (临床标签)' 
                                            else 'background-color: #f6ffed' for i in x], axis=0))
    
    st.markdown("""
    * 🟦 **蓝色列 (非特征列)**：诸如样本名、批次号等，**训练时必须在下方界面排除**，否则会导致所有模型崩溃！
    * 🟧 **橙色列 (目标列/Label)**：预测的目标答案。可以是文字（系统会自动转为数字）。
    * 🟩 **绿色列 (特征列/Features)**：纯数值型质谱定量数据。允许有少量缺失值（空着或者仪器导出的异常符号），平台会自动清洗填补。
    """)

# --- 第一阶段：数据上传 ---
st.header("1. 上传与核对数据")
uploaded_file = st.file_uploader("请上传特征矩阵文件 (支持 .csv 或 .xlsx)", type=['csv', 'xlsx'])

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file)
        st.session_state['df_raw'] = df; st.session_state['data_loaded'] = True
        st.success(f"✅ 文件上传成功！包含 {df.shape[0]} 样本，{df.shape[1]} 列信息。")
        with st.expander("👀 预览数据前 5 行"): st.dataframe(df.head())
    except Exception as e: st.error(f"读取文件失败: {e}")

# --- 第二阶段：参数设置与数据体检 ---
if st.session_state['data_loaded']:
    df = st.session_state['df_raw']
    st.divider()
    st.header("2. 设定目标与核心算法")
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("🎯 数据维度指定")
        target_col = st.selectbox("1. 目标列 (Label/Group)", options=df.columns)
        candidate_features = [c for c in df.columns if c != target_col]
        exclude_cols = st.multiselect("2. 排除非特征列 (⚠️必选项：如 Sample_ID)", options=candidate_features)
        feature_cols = [c for c in candidate_features if c not in exclude_cols]
        
    with col2:
        st.subheader("🧠 算法引擎选择")
        model_choice = st.selectbox("选择机器学习核心引擎:", [
            "XGBoost (极限梯度提升树)", 
            "RandomForest (随机森林)",
            "SVM (支持向量机)",
            "DNN (深度神经网络)"
        ])
        sel_method = st.selectbox("前置统计特征筛选", ['fdr', 'kbest', 'fwe'], index=0)

    # --- 🌟 新增：全自动数据体检与兼容性告警 (在用户设置好列之后立即执行) ---
    st.write("---")
    st.subheader("🩺 AI 数据兼容性实时体检")
    
    health_issues = 0
    
    # 检查 1：特征数量
    if len(feature_cols) < 5:
        st.error(f"❌ 特征数量严重不足 (仅剩 {len(feature_cols)} 个)。无法进行有效训练，请检查是否排除了过多的列。")
        health_issues += 1
        
    # 检查 2：列名重复问题 (极其容易导致 XGBoost 和 DNN 崩溃)
    if len(feature_cols) != len(set(feature_cols)):
        st.error("❌ 发现重复的代谢物/特征名称！所有算法（尤其是 XGBoost）都要求列名唯一，请回到 Excel 修改同名列后再上传。")
        health_issues += 1

    if health_issues == 0:
        # 进行深入的数据结构扫描
        with st.spinner("正在扫描特征矩阵质量..."):
            temp_X = df[feature_cols].copy()
            # 模拟第一道强制数值化清洗
            for col in temp_X.columns: temp_X[col] = pd.to_numeric(temp_X[col], errors='coerce')
            
            # 计算全表缺失率 (NaN)
            total_cells = temp_X.size
            missing_cells = temp_X.isna().sum().sum()
            missing_rate = missing_cells / total_cells if total_cells > 0 else 0
            
            # 检查样本标签分布均衡度
            class_counts = df[target_col].value_counts()
            min_class_count = class_counts.min()
            
            # 渲染体检报告
            col_health1, col_health2 = st.columns(2)
            with col_health1:
                if missing_rate > 0.3:
                    st.warning(f"⚠️ 整体缺失率高达 {missing_rate:.1%}。虽然平台会自动执行中位数填补，但过高的缺失率可能会削弱 SVM 和 DNN 的性能，建议回溯仪器数据。")
                elif missing_rate > 0.0:
                    st.success(f"✅ 数据矩阵完好。存在 {missing_rate:.1%} 的缺失值/异常符，将被平台内置的洗提机制自动填补，对模型无影响。")
                else:
                    st.success("✅ 数据矩阵完美！零缺失值。")
                    
            with col_health2:
                if min_class_count < 3:
                    st.error(f"❌ 样本极度不平衡！【{class_counts.idxmin()}】类只有 {min_class_count} 个样本。无法进行 5-Fold 交叉验证，请补充样本。")
                    health_issues += 1
                elif len(class_counts) < 2:
                    st.error("❌ 目标列必须至少包含 2 种不同的类别（如健康/患病）才能进行分类训练。")
                    health_issues += 1
                else:
                    st.success(f"✅ 标签分布健康。共 {len(class_counts)} 个类别，最小类包含 {min_class_count} 个样本，足以支撑交叉验证。")

    # --- 高级调参面板 (保持不变) ---
    with st.expander(f"⚙️ 打开 {model_choice.split()[0]} 超参数微调面板"):
        tune_col1, tune_col2 = st.columns(2)
        
        if "XGBoost" in model_choice:
            with tune_col1:
                xgb_n_estimators = st.slider("决策树数量 (n_estimators)", 50, 500, 100, step=50)
                xgb_max_depth = st.slider("树的最大深度 (max_depth)", 3, 15, 6)
            with tune_col2:
                xgb_lr = st.selectbox("学习率 (learning_rate)", [0.01, 0.05, 0.1, 0.2, 0.3], index=2)
            classifier_obj = XGBClassifier(n_estimators=xgb_n_estimators, max_depth=xgb_max_depth, learning_rate=xgb_lr, random_state=42, eval_metric='logloss')
            
        elif "RandomForest" in model_choice:
            with tune_col1:
                rf_n_estimators = st.slider("森林中树的数量 (n_estimators)", 50, 500, 100, step=50)
                rf_min_samples_split = st.slider("内部节点再划分所需最小样本数", 2, 10, 2)
            with tune_col2:
                rf_max_depth = st.selectbox("最大深度 (max_depth)", ["None (不限制)", 5, 10, 20, 50], index=0)
                rf_depth_val = None if rf_max_depth == "None (不限制)" else rf_max_depth
            classifier_obj = RandomForestClassifier(n_estimators=rf_n_estimators, max_depth=rf_depth_val, min_samples_split=rf_min_samples_split, random_state=42)

        elif "SVM" in model_choice:
            with tune_col1:
                svm_C = st.selectbox("正则化参数 C (惩罚系数)", [0.1, 1.0, 10.0, 100.0], index=1)
            with tune_col2:
                svm_kernel = st.selectbox("核函数 (Kernel)", ["rbf", "linear", "poly", "sigmoid"], index=0)
            classifier_obj = SVC(C=svm_C, kernel=svm_kernel, probability=True, random_state=42)

        elif "DNN" in model_choice:
            with tune_col1:
                dnn_layers = st.text_input("隐藏层架构 (逗号分隔)", "100, 50")
                dnn_max_iter = st.slider("最大迭代轮数 (max_iter)", 200, 2000, 500, step=100)
            with tune_col2:
                dnn_activation = st.selectbox("激活函数 (activation)", ["relu", "tanh", "logistic"])
                dnn_lr = st.selectbox("初始学习率", [0.001, 0.01, 0.05], index=0)
            try:
                hidden_layer_sizes = tuple(int(x.strip()) for x in dnn_layers.split(','))
            except:
                st.warning("架构格式错误，已恢复 (100, 50)")
                hidden_layer_sizes = (100, 50)
            classifier_obj = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes, activation=dnn_activation, learning_rate_init=dnn_lr, max_iter=dnn_max_iter, random_state=42)

    # --- 第三阶段：触发训练 (带安全锁) ---
    st.divider()
    st.header("3. 训练与打包")
    
    model_name_short = model_choice.split()[0]
    
    # 只有当 health_issues 为 0 时，才允许按钮生效并训练
    train_button_disabled = health_issues > 0
    
    if st.button(f"🚀 开始训练 {model_name_short} 并打包", type="primary", disabled=train_button_disabled):
        with st.spinner(f"正在清洗数据并训练 {model_name_short}，请稍候..."):
            try:
                # 数据清洗
                X_df = df[feature_cols].copy()
                for col in X_df.columns: X_df[col] = pd.to_numeric(X_df[col], errors='coerce') 
                X = X_df.values
                y = LabelEncoder().fit_transform(df[target_col].values)
                st.session_state['label_encoder'] = LabelEncoder().fit(df[target_col].values)
                
                # 构建 Pipeline
                ms_pipeline = Pipeline(steps=[
                    ('imputer', SimpleImputer(strategy='median')), 
                    ('scaler', StandardScaler()), 
                    ('feature_selector', MassSpecFeatureSelector(selection_method=sel_method)),
                    ('classifier', classifier_obj) 
                ])
                
                # 拟合与 CV
                ms_pipeline.fit(X, y)
                st.session_state['cv_score'] = np.mean(cross_val_score(ms_pipeline, X, y, cv=5))
                
                # 提取特征
                survived_indices = ms_pipeline.named_steps['feature_selector'].final_indices_
                survived_features = np.array(feature_cols)[survived_indices]
                
                # 智能化提取特征重要性
                if model_name_short in ["XGBoost", "RandomForest"]:
                    importances = ms_pipeline.named_steps['classifier'].feature_importances_
                else:
                    st.toast(f"正在通过置换算法解析 {model_name_short} 黑盒特征贡献度...", icon="🔍")
                    X_transformed = ms_pipeline[:-1].transform(X)
                    result = permutation_importance(ms_pipeline.named_steps['classifier'], X_transformed, y, n_repeats=5, random_state=42)
                    importances = result.importances_mean
                
                st.session_state['feature_importance_df'] = pd.DataFrame({
                    'Feature': survived_features, 'Importance': importances
                }).sort_values(by='Importance', ascending=False)
                
                st.session_state['trained_pipeline'] = ms_pipeline
                st.session_state['model_name_short'] = model_name_short
                st.session_state['model_trained'] = True
                st.success(f"✅ {model_name_short} 训练完成！")
            except Exception as e:
                st.error(f"训练失败: {e}")

    # --- 第四阶段：结果 ---
    if st.session_state['model_trained']:
        st.divider()
        st.header(f"📊 4. {st.session_state['model_name_short']} 训练结果与标志物鉴定")
        st.metric(label="5-Fold Cross Validation Accuracy", value=f"{st.session_state['cv_score']:.2%}")
        
        df_plot = st.session_state['feature_importance_df']
        fig, ax = plt.subplots(figsize=(10, 8)) 
        
        sns.barplot(x='Importance', y='Feature', data=df_plot.head(20), palette='viridis', ax=ax)
        ax.set_title(f"Top Biomarkers Importance ({st.session_state['model_name_short']})", fontsize=16, pad=15, fontweight='bold')
        ax.set_xlabel(f"{st.session_state['model_name_short']} Feature Weight / Permutation Score", fontsize=12)
        ax.set_ylabel("Metabolites / Features", fontsize=12)
        
        plt.subplots_adjust(left=0.4); sns.despine(); st.pyplot(fig)
        
        buffer = io.BytesIO()
        joblib.dump({'pipeline': st.session_state['trained_pipeline'], 'label_encoder': st.session_state['label_encoder']}, buffer)
        st.download_button(
            label=f"📥 下载 {st.session_state['model_name_short']} 智能核心 (.pkl)", 
            data=buffer.getvalue(),
            file_name=f"shimadzu_{st.session_state['model_name_short'].lower()}_core.pkl",
            mime="application/octet-stream"
        )
