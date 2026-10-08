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

with st.expander("📖 上传前必读：标准数据格式指南", expanded=False):
    st.info("💡 **要求：** 数据必须是【行】代表样本，【列】代表代谢物特征。并且包含明确的【分类标签】。")
    template_df = pd.DataFrame({
        "Sample_ID": ["S_001", "S_002", "S_003", "..."],
        "Group": ["Healthy", "Disease_A", "Healthy", "..."],
        "Metabolite_1": [1024.5, 453.2, 980.1, "..."],
        "Metabolite_N": ["...", "...", "...", "..."]
    })
    st.write(template_df)

# --- 第一阶段：数据上传 ---
st.header("1. 上传与核对数据")
uploaded_file = st.file_uploader("请上传特征矩阵文件 (支持 .csv 或 .xlsx)", type=['csv', 'xlsx'])

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file)
        st.session_state['df_raw'] = df; st.session_state['data_loaded'] = True
        st.success(f"✅ 文件上传成功！包含 {df.shape[0]} 样本，{df.shape[1]} 列信息。")
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

    st.write("---")
    st.subheader("🩺 AI 数据兼容性实时体检")
    health_issues = 0
    if len(feature_cols) < 5:
        st.error("❌ 特征数量严重不足。"); health_issues += 1
    if len(feature_cols) != len(set(feature_cols)):
        st.error("❌ 发现重复的代谢物名称！请修改。"); health_issues += 1

    if health_issues == 0:
        temp_X = df[feature_cols].copy()
        missing_rate = temp_X.isna().sum().sum() / temp_X.size if temp_X.size > 0 else 0
        class_counts = df[target_col].value_counts()
        
        col_h1, col_h2 = st.columns(2)
        with col_h1:
            if missing_rate > 0.0: st.success(f"✅ 存在 {missing_rate:.1%} 缺失值，将自动填补。")
            else: st.success("✅ 数据矩阵完美！零缺失值。")
        with col_h2:
            if class_counts.min() < 3:
                st.error("❌ 样本极度不平衡或类别过少！"); health_issues += 1
            else: st.success("✅ 标签分布健康。")

    with st.expander(f"⚙️ 打开 {model_choice.split()[0]} 超参数微调面板"):
        tune_col1, tune_col2 = st.columns(2)
        if "XGBoost" in model_choice:
            with tune_col1:
                xgb_n_estimators = st.slider("决策树数量", 50, 500, 100, step=50)
                xgb_max_depth = st.slider("树的最大深度", 3, 15, 6)
            with tune_col2:
                xgb_lr = st.selectbox("学习率", [0.01, 0.05, 0.1, 0.2, 0.3], index=2)
            classifier_obj = XGBClassifier(n_estimators=xgb_n_estimators, max_depth=xgb_max_depth, learning_rate=xgb_lr, random_state=42, eval_metric='logloss')
            
        elif "RandomForest" in model_choice:
            with tune_col1:
                rf_n_estimators = st.slider("森林中树的数量", 50, 500, 100, step=50)
                rf_min_samples_split = st.slider("内部节点再划分最小样本数", 2, 10, 2)
            with tune_col2:
                rf_max_depth = st.selectbox("最大深度", ["None (不限制)", 5, 10, 20, 50], index=0)
                rf_depth_val = None if rf_max_depth == "None (不限制)" else rf_max_depth
            classifier_obj = RandomForestClassifier(n_estimators=rf_n_estimators, max_depth=rf_depth_val, min_samples_split=rf_min_samples_split, random_state=42)

        elif "SVM" in model_choice:
            with tune_col1:
                svm_C = st.selectbox("正则化参数 C", [0.1, 1.0, 10.0, 100.0], index=1)
            with tune_col2:
                svm_kernel = st.selectbox("核函数", ["rbf", "linear", "poly", "sigmoid"], index=0)
            classifier_obj = SVC(C=svm_C, kernel=svm_kernel, probability=True, random_state=42)

        elif "DNN" in model_choice:
            with tune_col1:
                dnn_layers = st.text_input("隐藏层架构", "100, 50")
                dnn_max_iter = st.slider("最大迭代轮数", 200, 2000, 500, step=100)
            with tune_col2:
                dnn_activation = st.selectbox("激活函数", ["relu", "tanh", "logistic"])
                dnn_lr = st.selectbox("初始学习率", [0.001, 0.01, 0.05], index=0)
            try: hidden_layer_sizes = tuple(int(x.strip()) for x in dnn_layers.split(','))
            except: hidden_layer_sizes = (100, 50)
            classifier_obj = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes, activation=dnn_activation, learning_rate_init=dnn_lr, max_iter=dnn_max_iter, random_state=42)

    st.divider()
    st.header("3. 训练与打包")
    model_name_short = model_choice.split()[0]
    
    if st.button(f"🚀 开始训练 {model_name_short} 并打包", type="primary", disabled=(health_issues > 0)):
        with st.spinner(f"正在清洗数据并训练 {model_name_short}，请稍候..."):
            try:
                X_df = df[feature_cols].copy()
                for col in X_df.columns: X_df[col] = pd.to_numeric(X_df[col], errors='coerce') 
                X = X_df.values
                y = LabelEncoder().fit_transform(df[target_col].values)
                st.session_state['label_encoder'] = LabelEncoder().fit(df[target_col].values)
                
                ms_pipeline = Pipeline(steps=[
                    ('imputer', SimpleImputer(strategy='median')), 
                    ('scaler', StandardScaler()), 
                    ('feature_selector', MassSpecFeatureSelector(selection_method=sel_method)),
                    ('classifier', classifier_obj) 
                ])
                
                ms_pipeline.fit(X, y)
                st.session_state['cv_score'] = np.mean(cross_val_score(ms_pipeline, X, y, cv=5))
                
                survived_indices = ms_pipeline.named_steps['feature_selector'].final_indices_
                survived_features = np.array(feature_cols)[survived_indices]
                
                # ---------------- 核心修复代码段 ----------------
                if model_name_short in ["XGBoost", "RandomForest"]:
                    importances = ms_pipeline.named_steps['classifier'].feature_importances_
                else:
                    st.toast(f"正在捕捉 {model_name_short} 预测概率微小下降计算贡献度...", icon="🔍")
                    X_transformed = ms_pipeline[:-1].transform(X)
                    try:
                        # 修复1：使用 neg_log_loss 捕捉概率分布的微小变化，而不是僵硬的准确率
                        result = permutation_importance(ms_pipeline.named_steps['classifier'], X_transformed, y, n_repeats=5, random_state=42, scoring='neg_log_loss')
                        importances = np.abs(result.importances_mean)
                    except:
                        result = permutation_importance(ms_pipeline.named_steps['classifier'], X_transformed, y, n_repeats=5, random_state=42)
                        importances = np.abs(result.importances_mean)
                
                # 修复2：Min-Max 强制归一化到 0~1 的区间，确保无论数值多小，条形图都能完美按比例显示长度
                if np.max(importances) > 0:
                    importances = (importances - np.min(importances)) / (np.max(importances) - np.min(importances) + 1e-9)
                # ------------------------------------------------
                
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
        ax.set_title(f"Top Biomarkers Relative Importance ({st.session_state['model_name_short']})", fontsize=16, pad=15, fontweight='bold')
        ax.set_xlabel(f"{st.session_state['model_name_short']} Normalized Importance Score (0-1)", fontsize=12)
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
