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
from sklearn.neural_network import MLPClassifier
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

st.title("🛠️ 多引擎模型训练工场")
st.markdown("支持 XGBoost 与深度神经网络 (DNN)，支持超参数微调，自动抽取核心生物标志物。")

# --- 第一阶段：数据上传 ---
st.header("1. 上传与核对数据")
uploaded_file = st.file_uploader("请上传特征矩阵文件 (支持 .csv 或 .xlsx)", type=['csv', 'xlsx'])

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file)
        st.session_state['df_raw'] = df; st.session_state['data_loaded'] = True
        st.success(f"✅ 文件上传成功！包含 {df.shape[0]} 样本，{df.shape[1]} 特征。")
        with st.expander("👀 预览数据前 5 行"): st.dataframe(df.head())
    except Exception as e: st.error(f"读取文件失败: {e}")

# --- 第二阶段：参数设置 ---
if st.session_state['data_loaded']:
    df = st.session_state['df_raw']
    st.divider()
    st.header("2. 设定目标与核心算法")
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("🎯 数据维度指定")
        target_col = st.selectbox("1. 目标列 (Label)", options=df.columns)
        candidate_features = [c for c in df.columns if c != target_col]
        exclude_cols = st.multiselect("2. 排除非特征列 (如编号 Sample_ID)", options=candidate_features)
        feature_cols = [c for c in candidate_features if c not in exclude_cols]
        
    with col2:
        st.subheader("🧠 算法引擎与特征筛选")
        model_choice = st.selectbox("选择机器学习核心引擎:", ["XGBoost (树模型王者，推荐)", "Deep Neural Network (深度神经网络)"])
        sel_method = st.selectbox("前置统计特征筛选", ['fdr', 'kbest', 'fwe'], index=0)

    # --- 高级调参面板 ---
    with st.expander("⚙️ 打开模型超参数微调面板 (Hyperparameter Tuning)"):
        st.markdown(f"当前正在配置：**{model_choice}**")
        tune_col1, tune_col2 = st.columns(2)
        
        if "XGBoost" in model_choice:
            with tune_col1:
                xgb_n_estimators = st.slider("决策树数量 (n_estimators)", 50, 500, 100, step=50, help="树越多拟合越强，但容易过拟合")
                xgb_max_depth = st.slider("树的最大深度 (max_depth)", 3, 15, 6, help="单棵树的复杂度")
            with tune_col2:
                xgb_lr = st.selectbox("学习率 (learning_rate)", [0.01, 0.05, 0.1, 0.2, 0.3], index=2)
                
            classifier_obj = XGBClassifier(n_estimators=xgb_n_estimators, max_depth=xgb_max_depth, learning_rate=xgb_lr, random_state=42, eval_metric='logloss')
            
        else:
            with tune_col1:
                dnn_layers = st.text_input("隐藏层架构 (用逗号分隔)", "100, 50", help="例如 '100, 50' 代表两层，分别有100和50个神经元")
                dnn_max_iter = st.slider("最大训练轮数 (max_iter)", 200, 2000, 500, step=100)
            with tune_col2:
                dnn_activation = st.selectbox("激活函数 (activation)", ["relu", "tanh", "logistic"])
                dnn_lr = st.selectbox("初始学习率 (learning_rate_init)", [0.001, 0.01, 0.05], index=0)
                
            # 解析字符串为元组
            try:
                hidden_layer_sizes = tuple(int(x.strip()) for x in dnn_layers.split(','))
            except:
                st.warning("架构格式错误，已恢复默认 (100, 50)")
                hidden_layer_sizes = (100, 50)
                
            classifier_obj = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes, activation=dnn_activation, learning_rate_init=dnn_lr, max_iter=dnn_max_iter, random_state=42)

    # --- 第三阶段：触发训练 ---
    st.divider()
    st.header("3. 训练与打包")
    
    if st.button(f"🚀 开始训练 {model_choice.split()[0]} 并打包", type="primary"):
        if len(feature_cols) < 2: st.error("特征列太少！")
        else:
            with st.spinner(f"正在清洗数据并训练 {model_choice}，请稍候..."):
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
                    if "XGBoost" in model_choice:
                        importances = ms_pipeline.named_steps['classifier'].feature_importances_
                    else:
                        st.toast("正在通过置换算法解析 DNN 黑盒特征贡献度...", icon="🔍")
                        # 截取分类器之前的数据管道，转换数据
                        X_transformed = ms_pipeline[:-1].transform(X)
                        # 计算置换重要性
                        result = permutation_importance(ms_pipeline.named_steps['classifier'], X_transformed, y, n_repeats=5, random_state=42)
                        importances = result.importances_mean
                    
                    st.session_state['feature_importance_df'] = pd.DataFrame({
                        'Feature': survived_features, 'Importance': importances
                    }).sort_values(by='Importance', ascending=False)
                    
                    st.session_state['trained_pipeline'] = ms_pipeline
                    st.session_state['model_name'] = model_choice.split()[0]
                    st.session_state['model_trained'] = True
                    st.success("✅ 模型训练完成！")
                except Exception as e:
                    st.error(f"训练失败: {e}")

    # --- 第四阶段：结果 ---
    if st.session_state['model_trained']:
        st.divider()
        st.header(f"📊 4. {st.session_state['model_name']} 训练结果")
        st.metric(label="5-Fold Cross Validation Accuracy", value=f"{st.session_state['cv_score']:.2%}")
        
        df_plot = st.session_state['feature_importance_df']
        fig, ax = plt.subplots(figsize=(10, 8)) 
        sns.barplot(x='Importance', y='Feature', data=df_plot.head(20), palette='viridis', ax=ax)
        ax.set_title(f"Top Biomarkers Importance ({st.session_state['model_name']})", fontsize=16, pad=15, fontweight='bold')
        ax.set_xlabel("Feature Weight / Permutation Importance Score", fontsize=12)
        plt.subplots_adjust(left=0.4); sns.despine(); st.pyplot(fig)
        
        buffer = io.BytesIO()
        joblib.dump({'pipeline': st.session_state['trained_pipeline'], 'label_encoder': st.session_state['label_encoder']}, buffer)
        st.download_button(
            label=f"📥 下载智能核心 (.pkl)", data=buffer.getvalue(),
            file_name=f"shimadzu_{st.session_state['model_name'].lower()}_core.pkl",
            mime="application/octet-stream"
        )
