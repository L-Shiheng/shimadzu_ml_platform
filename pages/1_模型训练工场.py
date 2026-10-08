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
from sklearn.metrics import confusion_matrix, accuracy_score
import joblib
import io
import datetime

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

    # --- 智能感知数据体量 ---
    n_samples = df.shape[0]
    n_features = len(feature_cols)
    is_small_sample = n_samples < 1000
    is_high_dim = n_features > n_samples

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
            else: st.success(f"✅ 标签分布健康 (当前为 {n_samples} 样本, {n_features} 原始特征)。")

    # --- 数据感知型双层调参面板 ---
    st.markdown("### ⚙️ 专家级超参数配置")
    st.info("💡 系统已根据您上传的**特征/样本比例**自动为您预设了抗过拟合基准。您可在此基础上微调。")
    
    tune_tabs = st.tabs(["🎯 基础引擎动力 (Basic)", "🛡️ 高级抗过拟合装甲 (Advanced)"])
    
    if "XGBoost" in model_choice:
        with tune_tabs[0]:
            col_b1, col_b2 = st.columns(2)
            with col_b1:
                xgb_n_estimators = st.slider("树的数量 (n_estimators)", 50, 1000, 100 if is_small_sample else 300, step=50, help="迭代次数，树越多模型越复杂。大样本可适当增加。")
                xgb_max_depth = st.slider("最大深度 (max_depth)", 3, 15, 6)
            with col_b2:
                xgb_lr = st.selectbox("学习率 (learning_rate)", [0.01, 0.05, 0.1, 0.2, 0.3], index=2 if is_small_sample else 1)
        with tune_tabs[1]:
            col_a1, col_a2 = st.columns(2)
            with col_a1:
                xgb_subsample = st.slider("行采样率 (subsample)", 0.5, 1.0, 0.8 if is_small_sample else 0.7, step=0.1, help="每棵树使用的样本比例，增加随机性防过拟合。")
                xgb_colsample = st.slider("列采样率 (colsample_bytree)", 0.3, 1.0, 0.5 if is_high_dim else 0.8, step=0.1, help="高维特征(>100)强烈建议调低，避免模型过度依赖单一标志物。")
            with col_a2:
                xgb_alpha = st.slider("L1 正则化 (reg_alpha)", 0.0, 5.0, 0.1, step=0.1, help="产生稀疏模型，对抗高维度。")
                xgb_lambda = st.slider("L2 正则化 (reg_lambda)", 0.0, 10.0, 1.0, step=0.5)
        classifier_obj = XGBClassifier(n_estimators=xgb_n_estimators, max_depth=xgb_max_depth, learning_rate=xgb_lr, 
                                       subsample=xgb_subsample, colsample_bytree=xgb_colsample, 
                                       reg_alpha=xgb_alpha, reg_lambda=xgb_lambda,
                                       random_state=42, eval_metric='logloss')
        
    elif "RandomForest" in model_choice:
        with tune_tabs[0]:
            col_b1, col_b2 = st.columns(2)
            with col_b1:
                rf_n_estimators = st.slider("森林规模 (n_estimators)", 50, 1000, 200 if is_small_sample else 500, step=50)
            with col_b2:
                rf_max_depth = st.selectbox("最大深度 (max_depth)", ["None (不限制)", 5, 10, 20, 30], index=1 if is_small_sample else 0)
                rf_depth_val = None if rf_max_depth == "None (不限制)" else rf_max_depth
        with tune_tabs[1]:
            col_a1, col_a2 = st.columns(2)
            with col_a1:
                rf_min_samples_split = st.slider("分裂最小样本数 (min_samples_split)", 2, 20, 2 if is_small_sample else 5)
            with col_a2:
                rf_min_samples_leaf = st.slider("叶节点最小样本数 (min_samples_leaf)", 1, 10, 1)
        classifier_obj = RandomForestClassifier(n_estimators=rf_n_estimators, max_depth=rf_depth_val, 
                                                min_samples_split=rf_min_samples_split, min_samples_leaf=rf_min_samples_leaf, random_state=42)

    elif "SVM" in model_choice:
        with tune_tabs[0]:
            col_b1, col_b2 = st.columns(2)
            with col_b1:
                svm_C = st.selectbox("正则化惩罚系数 (C)", [0.01, 0.1, 1.0, 10.0, 100.0], index=2, help="特征多时适当减小C，样本多时可增大C。")
            with col_b2:
                svm_kernel = st.selectbox("核函数 (Kernel)", ["linear", "rbf", "poly", "sigmoid"], index=0 if (is_high_dim and is_small_sample) else 1, help="小样本高维强烈首选 Linear 避免过拟合；大样本或低维选 RBF。")
        with tune_tabs[1]:
            st.info("RBF / Poly 核函数高级设定：")
            svm_gamma = st.selectbox("Gamma (核函数系数)", ["scale", "auto", 0.001, 0.01, 0.1, 1.0], index=0, help="scale为自适应。特征多偏小，样本多更小以获得平滑边界。")
        classifier_obj = SVC(C=svm_C, kernel=svm_kernel, gamma=svm_gamma, probability=True, random_state=42)

    elif "DNN" in model_choice:
        with tune_tabs[0]:
            col_b1, col_b2 = st.columns(2)
            with col_b1:
                dnn_layers = st.text_input("隐藏层架构 (用逗号分隔)", "64, 32" if is_small_sample else "128, 64")
                dnn_max_iter = st.slider("最大迭代轮数 (max_iter)", 200, 2000, 500, step=100)
            with col_b2:
                dnn_activation = st.selectbox("激活函数 (activation)", ["relu", "tanh", "logistic"])
                dnn_lr = st.selectbox("初始学习率 (learning_rate_init)", [1e-4, 1e-3, 1e-2], index=1)
        with tune_tabs[1]:
            col_a1, col_a2 = st.columns(2)
            with col_a1:
                dnn_alpha = st.selectbox("L2 正则化 / Weight Decay (alpha)", [1e-5, 1e-4, 1e-3, 1e-2, 0.1], index=2, help="代替Dropout抑制过拟合，小样本建议设置在 1e-3 ~ 1e-2。")
            with col_a2:
                dnn_batch = st.selectbox("批大小 (batch_size)", ["auto", 16, 32, 64, 128], index=1 if is_small_sample else 0)
            st.warning("📌 **架构提示：** 受限于当前云端轻量化部署的服务器算力与依赖环境，DNN 引擎采用 L2 正则化作为 Dropout 技术的平替。若您需要完整的重型深度学习框架（如 PyTorch/TensorFlow）并在本地高性能工作站进行大规模组学训练，请询问岛津相关人员获取完整版的本地端部署代码。")
            
        try: hidden_layer_sizes = tuple(int(x.strip()) for x in dnn_layers.split(','))
        except: hidden_layer_sizes = (64, 32)
        classifier_obj = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes, activation=dnn_activation, 
                                       learning_rate_init=dnn_lr, max_iter=dnn_max_iter, alpha=dnn_alpha, 
                                       batch_size=dnn_batch, random_state=42, early_stopping=True)

    st.divider()
    st.header("3. 训练与打包")
    model_name_short = model_choice.split()[0]
    
    if st.button(f"🚀 开始训练 {model_name_short} 并打包", type="primary", disabled=(health_issues > 0)):
        with st.spinner(f"正在清洗数据并训练 {model_name_short}，请稍候..."):
            try:
                X_df = df[feature_cols].copy()
                for col in X_df.columns: X_df[col] = pd.to_numeric(X_df[col], errors='coerce') 
                X = X_df.values
                
                le = LabelEncoder()
                y = le.fit_transform(df[target_col].values)
                st.session_state['label_encoder'] = le
                
                ms_pipeline = Pipeline(steps=[
                    ('imputer', SimpleImputer(strategy='median')), 
                    ('scaler', StandardScaler()), 
                    ('feature_selector', MassSpecFeatureSelector(selection_method=sel_method)),
                    ('classifier', classifier_obj) 
                ])
                
                ms_pipeline.fit(X, y)
                st.session_state['cv_score'] = np.mean(cross_val_score(ms_pipeline, X, y, cv=5))
                
                y_pred_train = ms_pipeline.predict(X)
                st.session_state['train_acc'] = accuracy_score(y, y_pred_train)
                st.session_state['cm_train'] = confusion_matrix(y, y_pred_train)
                
                survived_indices = ms_pipeline.named_steps['feature_selector'].final_indices_
                survived_features = np.array(feature_cols)[survived_indices]
                
                if model_name_short in ["XGBoost", "RandomForest"]:
                    importances = ms_pipeline.named_steps['classifier'].feature_importances_
                else:
                    st.toast(f"正在捕捉 {model_name_short} 预测置信度计算特征贡献...", icon="🔍")
                    X_transformed = ms_pipeline[:-1].transform(X)
                    try:
                        result = permutation_importance(ms_pipeline.named_steps['classifier'], X_transformed, y, n_repeats=5, random_state=42, scoring='neg_log_loss')
                        importances = np.abs(result.importances_mean)
                    except:
                        result = permutation_importance(ms_pipeline.named_steps['classifier'], X_transformed, y, n_repeats=5, random_state=42)
                        importances = np.abs(result.importances_mean)
                
                if np.max(importances) > 0:
                    importances = (importances - np.min(importances)) / (np.max(importances) - np.min(importances) + 1e-9)
                
                st.session_state['feature_importance_df'] = pd.DataFrame({
                    'Feature': survived_features, 'Importance': importances
                }).sort_values(by='Importance', ascending=False)
                
                # --- 新增：自动生成论文可用的超参数日志 ---
                final_params = ms_pipeline.named_steps['classifier'].get_params()
                log_text = f"=========================================\n"
                log_text += f" Shimadzu Clinical AI - Model Hyperparameters Log \n"
                log_text += f"=========================================\n"
                log_text += f"Date Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
                log_text += f"Model Engine: {model_name_short}\n"
                log_text += f"Input Matrix Size: {X.shape[0]} samples, {X.shape[1]} original features\n"
                log_text += f"Features retained after Pre-screening: {len(survived_features)}\n\n"
                log_text += f"--- Classifier Hyperparameters ---\n"
                for param, val in final_params.items():
                    log_text += f"- {param}: {val}\n"
                log_text += f"\nNote: Preprocessing included Median Imputation and Standard Scaling (Z-score).\n"
                st.session_state['param_log'] = log_text

                st.session_state['trained_pipeline'] = ms_pipeline
                st.session_state['model_name_short'] = model_name_short
                st.session_state['model_trained'] = True
                st.success(f"✅ {model_name_short} 训练完成！")
            except Exception as e:
                st.error(f"训练失败: {e}")

    # --- 第四阶段：结果展示 ---
    if st.session_state['model_trained']:
        st.divider()
        st.header(f"📊 4. {st.session_state['model_name_short']} 训练结果分析")
        
        col_metric1, col_metric2 = st.columns(2)
        with col_metric1:
            st.metric(label="5-Fold Cross Validation Accuracy\n(五折交叉验证准确率 - 评估泛化能力)", 
                      value=f"{st.session_state['cv_score']:.2%}")
        with col_metric2:
            st.metric(label="Training Set Accuracy\n(当前训练集回测拟合率)", 
                      value=f"{st.session_state['train_acc']:.2%}")
        
        col_plot1, col_plot2 = st.columns([1, 1.5])
        with col_plot1:
            st.subheader("Training Set Confusion Matrix")
            st.markdown("**(模型对当前数据的“背题”情况)**")
            fig_cm, ax_cm = plt.subplots(figsize=(6, 5))
            le_classes = st.session_state['label_encoder'].classes_
            sns.heatmap(st.session_state['cm_train'], annot=True, fmt='d', cmap='Blues', ax=ax_cm,
                        xticklabels=le_classes, yticklabels=le_classes)
            ax_cm.set_ylabel("True Label"); ax_cm.set_xlabel("Predicted Label")
            st.pyplot(fig_cm)
            
        with col_plot2:
            st.subheader(f"Top Biomarkers Relative Importance")
            df_plot = st.session_state['feature_importance_df']
            fig_bar, ax_bar = plt.subplots(figsize=(10, 8)) 
            sns.barplot(x='Importance', y='Feature', data=df_plot.head(20), palette='viridis', ax=ax_bar)
            ax_bar.set_xlabel(f"{st.session_state['model_name_short']} Normalized Score (0-1)", fontsize=12)
            plt.subplots_adjust(left=0.4); sns.despine()
            st.pyplot(fig_bar)
        
        st.divider()
        st.header("📥 5. 导出核心资产与参数日志")
        col_dl1, col_dl2 = st.columns(2)
        
        buffer = io.BytesIO()
        joblib.dump({'pipeline': st.session_state['trained_pipeline'], 'label_encoder': st.session_state['label_encoder']}, buffer)
        
        with col_dl1:
            st.download_button(
                label=f"📦 打包下载智能核心 (.pkl) 供应用终端使用", 
                data=buffer.getvalue(),
                file_name=f"shimadzu_{st.session_state['model_name_short'].lower()}_core.pkl",
                mime="application/octet-stream"
            )
        with col_dl2:
            st.download_button(
                label="📝 下载模型超参数日志 (SCI 论文 Methods 写作必备)", 
                data=st.session_state['param_log'].encode('utf-8'),
                file_name=f"{st.session_state['model_name_short']}_hyperparameters_log.txt",
                mime="text/plain",
                help="该日志提取自系统底层，包含最终确定的所有抗过拟合参数配置，可直接作为材料与方法补充。"
            )
