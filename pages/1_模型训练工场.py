import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder, label_binarize
from sklearn.model_selection import cross_val_score
from sklearn.inspection import permutation_importance
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, accuracy_score, roc_curve, auc
import joblib
import io
import datetime

# 尝试导入特征筛选类
try:
    from utils.data_processor import MassSpecFeatureSelector
except ImportError:
    st.error("⚠️ 警告：未找到 utils/data_processor.py 模块。已启用平台内置的默认特征筛选器。")
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
st.set_page_config(page_title="组学与临床数据机器学习建模平台", page_icon="📊", layout="wide")

for key in ['data_loaded', 'model_trained', 'trained_pipeline', 'df_raw']:
    if key not in st.session_state: st.session_state[key] = None if key in ['trained_pipeline', 'df_raw'] else False

# 🌟 新增：ROC 曲线对比记忆库
if 'roc_history' not in st.session_state:
    st.session_state['roc_history'] = []

st.title("📊 组学与临床数据机器学习建模平台")

with st.expander("📖 数据格式与准备规范指南", expanded=False):
    st.info("💡 **系统要求：** 数据矩阵需以【行】为样本观测值，以【列】为特征变量（代谢物、基因或临床生化指标）。数据须包含用于监督学习的【分类标签（如组别）】。")
    template_df = pd.DataFrame({
        "Sample_ID": ["S_001", "S_002", "S_003", "..."],
        "Group": ["Healthy", "Disease_A", "Healthy", "..."],
        "Feature_1": [1024.5, 453.2, 980.1, "..."],
        "Feature_N": ["...", "...", "...", "..."]
    })
    st.write(template_df)

# --- 第一阶段：数据上传 ---
st.header("1. 导入特征矩阵数据")
uploaded_file = st.file_uploader("请上传待分析的数据集 (支持 .csv 或 .xlsx 格式)", type=['csv', 'xlsx'])

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file)
        st.session_state['df_raw'] = df; st.session_state['data_loaded'] = True
        st.success(f"✅ 数据矩阵导入成功。共检测到 {df.shape[0]} 个样本，{df.shape[1]} 个变量。")
    except Exception as e: st.error(f"文件读取异常: {e}")

# --- 第二阶段：参数设置与数据体检 ---
if st.session_state['data_loaded']:
    df = st.session_state['df_raw']
    st.divider()
    st.header("2. 变量映射与核心算法配置")
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("🎯 变量映射定义")
        target_col = st.selectbox("1. 指定目标变量 (因变量/Label)", options=df.columns)
        candidate_features = [c for c in df.columns if c != target_col]
        exclude_cols = st.multiselect("2. 排除非特征变量 (⚠️必选项：必须排除如 Sample_ID 等非定量指标)", options=candidate_features)
        feature_cols = [c for c in candidate_features if c not in exclude_cols]
        
    with col2:
        st.subheader("🧠 核心算法模型")
        model_choice = st.selectbox("选择分类器模型:", [
            "XGBoost (极限梯度提升树)", 
            "RandomForest (随机森林)",
            "SVM (支持向量机)",
            "MLP (多层感知机 / 神经网络)"
        ])
        sel_method = st.selectbox("前置统计学特征筛选方法", ['fdr', 'kbest', 'fwe'], index=0)

    st.write("---")
    st.subheader("🩺 数据兼容性与结构评估")
    health_issues = 0
    if len(feature_cols) < 5:
        st.error("❌ 错误：保留特征数量过少，无法支持高维空间建模。"); health_issues += 1
    if len(feature_cols) != len(set(feature_cols)):
        st.error("❌ 错误：检测到重复的特征列名，违反算法矩阵输入要求，请修正数据源。"); health_issues += 1

    n_samples = df.shape[0]
    n_features = len(feature_cols)
    is_small_sample = n_samples < 1000
    is_high_dim = n_features > n_samples

    if health_issues == 0:
        temp_X = df[feature_cols].copy()
        missing_rate = temp_X.isna().sum().sum() / temp_X.size if temp_X.size > 0 else 0
        class_counts = df[target_col].value_counts()
        imbalance_ratio = class_counts.max() / class_counts.min() if class_counts.min() > 0 else float('inf')
        
        col_h1, col_h2 = st.columns(2)
        with col_h1:
            if missing_rate > 0.0: st.success(f"✅ 提示：存在 {missing_rate:.1%} 缺失值，系统将启用中位数策略自动填补。")
            else: st.success("✅ 矩阵完整性验证通过：无缺失值。")
        with col_h2:
            if class_counts.min() < 3:
                st.error("❌ 错误：单类样本过少 (<3)，无法支持交叉验证，请调整数据或合并分类。"); health_issues += 1
            elif imbalance_ratio > 3:
                st.warning(f"⚠️ 提示：检测到样本分布极度失衡 (最大组别是最小组别的 {imbalance_ratio:.1f} 倍)。强烈建议在下方开启【类别权重平衡】机制。")
            else: 
                st.success(f"✅ 样本分布评估通过 (当前数据形态: {n_samples} 样本, {n_features} 原始特征)。")

    # --- 数据感知型双层调参面板 ---
    st.markdown("### ⚙️ 模型超参数配置 (Hyperparameters Tuning)")
    st.info("💡 提示：系统已基于当前数据集的特征/样本量比值 ($p/n$) 自动预设了适用的正则化基准。支持自定义微调。")
    
    tune_tabs = st.tabs(["⚙️ 基础模型参数 (Base Hyperparameters)", "🛡️ 正则化与防过拟合策略 (Regularization)"])
    
    if "XGBoost" in model_choice:
        with tune_tabs[0]:
            col_b1, col_b2 = st.columns(2)
            with col_b1:
                xgb_n_estimators = st.slider("迭代次数 (n_estimators)", 50, 1000, 100 if is_small_sample else 300, step=50, help="基分类器数量，大样本集可适当增加以提高拟合能力。")
                xgb_max_depth = st.slider("最大树深 (max_depth)", 3, 15, 6)
            with col_b2:
                xgb_lr = st.selectbox("学习率 (learning_rate)", [0.01, 0.05, 0.1, 0.2, 0.3], index=2 if is_small_sample else 1)
                st.write("")
                use_balance = st.checkbox("⚖️ 启用类别权重平衡 (应对样本极度不均衡)", value=(imbalance_ratio > 3), 
                                          help="自动增加少数类的误判惩罚。可能略微降低总体准确率，但能防止罕见样本被漏诊，不捏造假数据。")
                xgb_scale_pos = imbalance_ratio if use_balance else 1.0 
                
        with tune_tabs[1]:
            col_a1, col_a2 = st.columns(2)
            with col_a1:
                xgb_subsample = st.slider("样本行采样率 (subsample)", 0.5, 1.0, 0.8 if is_small_sample else 0.7, step=0.1, help="引入随机性以降低方差。")
                xgb_colsample = st.slider("特征列采样率 (colsample_bytree)", 0.3, 1.0, 0.5 if is_high_dim else 0.8, step=0.1, help="在高维数据中降低此值可迫使模型评估次要特征，避免严重过拟合。")
                xgb_gamma = st.slider("最小分裂损失 (gamma)", 0.0, 10.0, 0.0, step=0.5, help="节点分裂所需的最小损失减少值，值越大模型越保守。")
            with col_a2:
                xgb_alpha = st.slider("L1 正则化系数 (reg_alpha)", 0.0, 5.0, 0.1, step=0.1, help="促使特征权重稀疏化，适合高维空间特征选择。")
                xgb_lambda = st.slider("L2 正则化系数 (reg_lambda)", 0.0, 20.0, 1.0, step=0.5, help="对叶子权重进行L2惩罚，防止权重过大。")
        classifier_obj = XGBClassifier(n_estimators=xgb_n_estimators, max_depth=xgb_max_depth, learning_rate=xgb_lr, 
                                       subsample=xgb_subsample, colsample_bytree=xgb_colsample, 
                                       reg_alpha=xgb_alpha, reg_lambda=xgb_lambda, gamma=xgb_gamma,
                                       scale_pos_weight=xgb_scale_pos, random_state=42, eval_metric='logloss')
        
    elif "RandomForest" in model_choice:
        with tune_tabs[0]:
            col_b1, col_b2 = st.columns(2)
            with col_b1:
                rf_n_estimators = st.slider("决策树规模 (n_estimators)", 50, 1000, 200 if is_small_sample else 500, step=50)
            with col_b2:
                rf_max_depth = st.selectbox("最大树深 (max_depth)", ["None (无限制)", 5, 10, 20, 30], index=1 if is_small_sample else 0)
                rf_depth_val = None if rf_max_depth == "None (无限制)" else rf_max_depth
                use_balance = st.checkbox("⚖️ 启用类别权重平衡 (class_weight='balanced')", value=(imbalance_ratio > 3),
                                          help="自动根据频率反比分配分类权重，惩罚模型对占优类的过度倾向。")
                rf_class_weight = "balanced" if use_balance else None
                
        with tune_tabs[1]:
            col_a1, col_a2 = st.columns(2)
            with col_a1:
                rf_min_samples_split = st.slider("节点分裂最小样本数 (min_samples_split)", 2, 20, 2 if is_small_sample else 5)
            with col_a2:
                rf_min_samples_leaf = st.slider("叶节点最小样本数 (min_samples_leaf)", 1, 10, 1)
        classifier_obj = RandomForestClassifier(n_estimators=rf_n_estimators, max_depth=rf_depth_val, 
                                                min_samples_split=rf_min_samples_split, min_samples_leaf=rf_min_samples_leaf, 
                                                class_weight=rf_class_weight, random_state=42)

    elif "SVM" in model_choice:
        with tune_tabs[0]:
            col_b1, col_b2 = st.columns(2)
            with col_b1:
                svm_C = st.selectbox("误差惩罚权重 (C)", [0.01, 0.1, 1.0, 10.0, 100.0], index=2, help="特征数远大于样本时建议减小 C 以增强正则化约束。")
            with col_b2:
                svm_kernel = st.selectbox("核函数类型 (Kernel)", ["linear", "rbf", "poly", "sigmoid"], index=0 if (is_high_dim and is_small_sample) else 1, help="高维小样本情形下推荐线性核 (Linear) 避免过拟合；非线性边界需求可选用 RBF。")
                use_balance = st.checkbox("⚖️ 启用类别权重平衡 (class_weight='balanced')", value=(imbalance_ratio > 3))
                svm_class_weight = "balanced" if use_balance else None
                
        with tune_tabs[1]:
            st.info("RBF / Poly 核函数参数设定：")
            svm_gamma = st.selectbox("核函数映射系数 (Gamma)", ["scale", "auto", 0.0001, 0.001, 0.01, 0.1, 1.0], index=0, help="决定单样本的影响范围，规模较小有助于获得平滑边界。")
        classifier_obj = SVC(C=svm_C, kernel=svm_kernel, gamma=svm_gamma, probability=True, 
                             class_weight=svm_class_weight, random_state=42)

    elif "MLP" in model_choice:
        with tune_tabs[0]:
            col_b1, col_b2 = st.columns(2)
            with col_b1:
                dnn_layers = st.text_input("网络拓扑架构 (逗号分隔节点数)", "64, 32" if is_small_sample else "128, 64")
                dnn_max_iter = st.slider("最大迭代次数 (max_iter)", 200, 2000, 500, step=100)
            with col_b2:
                dnn_activation = st.selectbox("激活函数 (activation)", ["relu", "tanh", "logistic"])
                dnn_lr = st.selectbox("初始学习率 (learning_rate_init)", [1e-4, 1e-3, 1e-2], index=1)
                
            if imbalance_ratio > 3:
                st.warning("⚠️ **数据失衡警告：** 当前数据存在类别失衡。原生的 MLPClassifier 不支持内在的分类权重平衡。如需处理高失衡临床队列，请优先考虑 XGBoost 或 Random Forest。")
                
        with tune_tabs[1]:
            col_a1, col_a2 = st.columns(2)
            with col_a1:
                dnn_alpha = st.selectbox("L2 权重衰减惩罚 (Weight Decay/alpha)", [1e-5, 1e-4, 1e-3, 1e-2, 0.1], index=2, help="在 Scikit-learn 轻量化架构中作为 Dropout 技术的等效平替方案。组学小样本建议设置为 1e-3 ~ 1e-2。")
            with col_a2:
                dnn_batch = st.selectbox("小批量规模 (batch_size)", ["auto", 16, 32, 64, 128, 256, 512], index=1 if is_small_sample else 0)
            st.info("📌 **关于深度学习环境的声明：** 鉴于 Web 端服务器约束，本平台采用 Scikit-learn MLP。若需在极不平衡队列中应用 Focal Loss，或完整部署超大规模队列，请向岛津相关人员索取 PyTorch 进阶版脚本。")
            
        try: hidden_layer_sizes = tuple(int(x.strip()) for x in dnn_layers.split(','))
        except: hidden_layer_sizes = (64, 32)
        classifier_obj = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes, activation=dnn_activation, 
                                       learning_rate_init=dnn_lr, max_iter=dnn_max_iter, alpha=dnn_alpha, 
                                       batch_size=dnn_batch, random_state=42, early_stopping=True)

    st.divider()
    st.header("3. 模型训练与序列化封装")
    model_name_short = model_choice.split()[0]
    
    if st.button(f"🚀 启动训练 ({model_name_short})", type="primary", disabled=(health_issues > 0)):
        with st.spinner(f"正在执行矩阵预处理、降维及 {model_name_short} 算法训练..."):
            try:
                X_df = df[feature_cols].copy()
                for col in X_df.columns: X_df[col] = pd.to_numeric(X_df[col], errors='coerce') 
                X = X_df.values
                
                le = LabelEncoder()
                y = le.fit_transform(df[target_col].values)
                st.session_state['label_encoder'] = le
                is_multiclass = len(le.classes_) > 2
                
                ms_pipeline = Pipeline(steps=[
                    ('imputer', SimpleImputer(strategy='median')), 
                    ('scaler', StandardScaler()), 
                    ('feature_selector', MassSpecFeatureSelector(selection_method=sel_method)),
                    ('classifier', classifier_obj) 
                ])
                
                # --- CV 准确率与预测 ---
                ms_pipeline.fit(X, y)
                st.session_state['cv_score'] = np.mean(cross_val_score(ms_pipeline, X, y, cv=5))
                
                y_pred_train = ms_pipeline.predict(X)
                y_prob_train = ms_pipeline.predict_proba(X)
                
                st.session_state['train_acc'] = accuracy_score(y, y_pred_train)
                st.session_state['cm_train'] = confusion_matrix(y, y_pred_train)
                st.session_state['y_train_true'] = y
                st.session_state['y_prob_train'] = y_prob_train
                
                # 🌟 统一计算当前 ROC 核心数据以备提取
                if is_multiclass:
                    y_true_bin = label_binarize(y, classes=range(len(le.classes_)))
                    fpr_curr, tpr_curr, _ = roc_curve(y_true_bin.ravel(), y_prob_train.ravel())
                    auc_curr = auc(fpr_curr, tpr_curr)
                    st.session_state['cv_auc'] = np.mean(cross_val_score(ms_pipeline, X, y, cv=5, scoring='roc_auc_ovr'))
                else:
                    fpr_curr, tpr_curr, _ = roc_curve(y, y_prob_train[:, 1])
                    auc_curr = auc(fpr_curr, tpr_curr)
                    st.session_state['cv_auc'] = np.mean(cross_val_score(ms_pipeline, X, y, cv=5, scoring='roc_auc'))
                
                st.session_state['current_roc_data'] = {'fpr': fpr_curr, 'tpr': tpr_curr, 'auc': auc_curr}

                survived_indices = ms_pipeline.named_steps['feature_selector'].final_indices_
                survived_features = np.array(feature_cols)[survived_indices]
                
                if model_name_short in ["XGBoost", "RandomForest"]:
                    importances = ms_pipeline.named_steps['classifier'].feature_importances_
                else:
                    st.toast(f"正在执行 Permutation 算法解析特征贡献重要性...", icon="🔍")
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
                
                final_params = ms_pipeline.named_steps['classifier'].get_params()
                log_text = f"=========================================\n"
                log_text += f" Shimadzu Clinical Omics - Model Parameters Log \n"
                log_text += f"=========================================\n"
                log_text += f"Algorithm Engine: {model_name_short}\n"
                log_text += f"Input Matrix: {X.shape[0]} Observations, {X.shape[1]} Raw Features\n"
                log_text += f"Features Retained Post-screening: {len(survived_features)}\n\n"
                for param, val in final_params.items():
                    log_text += f"- {param}: {val}\n"
                st.session_state['param_log'] = log_text

                st.session_state['trained_pipeline'] = ms_pipeline
                st.session_state['model_name_short'] = model_name_short
                st.session_state['model_trained'] = True
                st.success(f"✅ 模型拟合结束。{model_name_short} 对象已就绪。")
            except Exception as e:
                st.error(f"建模异常终止: {e}")

    # --- 第四阶段：结果展示与多线 ROC 提取 ---
    if st.session_state['model_trained']:
        st.divider()
        st.header(f"📊 4. 模型诊断与关键生物学变量解析")
        
        # 指标看板
        col_metric1, col_metric2, col_metric3 = st.columns(3)
        with col_metric1:
            st.metric(label="5-Fold CV Accuracy\n(泛化精度 - 内部验证)", value=f"{st.session_state['cv_score']:.2%}")
        with col_metric2:
            auc_val = st.session_state['cv_auc']
            st.metric(label="5-Fold CV AUC Score\n(ROC曲线下面积)", value=f"{auc_val:.3f}" if not np.isnan(auc_val) else "N/A")
        with col_metric3:
            st.metric(label="Training Set Accuracy\n(内部拟合精度)", value=f"{st.session_state['train_acc']:.2%}")
        st.write("---")
        
        # 图像布局
        col_plot1, col_plot2, col_plot3 = st.columns([1, 1, 1.2])
        
        with col_plot1:
            st.subheader("Training Confusion Matrix")
            fig_cm, ax_cm = plt.subplots(figsize=(5, 4))
            le_classes = st.session_state['label_encoder'].classes_
            sns.heatmap(st.session_state['cm_train'], annot=True, fmt='d', cmap='Blues', ax=ax_cm, xticklabels=le_classes, yticklabels=le_classes)
            ax_cm.set_ylabel("True Label"); ax_cm.set_xlabel("Predicted Label")
            st.pyplot(fig_cm)
            
        with col_plot2:
            st.subheader("Training ROC Curve")
            fig_roc, ax_roc = plt.subplots(figsize=(5, 4))
            y_true = st.session_state['y_train_true']
            y_prob = st.session_state['y_prob_train']
            if len(le_classes) == 2:
                fpr, tpr, _ = roc_curve(y_true, y_prob[:, 1])
                ax_roc.plot(fpr, tpr, color='#01579B', lw=2, label=f'ROC (AUC = {auc(fpr, tpr):.3f})')
            else:
                for i, color in zip(range(len(le_classes)), sns.color_palette("husl", len(le_classes))):
                    fpr, tpr, _ = roc_curve(y_true == i, y_prob[:, i])
                    ax_roc.plot(fpr, tpr, color=color, lw=2, label=f'{le_classes[i]} (AUC={auc(fpr, tpr):.2f})')
            ax_roc.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--')
            ax_roc.set_xlabel('False Positive Rate'); ax_roc.set_ylabel('True Positive Rate')
            ax_roc.legend(loc="lower right", fontsize=8)
            st.pyplot(fig_roc)
            
        with col_plot3:
            st.subheader(f"Top Features Relative Importance")
            df_plot = st.session_state['feature_importance_df']
            fig_bar, ax_bar = plt.subplots(figsize=(6, 5)) 
            sns.barplot(x='Importance', y='Feature', data=df_plot.head(15), palette='viridis', ax=ax_bar)
            ax_bar.set_xlabel(f"Normalized Score (0-1)", fontsize=10); ax_bar.set_ylabel("")
            plt.subplots_adjust(left=0.3); sns.despine()
            st.pyplot(fig_bar)

        # 🌟 核心杀手锏：多线 ROC 提取面板 
        st.markdown("### 🏆 多模型 ROC 曲线看板")
        st.info("💡 如果您对当前的参数调试结果满意，可将其提取至下方的对比看板。切换不同模型反复提取，即可生成多线同框的顶级 SCI 图表。")
        
        col_roc1, col_roc2, col_roc3 = st.columns([2, 1, 1])
        with col_roc1:
            custom_curve_name = st.text_input("为当前模型打上标签：", value=f"{st.session_state['model_name_short']} (AUC={st.session_state['current_roc_data']['auc']:.3f})")
        with col_roc2:
            st.write("") # 占位对齐
            if st.button("➕ 提取当前曲线至看板", type="primary"):
                # 存入 session_state 的记忆库
                st.session_state['roc_history'].append({
                    'name': custom_curve_name,
                    'fpr': st.session_state['current_roc_data']['fpr'],
                    'tpr': st.session_state['current_roc_data']['tpr'],
                    'auc': st.session_state['current_roc_data']['auc']
                })
                st.rerun()
        with col_roc3:
            st.write("") 
            if st.button("🗑️ 清空对比看板") and len(st.session_state['roc_history']) > 0:
                st.session_state['roc_history'] = []
                st.rerun()
                
        # 渲染多线同框图
        if len(st.session_state['roc_history']) > 0:
            st.markdown("#### 综合诊断效能基准对比 (Benchmark)")
            fig_multi, ax_multi = plt.subplots(figsize=(8, 6))
            colors = sns.color_palette("Set1", n_colors=max(10, len(st.session_state['roc_history'])))
            
            for idx, roc_obj in enumerate(st.session_state['roc_history']):
                ax_multi.plot(roc_obj['fpr'], roc_obj['tpr'], color=colors[idx], lw=2, 
                              label=f"{roc_obj['name']}")
                
            ax_multi.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--')
            ax_multi.set_xlabel('False Positive Rate (1 - Specificity)', fontsize=12)
            ax_multi.set_ylabel('True Positive Rate (Sensitivity)', fontsize=12)
            ax_multi.set_title('Multi-Model ROC Curve Comparison', fontsize=14)
            ax_multi.legend(loc="lower right", fontsize=10)
            sns.despine()
            
            # 居中展示
            col_chart_left, col_chart_mid, col_chart_right = st.columns([1, 2, 1])
            with col_chart_mid:
                st.pyplot(fig_multi)

        st.divider()
        st.header("📥 5. 核心资产输出与科研通路分析接口")
        col_dl1, col_dl2, col_dl3 = st.columns(3)
        buffer = io.BytesIO()
        joblib.dump({'pipeline': st.session_state['trained_pipeline'], 'label_encoder': st.session_state['label_encoder']}, buffer)
        
        with col_dl1:
            st.download_button(label=f"📦 下载算法序列化模型 (.pkl)", data=buffer.getvalue(), file_name=f"shimadzu_model.pkl", mime="application/octet-stream")
        with col_dl2:
            st.download_button(label="📝 下载模型超参数日志 (.txt)", data=st.session_state['param_log'].encode('utf-8'), file_name=f"hyperparameters_log.txt", mime="text/plain")
        with col_dl3:
            csv_data = st.session_state['feature_importance_df'].to_csv(index=False).encode('utf-8-sig')
            st.download_button(label="📊 下载完整特征权重清单 (.csv)", data=csv_data, file_name=f"feature_weights.csv", mime="text/csv")
