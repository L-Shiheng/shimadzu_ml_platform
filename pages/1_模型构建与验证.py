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

# 导入特征筛选类
try:
    from utils.data_processor import MassSpecFeatureSelector
except ImportError:
    st.error("⚠️ 提示：未检测到 utils/data_processor.py 模块，系统已自动降级为内置基础特征筛选器。")
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

# ROC 曲线对比记忆库
if 'roc_history' not in st.session_state:
    st.session_state['roc_history'] = []

st.title("📊 多维数据模型构建与验证")

with st.expander("📖 数据格式说明与准备规范", expanded=False):
    st.info("💡 **矩阵格式要求：** 数据需以【行】为观测样本，【列】为特征变量（代谢物、基因表达量或临床指标）。数据集中必须包含用于监督学习的【类别标签】列。")
    template_df = pd.DataFrame({
        "Sample_ID": ["S_001", "S_002", "S_003", "..."],
        "Group": ["Healthy", "Disease_A", "Healthy", "..."],
        "Feature_1": [1024.5, 453.2, 980.1, "..."],
        "Feature_N": ["...", "...", "...", "..."]
    })
    st.write(template_df)

# --- 第一阶段：数据导入 ---
st.header("1. 导入特征矩阵数据")
uploaded_file = st.file_uploader("请上传待分析的格式化数据集 (支持 .csv 或 .xlsx)", type=['csv', 'xlsx'])

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file)
        st.session_state['df_raw'] = df; st.session_state['data_loaded'] = True
        st.success(f"✅ 数据矩阵载入成功。检测到 {df.shape[0]} 个观测样本，{df.shape[1]} 个变量。")
    except Exception as e: st.error(f"文件读取异常: {e}")

# --- 第二阶段：参数设置与数据质量评估 ---
if st.session_state['data_loaded']:
    df = st.session_state['df_raw']
    st.divider()
    st.header("2. 变量映射与基准算法配置")
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("🎯 变量属性定义")
        target_col = st.selectbox("1. 指定目标变量 (因变量/Label)", options=df.columns)
        candidate_features = [c for c in df.columns if c != target_col]
        exclude_cols = st.multiselect("2. 排除非特征变量 (必选项：如样本编号、批次号等非定量标识符)", options=candidate_features)
        feature_cols = [c for c in candidate_features if c not in exclude_cols]
        
    with col2:
        st.subheader("🧠 核心分类器配置")
        model_choice = st.selectbox("选择机器学习评估引擎:", [
            "XGBoost (极限梯度提升树)", 
            "RandomForest (随机森林)",
            "SVM (支持向量机)",
            "MLP (多层感知机)"
        ])
        sel_method = st.selectbox("前置特征统计检验方法", ['fdr', 'kbest', 'fwe'], index=0)

    st.write("---")
    st.subheader("🩺 矩阵兼容性与结构评估")
    health_issues = 0
    if len(feature_cols) < 5:
        st.error("❌ 数据不足：保留特征变量数量过少，无法支持高维空间投影评估。"); health_issues += 1
    if len(feature_cols) != len(set(feature_cols)):
        st.error("❌ 结构冲突：检测到重复的特征变量名，不符合矩阵正交输入要求，请校验数据源。"); health_issues += 1

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
            if missing_rate > 0.0: st.info(f"ℹ️ 数据质控报告：存在 {missing_rate:.1%} 缺失值，将执行中位数插补策略。")
            else: st.success("✅ 数据质控报告：矩阵完整，未检测到缺失值。")
        with col_h2:
            if class_counts.min() < 3:
                st.error("❌ 样本量不足：存在观测数小于 3 的分类组别，无法支持交叉验证分层要求。"); health_issues += 1
            elif imbalance_ratio > 3:
                st.warning(f"⚠️ 分布偏倚警告：检测到显著的样本类别不平衡 (多数组占比为少数组的 {imbalance_ratio:.1f} 倍)。建议于下方参数面板启用【类别权重平衡】参数。")
            else: 
                st.success(f"✅ 分布评估报告：类别分布均衡，具备良好的统计效能基底 (矩阵维度: {n_samples}×{n_features})。")

    # --- 模型超参数调优面板 ---
    st.markdown("### ⚙️ 算法超参数设定 (Hyperparameter Configuration)")
    st.info("💡 辅助提示：系统已依据当前数据集特征量与样本量的比例 ($p/n$)，为您预设了基础正则化参考阈值。请根据特定分析需求进行微调。")
    
    tune_tabs = st.tabs(["⚙️ 基础模型参数 (Base Settings)", "🛡️ 正则化与方差控制 (Regularization)"])
    
    if "XGBoost" in model_choice:
        with tune_tabs[0]:
            col_b1, col_b2 = st.columns(2)
            with col_b1:
                xgb_n_estimators = st.slider("最大迭代轮数 (n_estimators)", 50, 1000, 100 if is_small_sample else 300, step=50, help="基学习器集成数量。大样本集或低学习率时可适当增加以保障收敛。")
                xgb_max_depth = st.slider("单树最大深度 (max_depth)", 3, 15, 6, help="控制模型复杂度，数值过高易导致小样本数据过拟合。")
            with col_b2:
                xgb_lr = st.selectbox("学习率 (learning_rate)", [0.01, 0.05, 0.1, 0.2, 0.3], index=2 if is_small_sample else 1)
                st.write("")
                use_balance = st.checkbox("⚖️ 启用代价敏感学习 (权重平衡干预)", value=(imbalance_ratio > 3), 
                                          help="通过 scale_pos_weight 参数增加少数类样本的误判损失权重。推荐用于罕见病阳性率偏低的数据队列。")
                xgb_scale_pos = imbalance_ratio if use_balance else 1.0 
                
        with tune_tabs[1]:
            col_a1, col_a2 = st.columns(2)
            with col_a1:
                xgb_subsample = st.slider("样本行采样率 (subsample)", 0.5, 1.0, 0.8 if is_small_sample else 0.7, step=0.1, help="引入随机采样机制以降低模型方差。")
                xgb_colsample = st.slider("特征列采样率 (colsample_bytree)", 0.3, 1.0, 0.5 if is_high_dim else 0.8, step=0.1, help="高维组学矩阵建议调低此值，促使模型遍历评估次要变量的贡献度。")
                xgb_gamma = st.slider("叶节点分裂阈值 (gamma)", 0.0, 10.0, 0.0, step=0.5, help="控制节点分裂所需达到的最小损失下降量。值越高，模型越保守。")
            with col_a2:
                xgb_alpha = st.slider("L1 正则化系数 (reg_alpha)", 0.0, 5.0, 0.1, step=0.1, help="LASSO惩罚项，促使特征权重向零收缩，适用于高维稀疏特征。")
                xgb_lambda = st.slider("L2 正则化系数 (reg_lambda)", 0.0, 20.0, 1.0, step=0.5, help="Ridge惩罚项，平滑特征权重以缓解过拟合。")
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
                rf_max_depth = st.selectbox("最大树深 (max_depth)", ["None (无约束展开)", 5, 10, 20, 30], index=1 if is_small_sample else 0)
                rf_depth_val = None if rf_max_depth == "None (无约束展开)" else rf_max_depth
                use_balance = st.checkbox("⚖️ 启用分类权重自适应 (class_weight='balanced')", value=(imbalance_ratio > 3),
                                          help="依据各类别样本频率分布的倒数自动指派权重，修正模型趋同于占优类的系统性偏差。")
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
                svm_C = st.selectbox("软间隔惩罚系数 (C)", [0.01, 0.1, 1.0, 10.0, 100.0], index=2, help="控制分类间隔的最大化与分类误差容忍度之间的折中。")
            with col_b2:
                svm_kernel = st.selectbox("核函数类型 (Kernel)", ["linear", "rbf", "poly", "sigmoid"], index=0 if (is_high_dim and is_small_sample) else 1, help="高维特征空间推荐优先测试线性核 (Linear)；若寻求复杂非线性判别边界可评估 RBF。")
                use_balance = st.checkbox("⚖️ 启用分类权重平衡 (class_weight='balanced')", value=(imbalance_ratio > 3))
                svm_class_weight = "balanced" if use_balance else None
                
        with tune_tabs[1]:
            st.info("RBF / Poly 核函数特征空间映射参数：")
            svm_gamma = st.selectbox("核函数系数 (Gamma)", ["scale", "auto", 0.0001, 0.001, 0.01, 0.1, 1.0], index=0, help="定义单一样本的映射影响半径。")
        classifier_obj = SVC(C=svm_C, kernel=svm_kernel, gamma=svm_gamma, probability=True, 
                             class_weight=svm_class_weight, random_state=42)

    elif "MLP" in model_choice:
        with tune_tabs[0]:
            col_b1, col_b2 = st.columns(2)
            with col_b1:
                dnn_layers = st.text_input("网络隐藏层拓扑 (逗号分隔节点数值)", "64, 32" if is_small_sample else "128, 64")
                dnn_max_iter = st.slider("全局最大迭代约束 (max_iter)", 200, 2000, 500, step=100)
            with col_b2:
                dnn_activation = st.selectbox("激活函数 (activation)", ["relu", "tanh", "logistic"])
                dnn_lr = st.selectbox("初始学习速率 (learning_rate_init)", [1e-4, 1e-3, 1e-2], index=1)
                
            if imbalance_ratio > 3:
                st.warning("⚠️ **分布失衡提醒：** 当前队列存在类别比例显著失衡。Scikit-learn 的多层感知机架构未提供内置的代价敏感权重参数。为保障分类效能，建议更换为树模型或支持向量机。")
                
        with tune_tabs[1]:
            col_a1, col_a2 = st.columns(2)
            with col_a1:
                dnn_alpha = st.selectbox("L2 权重衰减系数 (Weight Decay/alpha)", [1e-5, 1e-4, 1e-3, 1e-2, 0.1], index=2, help="控制网络权重的复杂度生长。组学小样本集推荐设定为 1e-3 至 1e-2 以平替 Dropout 功能。")
            with col_a2:
                dnn_batch = st.selectbox("批梯度下降容量 (batch_size)", ["auto", 16, 32, 64, 128, 256, 512], index=1 if is_small_sample else 0)
            st.info("📌 **系统架构声明：** 受限于轻量化 Web 运行环境约束，本模块采用 Scikit-learn 的基础 MLP 架构并利用 L2 正则化约束泛化边界。若具备重构大规模临床深度模型（需引入 Focal Loss 或原生 Dropout 层）的需求，推荐向平台维护团队获取 PyTorch 框架本地部署指引。")
            
        try: hidden_layer_sizes = tuple(int(x.strip()) for x in dnn_layers.split(','))
        except: hidden_layer_sizes = (64, 32)
        classifier_obj = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes, activation=dnn_activation, 
                                       learning_rate_init=dnn_lr, max_iter=dnn_max_iter, alpha=dnn_alpha, 
                                       batch_size=dnn_batch, random_state=42, early_stopping=True)

    st.divider()
    st.header("3. 模型验证与特征解析执行")
    model_name_short = model_choice.split()[0]
    
    if st.button(f"🚀 初始化并运行建模流程 ({model_name_short})", type="primary", disabled=(health_issues > 0)):
        with st.spinner(f"后台正执行矩阵标准化、降维过滤与 {model_name_short} 交叉验证拟合..."):
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
                
                # 计算当前 ROC 数据备查
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
                    st.toast(f"系统正在应用 Permutation Importance 算法核算特征贡献权重...", icon="🔍")
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
                log_text += f" ML Pipeline - Parameter Configuration Log \n"
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
                st.success(f"✅ 模型拟合流程执行完毕。{model_name_short} 预测器对象及评估指标已生成。")
            except Exception as e:
                st.error(f"建模过程发生异常中断: {e}")

    # --- 第四阶段：结果展示与多线 ROC 提取 ---
    if st.session_state['model_trained']:
        st.divider()
        st.header(f"📊 4. 模型效能评估与生物标志物权重解析")
        
        # 指标看板
        col_metric1, col_metric2, col_metric3 = st.columns(3)
        with col_metric1:
            st.metric(label="5-Fold CV Accuracy\n(泛化预测精度)", value=f"{st.session_state['cv_score']:.2%}")
        with col_metric2:
            auc_val = st.session_state['cv_auc']
            st.metric(label="5-Fold CV AUC Score\n(受试者工作特征曲线下面积)", value=f"{auc_val:.3f}" if not np.isnan(auc_val) else "N/A")
        with col_metric3:
            st.metric(label="Training Set Accuracy\n(内部数据集拟合精度)", value=f"{st.session_state['train_acc']:.2%}")
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
            st.subheader(f"Relative Feature Importance")
            df_plot = st.session_state['feature_importance_df']
            fig_bar, ax_bar = plt.subplots(figsize=(6, 5)) 
            sns.barplot(x='Importance', y='Feature', data=df_plot.head(15), palette='viridis', ax=ax_bar)
            ax_bar.set_xlabel(f"Normalized Contribution Score (0-1)", fontsize=10); ax_bar.set_ylabel("")
            plt.subplots_adjust(left=0.3); sns.despine()
            st.pyplot(fig_bar)

        # 多线 ROC 提取面板 
        st.markdown("### 📈 多模型 ROC 曲线对比分析库 (Benchmark)")
        st.info("💡 操作提示：您可将当前验证结果的 ROC 坐标轴数据保存至下方对比库中。通过变更上方模型参数或核心算法，并反复执行提取动作，即可生成涵盖多个学习器的基准测试 (Benchmark) 对比视图。")
        
        col_roc1, col_roc2, col_roc3 = st.columns([2, 1, 1])
        with col_roc1:
            custom_curve_name = st.text_input("为当前评估结果设定标识名称：", value=f"{st.session_state['model_name_short']} (AUC={st.session_state['current_roc_data']['auc']:.3f})")
        with col_roc2:
            st.write("") 
            if st.button("➕ 添加至对比分析库", type="primary"):
                st.session_state['roc_history'].append({
                    'name': custom_curve_name,
                    'fpr': st.session_state['current_roc_data']['fpr'],
                    'tpr': st.session_state['current_roc_data']['tpr'],
                    'auc': st.session_state['current_roc_data']['auc']
                })
                st.rerun()
        with col_roc3:
            st.write("") 
            if st.button("🗑️ 清空当前对比库") and len(st.session_state['roc_history']) > 0:
                st.session_state['roc_history'] = []
                st.rerun()
                
        if len(st.session_state['roc_history']) > 0:
            st.markdown("#### 基准模型诊断效能比对 (Model Benchmark Comparison)")
            fig_multi, ax_multi = plt.subplots(figsize=(8, 6))
            colors = sns.color_palette("Set1", n_colors=max(10, len(st.session_state['roc_history'])))
            
            for idx, roc_obj in enumerate(st.session_state['roc_history']):
                ax_multi.plot(roc_obj['fpr'], roc_obj['tpr'], color=colors[idx], lw=2, 
                              label=f"{roc_obj['name']}")
                
            ax_multi.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--')
            ax_multi.set_xlabel('False Positive Rate (1 - Specificity)', fontsize=12)
            ax_multi.set_ylabel('True Positive Rate (Sensitivity)', fontsize=12)
            ax_multi.set_title('Cross-Validated ROC Curve Comparison', fontsize=14)
            ax_multi.legend(loc="lower right", fontsize=10)
            sns.despine()
            
            col_chart_left, col_chart_mid, col_chart_right = st.columns([1, 2, 1])
            with col_chart_mid:
                st.pyplot(fig_multi)

        st.divider()
        st.header("📥 5. 分析资产与日志数据导出")
        col_dl1, col_dl2, col_dl3 = st.columns(3)
        buffer = io.BytesIO()
        joblib.dump({'pipeline': st.session_state['trained_pipeline'], 'label_encoder': st.session_state['label_encoder']}, buffer)
        
        with col_dl1:
            st.download_button(label=f"📦 下载算法序列化模型 (.pkl)", data=buffer.getvalue(), file_name=f"ml_model_pipeline.pkl", mime="application/octet-stream", help="导出预训练模型结构与权重。可用于独立队列验证评估或外部应用系统的部署。")
        with col_dl2:
            st.download_button(label="📝 下载模型构建配置日志 (.txt)", data=st.session_state['param_log'].encode('utf-8'), file_name=f"hyperparameters_log.txt", mime="text/plain", help="生成模型训练超参数与前置数据处理步骤审计日志，为研究方法的撰写提供基准参考。")
        with col_dl3:
            csv_data = st.session_state['feature_importance_df'].to_csv(index=False).encode('utf-8-sig')
            st.download_button(label="📊 下载标志物贡献度评分明细 (.csv)", data=csv_data, file_name=f"feature_weights.csv", mime="text/csv", help="提取包含算法模型内全量有效特征的相对贡献系数列表。适用于跨平台网络映射或下游生物学通路(Pathway)富集分析。")
