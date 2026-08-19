import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score
from xgboost import XGBClassifier
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
            self.selection_method = selection_method
            self.n_features_to_select = n_features_to_select
            self.alpha = alpha
            self.remove_redundant = remove_redundant
            self.corr_threshold = corr_threshold
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
                        if not selected_subset:
                            selected_subset.append(idx)
                        else:
                            corr_vals = np.abs([np.corrcoef(X_selected[:, idx], X_selected[:, sel])[0, 1] for sel in selected_subset])
                            if np.max(corr_vals) <= self.corr_threshold:
                                selected_subset.append(idx)
                    self.final_indices_ = current_indices[selected_subset]
                else: self.final_indices_ = current_indices
            except:
                self.final_indices_ = current_indices
            return self
        def transform(self, X):
            return X.values[:, self.final_indices_] if hasattr(X, 'values') else X[:, self.final_indices_]

# --- 页面初始化与状态管理 ---
st.set_page_config(page_title="模型训练工场", page_icon="🛠️", layout="wide")

if 'data_loaded' not in st.session_state:
    st.session_state['data_loaded'] = False
if 'model_trained' not in st.session_state:
    st.session_state['model_trained'] = False
if 'trained_pipeline' not in st.session_state:
    st.session_state['trained_pipeline'] = None
if 'df_raw' not in st.session_state:
    st.session_state['df_raw'] = None

st.title("🛠️ XGBoost 模型训练工场")
st.markdown("上传实验数据，平台将自动执行数据清洗、统计学降维与去冗余，并训练出专业的 XGBoost 预测模型。")

# --- 第一阶段：数据上传 ---
st.header("1. 上传与核对数据")
uploaded_file = st.file_uploader("请上传带有分组标签（如健康/患病）的特征矩阵文件 (支持 .csv 或 .xlsx)", type=['csv', 'xlsx'])

if uploaded_file is not None:
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
            
        st.session_state['df_raw'] = df
        st.session_state['data_loaded'] = True
        st.success(f"✅ 文件 '{uploaded_file.name}' 上传成功！包含 {df.shape[0]} 个样本，{df.shape[1]} 列信息。")
        
        with st.expander("👀 预览数据前 5 行"):
            st.dataframe(df.head())
            
    except Exception as e:
        st.error(f"读取文件失败: {e}")

# --- 第二阶段：指定目标与参数设置 ---
if st.session_state['data_loaded']:
    df = st.session_state['df_raw']
    
    st.divider()
    st.header("2. 设定目标与清洗参数")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🎯 目标与特征范围")
        target_col = st.selectbox("1. 您希望模型预测什么？(选择目标列 / Label)", options=df.columns)
        
        candidate_features = [c for c in df.columns if c != target_col]
        exclude_cols = st.multiselect("2. 排除非特征列 (如样本编号、日期等)", options=candidate_features)
        
        # 提取用于训练的纯特征列名
        feature_cols = [c for c in candidate_features if c not in exclude_cols]
        
    with col2:
        st.subheader("⚙️ 行业预置清洗参数")
        preset_choice = st.selectbox(
            "选择预处理方案:",
            ("【自动挡】代谢组学常规推荐 (XGBoost优选)", "【手动挡】自定义高级设置")
        )
        
        if preset_choice == "【手动挡】自定义高级设置":
            sel_method = st.selectbox("统计筛选方法", ['fdr', 'kbest', 'fwe'])
            alpha_val = st.number_input("统计显著性阈值 (Alpha/P-value)", value=0.05, step=0.01)
            corr_thresh = st.slider("去冗余相关系数阈值", 0.5, 1.0, 0.9)
        else:
            sel_method = 'fdr'
            alpha_val = 0.05
            corr_thresh = 0.9
            st.info(f"**自动采用:** 检验方法=`{sel_method}`, 显著性=`{alpha_val}`, 冗余上限=`{corr_thresh}`")

    # --- 第三阶段：触发训练 ---
    st.divider()
    st.header("3. 训练与打包")
    
    if st.button("🚀 开始训练 XGBoost 模型并打包", type="primary"):
        if len(feature_cols) < 2:
            st.error("特征列太少，请检查是否排除了过多的列！")
        else:
            with st.spinner("正在执行降维和 XGBoost 模型训练，请稍候..."):
                try:
                    X = df[feature_cols].values
                    y_raw = df[target_col].values
                    
                    # 标签自动转换 (将文字标签转为 0, 1, 2...)
                    le = LabelEncoder()
                    y = le.fit_transform(y_raw)
                    st.session_state['label_encoder'] = le 
                    
                    # 构建 Pipeline
                    ms_pipeline = Pipeline(steps=[
                        ('imputer', SimpleImputer(strategy='median')), 
                        ('scaler', StandardScaler()), 
                        ('feature_selector', MassSpecFeatureSelector(
                            selection_method=sel_method,
                            alpha=alpha_val,
                            remove_redundant=True,
                            corr_threshold=corr_thresh
                        )),
                        ('classifier', XGBClassifier(n_estimators=100, learning_rate=0.1, random_state=42, use_label_encoder=False, eval_metric='logloss')) 
                    ])
                    
                    # 拟合与评估
                    ms_pipeline.fit(X, y)
                    scores = cross_val_score(ms_pipeline, X, y, cv=5)
                    st.session_state['cv_score'] = np.mean(scores)
                    
                    # 提取幸存特征与重要性
                    survived_indices = ms_pipeline.named_steps['feature_selector'].final_indices_
                    survived_features = np.array(feature_cols)[survived_indices]
                    importances = ms_pipeline.named_steps['classifier'].feature_importances_
                    
                    imp_df = pd.DataFrame({
                        'Feature': survived_features,
                        'Importance': importances
                    }).sort_values(by='Importance', ascending=False)
                    
                    # 保存至状态
                    st.session_state['feature_importance_df'] = imp_df
                    st.session_state['trained_pipeline'] = ms_pipeline
                    st.session_state['model_trained'] = True
                    
                    st.success("✅ 模型训练完成！")
                except Exception as e:
                    st.error(f"训练过程中发生错误: {e}")

    # --- 第四阶段：结果展示与下载 ---
    if st.session_state['model_trained']:
        st.divider()
        st.header("📊 4. 训练结果与生物标志物鉴定")
        
        st.metric(label="XGBoost 模型交叉验证准确率 (5-Fold CV)", value=f"{st.session_state['cv_score']:.2%}")
        
        # 绘图
        df_plot = st.session_state['feature_importance_df']
        top_n = min(20, len(df_plot))
        
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(x='Importance', y='Feature', data=df_plot.head(top_n), palette='viridis', ax=ax)
        ax.set_title(f"Top {top_n} 核心标志物特征重要性排名", fontsize=14, pad=15)
        ax.set_xlabel("XGBoost 权重贡献度", fontsize=12)
        ax.set_ylabel("代谢物 / 特征名称", fontsize=12)
        sns.despine()
        plt.tight_layout()
        st.pyplot(fig)
        
        # 打包供下一个页面使用的对象
        model_package = {
            'pipeline': st.session_state['trained_pipeline'],
            'label_encoder': st.session_state['label_encoder']
        }
        
        buffer = io.BytesIO()
        joblib.dump(model_package, buffer)
        
        st.download_button(
            label="📥 下载已训练的智能核心 (.pkl)",
            data=buffer.getvalue(),
            file_name="shimadzu_xgboost_core.pkl",
            mime="application/octet-stream",
            help="包含所有的清洗规则、你的降维算法、XGBoost树结构以及标签解码器。可在【模型应用终端】直接加载预测。"
        )
