import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import io
import base64
from sklearn.metrics import accuracy_score, confusion_matrix
import shap

# 尝试导入我们之前写的特征筛选类 (反序列化 .pkl 时需要)
try:
    from utils.data_processor import MassSpecFeatureSelector
except ImportError:
    from sklearn.base import BaseEstimator, TransformerMixin
    from sklearn.feature_selection import SelectKBest, SelectFdr, SelectFwe, f_classif
    class MassSpecFeatureSelector(BaseEstimator, TransformerMixin):
        pass

# --- 辅助函数：将高清 Matplotlib 图表转为 Base64 ---
def fig_to_base64(fig, dpi=300):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=dpi, bbox_inches='tight')
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')

# --- 页面初始化与状态管理 ---
st.set_page_config(page_title="模型应用终端", page_icon="🔬", layout="wide")

# 初始化防刷新的状态锁
if 'prediction_done' not in st.session_state:
    st.session_state['prediction_done'] = False

st.title("🔬 临床模型应用与诊断终端")
st.markdown("加载已训练的智能核心，对新批次数据/临床样本进行预测，并生成带高清图表（300dpi）的可解释性诊断报告。")

# --- 第一阶段：加载模型与数据 ---
col1, col2 = st.columns(2)
with col1:
    st.header("1. 加载智能核心")
    model_file = st.file_uploader("请上传在训练工场下载的 .pkl 模型文件", type=['pkl'])

with col2:
    st.header("2. 上传待预测样本")
    data_file = st.file_uploader("请上传特征矩阵文件 (支持 .csv 或 .xlsx)", type=['csv', 'xlsx'])

if model_file and data_file:
    try:
        model_package = joblib.load(model_file)
        pipeline = model_package['pipeline']
        label_encoder = model_package['label_encoder']
        st.success("✅ 智能核心 (.pkl) 加载成功！")
    except Exception as e:
        st.error(f"模型解析失败: {e}")
        st.stop()
        
    try:
        df = pd.read_csv(data_file) if data_file.name.endswith('.csv') else pd.read_excel(data_file)
        st.success(f"✅ 数据加载成功！包含 {df.shape[0]} 个样本。")
    except Exception as e:
        st.error(f"数据读取失败: {e}")
        st.stop()

    st.divider()
    
    # --- 第二阶段：参数对齐 ---
    st.header("3. 数据字段映射")
    st.info("请告诉系统哪些列是样本编号，哪些是特征。系统将自动过滤脏数据并与模型匹配。")
    
    col_id, col_label = st.columns(2)
    with col_id:
        id_col = st.selectbox("选择【样本编号】列 (用于输出预测报告)", options=df.columns)
    with col_label:
        has_labels = st.checkbox("本次上传的数据包含真实标签 (用于独立测试集对答案验证)", value=False)
        if has_labels:
            label_col = st.selectbox("选择【真实标签】列", options=[c for c in df.columns if c != id_col])
        else:
            label_col = None

    # --- 第三阶段：执行预测 (核心计算) ---
    if st.button("🚀 开始预测并生成临床报告", type="primary"):
        with st.spinner("正在执行脏数据清洗、特征提取、预测及 SHAP 解析..."):
            try:
                exclude_cols = [id_col]
                if has_labels: exclude_cols.append(label_col)
                feature_cols = [c for c in df.columns if c not in exclude_cols]
                
                X_df = df[feature_cols].copy()
                for col in X_df.columns:
                    X_df[col] = pd.to_numeric(X_df[col], errors='coerce')
                X_raw = X_df.values
                
                y_pred_encoded = pipeline.predict(X_raw)
                y_pred_text = label_encoder.inverse_transform(y_pred_encoded)
                
                try:
                    y_prob = pipeline.predict_proba(X_raw)
                    prob_max = np.max(y_prob, axis=1)
                except:
                    prob_max = ["N/A"] * len(y_pred_text)
                    
                result_df = pd.DataFrame({
                    'Sample_ID': df[id_col],
                    'Predicted_Class': y_pred_text,
                    'Confidence': prob_max
                })
                if has_labels:
                    result_df.insert(1, 'True_Class', df[label_col])
                
                preprocessor = pipeline[:-1]
                classifier = pipeline.named_steps['classifier']
                X_processed = preprocessor.transform(X_raw)
                
                survived_indices = pipeline.named_steps['feature_selector'].final_indices_
                survived_features = np.array(feature_cols)[survived_indices]
                
                html_plots = {}
                model_type = type(classifier).__name__
                
                st.session_state['has_labels'] = has_labels
                st.session_state['model_type'] = model_type
                
                if has_labels:
                    y_true_encoded = label_encoder.transform(df[label_col].astype(str))
                    acc = accuracy_score(y_true_encoded, y_pred_encoded)
                    
                    fig_cm, ax_cm = plt.subplots(figsize=(6, 5), dpi=300)
                    cm = confusion_matrix(y_true_encoded, y_pred_encoded)
                    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax_cm, 
                                xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
                    ax_cm.set_title("Confusion Matrix", fontsize=14, pad=10)
                    ax_cm.set_ylabel("True Label"); ax_cm.set_xlabel("Predicted Label")
                    plt.tight_layout()
                    
                    html_plots['confusion_matrix'] = fig_to_base64(fig_cm)
                    st.session_state['acc'] = acc
                    st.session_state['fig_cm'] = fig_cm
                
                if model_type in ['XGBClassifier', 'RandomForestClassifier']:
                    st.toast("正在计算 SHAP TreeExplainer...", icon="🌳")
                    explainer = shap.TreeExplainer(classifier)
                    shap_values = explainer.shap_values(X_processed)
                    
                    fig_shap, ax_shap = plt.subplots(figsize=(10, 8), dpi=300)
                    if isinstance(shap_values, list):
                        shap_val_to_plot = shap_values[1]
                    else:
                        shap_val_to_plot = shap_values
                        
                    shap.summary_plot(shap_val_to_plot, X_processed, feature_names=survived_features, show=False)
                    plt.title("SHAP Global Feature Importance & Impact", fontsize=16, pad=15, fontweight='bold')
                    plt.tight_layout()
                    
                    html_plots['shap_summary'] = fig_to_base64(fig_shap)
                    st.session_state['fig_shap'] = fig_shap
                else:
                    html_plots['shap_summary'] = None
                
                # --- 生成 HTML 并存入状态 ---
                html_template = f"""
                <html>
                <head>
                    <meta charset="utf-8">
                    <title>Shimadzu Clinical MS AI Report</title>
                    <style>
                        body {{ font-family: Arial, sans-serif; margin: 40px; color: #333; }}
                        h1, h2 {{ color: #01579B; border-bottom: 2px solid #03A9F4; padding-bottom: 10px; }}
                        table {{ border-collapse: collapse; width: 100%; margin-top: 20px; font-size: 14px; }}
                        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: center; }}
                        th {{ background-color: #f2f2f2; }}
                        .img-container {{ text-align: center; margin: 30px 0; }}
                        .img-container img {{ max-width: 90%; border: 1px solid #eee; box-shadow: 2px 2px 12px #aaa; }}
                    </style>
                </head>
                <body>
                    <h1>Shimadzu Multi-Omics AI Diagnostic Report</h1>
                    <p><b>Model Engine:</b> {model_type}</p>
                    <p><b>Total Samples Processed:</b> {len(result_df)}</p>
                """
                if has_labels:
                    html_template += f"<h2>1. Model Performance Validation</h2>"
                    html_template += f"<p><b>Test Set Accuracy:</b> {acc:.2%}</p>"
                    html_template += f'<div class="img-container"><h3>Confusion Matrix (300 DPI)</h3><img src="data:image/png;base64,{html_plots["confusion_matrix"]}" alt="Confusion Matrix"></div>'
                
                if html_plots.get('shap_summary'):
                    html_template += f"""
                    <h2>2. SHAP Interpretability Analysis</h2>
                    <p>This plot shows how each biomarker contributes to the model's decision for the current cohort. <br>
                    <i>(Right click the image to save the high-resolution 300 DPI version for publication)</i></p>
                    <div class="img-container"><img src="data:image/png;base64,{html_plots['shap_summary']}" alt="SHAP Summary"></div>
                    """
                html_template += f"<h2>3. Detailed Prediction Results</h2>{result_df.to_html(index=False)}</body></html>"
                
                # 锁住所有结果！
                st.session_state['result_df'] = result_df
                st.session_state['html_plots'] = html_plots
                st.session_state['html_report'] = html_template
                st.session_state['csv_report'] = result_df.to_csv(index=False).encode('utf-8-sig')
                
                # 标记计算完成
                st.session_state['prediction_done'] = True
                
            except Exception as e:
                st.error(f"预测或生成报告时发生错误: {e}")

# --- 第四阶段：独立渲染结果展示与下载区 (不怕刷新！) ---
if st.session_state.get('prediction_done'):
    st.divider()
    st.header("📊 4. 预测结果与诊断报告")
    
    if st.session_state['has_labels']:
        st.metric("Test Set Accuracy (测试集准确率)", f"{st.session_state['acc']:.2%}")
        col_plot1, col_plot2 = st.columns(2)
        with col_plot1:
            st.subheader("混淆矩阵")
            st.pyplot(st.session_state['fig_cm'])
        with col_plot2:
            if st.session_state['html_plots'].get('shap_summary'):
                st.subheader("SHAP 生物标志物贡献解析")
                st.pyplot(st.session_state['fig_shap'])
            else:
                st.info(f"当前模型引擎 ({st.session_state['model_type']}) 不属于树模型，已跳过耗时的 SHAP 运算。")
    else:
        if st.session_state['html_plots'].get('shap_summary'):
            st.subheader("当前批次样本 SHAP 特征贡献解析")
            st.pyplot(st.session_state['fig_shap'])
    
    st.subheader("🔬 详细预测清单")
    st.dataframe(st.session_state['result_df'])
    
    st.divider()
    st.header("📥 5. 导出临床交付文件")
    
    col_d1, col_d2 = st.columns(2)
    with col_d1:
        st.download_button(
            label="📄 1. 下载临床检验结果表格 (.csv)",
            data=st.session_state['csv_report'],
            file_name="Clinical_Predictions.csv",
            mime="text/csv"
        )
    with col_d2:
        st.download_button(
            label="📑 2. 下载高清临床诊断报告 (HTML, 图片300dpi)",
            data=st.session_state['html_report'],
            file_name="Shimadzu_Clinical_Report.html",
            mime="text/html"
        )
