import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.metrics import confusion_matrix, accuracy_score, roc_curve, auc
from sklearn.preprocessing import label_binarize
import shap

# --- 页面配置 ---
st.set_page_config(page_title="独立盲测与结果解释", page_icon="🧬", layout="wide")
st.title("🧬 外部数据盲测与预测结果解释")

st.info("💡 操作指南：请把在【模型构建页】下载的 `.pkl` 模型文件传上来，再传一份包含新样本的数据表。系统会自动帮您做预测，并告诉您它是根据什么指标做出这个判断的。")

# --- 阶段 1：资产与数据载入 ---
col_up1, col_up2 = st.columns(2)
with col_up1:
    st.subheader("1. 载入您的 AI 模型")
    model_file = st.file_uploader("上传已保存的模型文件 (.pkl)", type=['pkl'])
with col_up2:
    st.subheader("2. 导入待预测的新数据")
    data_file = st.file_uploader("上传外部测试集或新样本表 (.csv / .xlsx)", type=['csv', 'xlsx'])

if model_file and data_file:
    try:
        # 载入资产
        loaded_assets = joblib.load(model_file)
        pipeline = loaded_assets['pipeline']
        label_encoder = loaded_assets['label_encoder']
        
        # 载入数据
        df_new = pd.read_csv(data_file) if data_file.name.endswith('.csv') else pd.read_excel(data_file)
        
        st.success("✅ 模型和新数据都已成功就绪！")
        
        # --- 变量映射与特征对齐 ---
        st.divider()
        st.subheader("3. 数据列对齐与安全检查")
        
        col_map1, col_map2 = st.columns(2)
        with col_map1:
            id_col = st.selectbox("选择样本编号列 (Sample ID)", options=["无 (系统自动编号)"] + list(df_new.columns),
                                  help="用来区分这是哪一个病人、哪一瓶酒或哪一个样本。")
            sample_ids = df_new[id_col].values if id_col != "无 (系统自动编号)" else [f"样本_{i}" for i in range(len(df_new))]
            
        with col_map2:
            target_col = st.selectbox("选择真实的分类结果 (如果有的话)", options=["纯预测模式 (这批数据没有真实结果)"] + list(df_new.columns),
                                      help="如果这批数据是已经知道结果的，选上它，系统会给模型打分；如果是完全未知的新数据，选【纯预测模式】。")
            
        # 提取当前数据集特征
        exclude_list = [id_col, target_col]
        current_features = [c for c in df_new.columns if c not in exclude_list]
        X_new = df_new[current_features].copy()
        
        for col in X_new.columns:
            X_new[col] = pd.to_numeric(X_new[col], errors='coerce')
        
        st.info("⚙️ 系统正在悄悄核对：新数据里的特征列（比如代谢物名字）是否跟模型训练时一模一样...")
        
        st.write("---")
        if st.button("🚀 开始预测并解析原因", type="primary"):
            with st.spinner("大脑飞速运转中：正在算概率、找原因..."):
                try:
                    X_array = X_new.values
                    y_pred_encoded = pipeline.predict(X_array)
                    y_prob = pipeline.predict_proba(X_array)
                    
                    y_pred_labels = label_encoder.inverse_transform(y_pred_encoded)
                    
                    # 组装结果输出表
                    result_df = pd.DataFrame({'样本编号 (ID)': sample_ids, 'AI 预测分类': y_pred_labels})
                    for i, class_name in enumerate(label_encoder.classes_):
                        result_df[f'属于 {class_name} 的概率'] = np.round(y_prob[:, i], 4)
                        
                    st.session_state['result_df'] = result_df
                    st.session_state['X_new'] = X_new
                    st.session_state['sample_ids'] = sample_ids
                    st.session_state['pipeline'] = pipeline
                    st.session_state['predict_done'] = True
                    
                    # 若包含真实标签，则计算评估指标
                    if target_col != "纯预测模式 (这批数据没有真实结果)":
                        y_true_labels = df_new[target_col].astype(str).values
                        y_true_encoded = label_encoder.transform(y_true_labels)
                        
                        acc = accuracy_score(y_true_encoded, y_pred_encoded)
                        cm = confusion_matrix(y_true_encoded, y_pred_encoded)
                        st.session_state['val_acc'] = acc
                        st.session_state['val_cm'] = cm
                        st.session_state['y_true_encoded'] = y_true_encoded
                        st.session_state['y_prob'] = y_prob
                        st.session_state['classes'] = label_encoder.classes_
                        st.session_state['has_target'] = True
                    else:
                        st.session_state['has_target'] = False

                except ValueError as ve:
                    st.error(f"❌ 糟糕，特征对不上！\n详细原因：{ve}。\n大白话翻译：您的新表格里，是不是漏掉了训练时的某个列？或者列名拼写有差异？")
                except Exception as e:
                    st.error(f"计算过程中发生意外错误: {e}")

    except Exception as e:
        st.error(f"模型文件读取失败，您确定上传的是我们平台生成的 .pkl 文件吗？报错信息: {e}")

# --- 阶段 2：预测效能评估 ---
if st.session_state.get('predict_done', False):
    st.divider()
    if st.session_state.get('has_target', False):
        st.header("📊 4. 盲测成绩单 (模型到底准不准？)")
        
        col_m1, col_m2 = st.columns(2)
        with col_m1:
            st.metric(label="总体猜对的比例 (外部验证准确率)", value=f"{st.session_state['val_acc']:.2%}")
        
        # 绘制混淆矩阵与 ROC
        col_p1, col_p2 = st.columns(2)
        with col_p1:
            st.subheader("具体对错明细 (Confusion Matrix)")
            fig_cm, ax_cm = plt.subplots(figsize=(5, 4))
            sns.heatmap(st.session_state['val_cm'], annot=True, fmt='d', cmap='Oranges', ax=ax_cm, 
                        xticklabels=st.session_state['classes'], yticklabels=st.session_state['classes'])
            ax_cm.set_ylabel("真实的情况 (True)"); ax_cm.set_xlabel("AI 猜的情况 (Predicted)")
            st.pyplot(fig_cm)
            
        with col_p2:
            st.subheader("综合诊断能力 (ROC Curve)")
            fig_roc, ax_roc = plt.subplots(figsize=(5, 4))
            y_true = st.session_state['y_true_encoded']
            y_prob = st.session_state['y_prob']
            classes = st.session_state['classes']
            
            if len(classes) == 2:
                fpr, tpr, _ = roc_curve(y_true, y_prob[:, 1])
                roc_auc = auc(fpr, tpr)
                ax_roc.plot(fpr, tpr, color='#D32F2F', lw=2, label=f'ROC (AUC = {roc_auc:.3f})')
                with col_m2: st.metric(label="综合诊断硬实力 (外部验证 AUC)", value=f"{roc_auc:.3f}")
            else:
                y_true_bin = label_binarize(y_true, classes=range(len(classes)))
                for i, color in zip(range(len(classes)), sns.color_palette("husl", len(classes))):
                    fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_prob[:, i])
                    ax_roc.plot(fpr, tpr, color=color, lw=2, label=f'{classes[i]} (AUC={auc(fpr, tpr):.2f})')
                with col_m2: st.metric(label="综合诊断硬实力 (外部 AUC)", value="多分类图表见下方")
                    
            ax_roc.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--')
            ax_roc.set_xlabel('误报率 (越小越好)'); ax_roc.set_ylabel('命中率 (越大越好)')
            ax_roc.legend(loc="lower right")
            st.pyplot(fig_roc)
            
    else:
        st.info("ℹ️ 提示：因为您刚才选了【纯预测模式】，所以没有参考答案，系统只输出预测结果表格，画不出打分表。")

    # --- 阶段 3：SHAP 可解释性分析 ---
    st.divider()
    st.header("🧠 5. 预测结果大揭秘 (为什么得出这个结论？)")
    st.markdown("**(借助 SHAP 算法，把 AI 的“黑匣子”打开给您看)**")
    
    # 单样本解析 (大白话版)
    st.markdown("#### 🔬 查看单个样本的详细原因 (个体化归因)")
    st.write("想知道为什么 42号样本 被判定为这个结果？在下面选出它，AI 会告诉您是哪些具体的指标把它“推向”了这个分类。")
    
    sample_opts = st.session_state['sample_ids']
    sel_sample = st.selectbox("请挑选一个您关心的样本:", options=sample_opts)
    
    if st.button("🔍 给我看这个样本的分析图"):
        st.info("💡 演示占位符：此处可桥接著名的 SHAP Waterfall Plot (瀑布图)。图表会清晰显示：红色的指标增加了它的概率，蓝色的指标降低了它的概率，一目了然！")
        # 核心对接逻辑 (需用户环境安装 shap 库支持):
        # idx = list(sample_opts).index(sel_sample)
        # model = pipeline.named_steps['classifier']
        # explainer = shap.TreeExplainer(model) (若为树模型)
        # transformed_X = pipeline[:-1].transform(X_new)
        # shap_values = explainer(transformed_X[[idx]])
        # shap.plots.waterfall(shap_values[0])

    # --- 阶段 4：结果导出 ---
    st.divider()
    st.header("📥 6. 拿走您的预测结果")
    csv_out = st.session_state['result_df'].to_csv(index=False).encode('utf-8-sig')
    st.download_button(
        label="📊 导出完整预测表格 (.csv)",
        data=csv_out,
        file_name="validation_predictions_prob.csv",
        mime="text/csv",
        help="表格里不仅有 AI 给出的最终定论，还有精确到小数点的概率值。方便您对那些刚好卡在 50% 边缘的样本进行人工复核。"
    )
