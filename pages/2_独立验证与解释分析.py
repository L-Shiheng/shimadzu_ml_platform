import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.metrics import confusion_matrix, accuracy_score, roc_curve, auc
from sklearn.preprocessing import label_binarize

# --- 页面配置 ---
st.set_page_config(page_title="独立验证与解释分析", page_icon="🧬", layout="wide")
st.title("🧬 外部数据验证与预测结果分析")

st.info("💡 操作提示：请上传由【模型构建】阶段导出的算法模型文件 (.pkl) 及新的待测数据矩阵。系统将执行预测评估，并提供特征归因分析。")

# --- 阶段 1：资产与数据载入 ---
col_up1, col_up2 = st.columns(2)
with col_up1:
    st.subheader("1. 导入预测模型")
    model_file = st.file_uploader("上传已训练的模型文件 (.pkl)", type=['pkl'])
with col_up2:
    st.subheader("2. 导入待测数据集")
    data_file = st.file_uploader("上传外部验证集或新样本数据 (.csv / .xlsx)", type=['csv', 'xlsx'])

if model_file and data_file:
    try:
        # 载入资产
        loaded_assets = joblib.load(model_file)
        pipeline = loaded_assets['pipeline']
        label_encoder = loaded_assets['label_encoder']
        
        # 载入数据
        df_new = pd.read_csv(data_file) if data_file.name.endswith('.csv') else pd.read_excel(data_file)
        
        st.success("✅ 模型对象与数据矩阵载入成功。")
        
        # --- 变量映射与特征对齐 ---
        st.divider()
        st.subheader("3. 数据字段映射与一致性校验")
        
        col_map1, col_map2 = st.columns(2)
        with col_map1:
            id_col = st.selectbox("选择样本标识列 (Sample ID)", options=["无 (系统自动生成索引)"] + list(df_new.columns),
                                  help="用于在最终导出报告中唯一标识每个观测样本。")
            sample_ids = df_new[id_col].values if id_col != "无 (系统自动生成索引)" else [f"样本_{i}" for i in range(len(df_new))]
            
        with col_map2:
            target_col = st.selectbox("选择目标分类变量 (可选)", options=["预测模式 (无实际目标标签)"] + list(df_new.columns),
                                      help="若该数据集包含实际结果，系统将进行准确率评估；若为全新盲测数据，请选择【预测模式】。")
            
        # 提取当前数据集特征
        exclude_list = [id_col, target_col]
        current_features = [c for c in df_new.columns if c not in exclude_list]
        X_new = df_new[current_features].copy()
        
        for col in X_new.columns:
            X_new[col] = pd.to_numeric(X_new[col], errors='coerce')
        
        st.info("⚙️ 校验中：系统正核对输入矩阵特征维度是否与原训练管线完全匹配...")
        
        st.write("---")
        if st.button("🚀 运行预测与特征分析", type="primary"):
            with st.spinner("系统正在执行算法推理与数据处理，请稍候..."):
                try:
                    X_array = X_new.values
                    y_pred_encoded = pipeline.predict(X_array)
                    y_prob = pipeline.predict_proba(X_array)
                    
                    y_pred_labels = label_encoder.inverse_transform(y_pred_encoded)
                    
                    # 组装结果输出表
                    result_df = pd.DataFrame({'样本标识 (ID)': sample_ids, '预测分类': y_pred_labels})
                    for i, class_name in enumerate(label_encoder.classes_):
                        result_df[f'分类概率 ({class_name})'] = np.round(y_prob[:, i], 4)
                        
                    st.session_state['result_df'] = result_df
                    st.session_state['X_new'] = X_new
                    st.session_state['sample_ids'] = sample_ids
                    st.session_state['pipeline'] = pipeline
                    st.session_state['predict_done'] = True
                    
                    # 若包含真实标签，则计算评估指标
                    if target_col != "预测模式 (无实际目标标签)":
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
                    # ==========================================
                    # ☁️ 中心化云端日志记录模块 (验证与预测专用版)
                    # ==========================================
                    try:
                        import datetime
                        import uuid
                        import io
                        from github import Github
                        from github.GithubException import UnknownObjectException
                        
                        if 'session_id' not in st.session_state:
                            st.session_state['session_id'] = str(uuid.uuid4())[:8]
                            
                        # 自动抓取当前页面的运行数据
                        new_log_data = [{
                            "时间": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            "页面": "独立验证与解释分析", 
                            "操作": "执行外部验证与模型预测", 
                            "模型": model_file.name,          # 🪄 神奇之处：自动提取上传的 .pkl 文件名
                            "样本量": int(X_new.shape[0]),    # 🪄 神奇之处：自动计算上传的数据行数
                            "会话ID": st.session_state['session_id'],
                            "使用者": "罗世恒"
                        }]
                        
                        GITHUB_TOKEN = st.secrets["GITHUB_TOKEN"] 
                        REPO_NAME = "L-Shiheng/Central_App_Logs"
                        APP_ID = "组学建模平台" 
                        FILE_PATH = f"{APP_ID}_运行日志.csv" 
                        
                        g = Github(GITHUB_TOKEN)
                        repo = g.get_repo(REPO_NAME)
                        
                        try:
                            contents = repo.get_contents(FILE_PATH)
                            df_old = pd.read_csv(io.StringIO(contents.decoded_content.decode('utf-8')))
                            df_combined = pd.concat([df_old, pd.DataFrame(new_log_data)], ignore_index=True)
                            csv_data = df_combined.to_csv(index=False)
                            repo.update_file(contents.path, f"🤖 追加预测日志 - {APP_ID}", csv_data, contents.sha)
                        except UnknownObjectException:
                            csv_data = pd.DataFrame(new_log_data).to_csv(index=False)
                            repo.create_file(FILE_PATH, f"🤖 初始化日志库 - {APP_ID}", csv_data)
                            
                        # 这边为了不打扰主流程，就不弹 toast 提示了，让它真正“静默”
                        
                    except Exception as e:
                        st.error(f"🚨 日志同步失败: {e}") 
                    # ==========================================

                except ValueError as ve:
                    st.error(f"❌ 矩阵特征映射失败。\n系统提示：输入数据的特征列与原模型不一致。请检查是否存在拼写差异或特征遗漏。详细错误：{ve}")
                except Exception as e:
                    st.error(f"运算过程中发生异常中断: {e}")

    except Exception as e:
        st.error(f"模型解析异常，请确保上传了平台生成的合法 .pkl 文件。错误详情: {e}")

# --- 阶段 2：预测效能评估 ---
if st.session_state.get('predict_done', False):
    st.divider()
    if st.session_state.get('has_target', False):
        st.header("📊 4. 模型评估报告 (Validation Metrics)")
        
        col_m1, col_m2 = st.columns(2)
        with col_m1:
            st.metric(label="预测准确率 (Validation Accuracy)", value=f"{st.session_state['val_acc']:.2%}")
        
        # 绘制混淆矩阵与 ROC
        col_p1, col_p2 = st.columns(2)
        with col_p1:
            st.subheader("混淆矩阵 (Confusion Matrix)")
            fig_cm, ax_cm = plt.subplots(figsize=(5, 4))
            sns.heatmap(st.session_state['val_cm'], annot=True, fmt='d', cmap='Oranges', ax=ax_cm, 
                        xticklabels=st.session_state['classes'], yticklabels=st.session_state['classes'])
            ax_cm.set_ylabel("True Label"); ax_cm.set_xlabel("Predicted Label")
            st.pyplot(fig_cm)
            
        with col_p2:
            st.subheader("ROC 曲线与受试者工作特征 (ROC Curve)")
            fig_roc, ax_roc = plt.subplots(figsize=(5, 4))
            y_true = st.session_state['y_true_encoded']
            y_prob = st.session_state['y_prob']
            classes = st.session_state['classes']
            
            if len(classes) == 2:
                fpr, tpr, _ = roc_curve(y_true, y_prob[:, 1])
                roc_auc = auc(fpr, tpr)
                ax_roc.plot(fpr, tpr, color='#D32F2F', lw=2, label=f'ROC (AUC = {roc_auc:.3f})')
                with col_m2: st.metric(label="受试者工作特征曲线下面积 (AUC)", value=f"{roc_auc:.3f}")
            else:
                y_true_bin = label_binarize(y_true, classes=range(len(classes)))
                for i, color in zip(range(len(classes)), sns.color_palette("husl", len(classes))):
                    fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_prob[:, i])
                    ax_roc.plot(fpr, tpr, color=color, lw=2, label=f'{classes[i]} (AUC={auc(fpr, tpr):.2f})')
                with col_m2: st.metric(label="受试者工作特征曲线下面积 (AUC)", value="见多分类曲线集")
                    
            ax_roc.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--')
            ax_roc.set_xlabel('FPR'); ax_roc.set_ylabel('TPR')
            ax_roc.legend(loc="lower right")
            st.pyplot(fig_roc)
            
    else:
        st.info("ℹ️ 系统提示：当前任务为“预测模式”，因未提供实际参照标签，系统仅输出各样本的分类判定结果。")

    # --- 阶段 3：SHAP 可解释性分析 ---
    st.divider()
    st.header("🧠 5. 特征归因分析 (SHAP 解释)")
    st.markdown("**(借助 SHapley Additive exPlanations 算法解析模型判定依据)**")
    
    # 1. 全局特征归因分析
    st.markdown("#### 🌐 全局特征影响汇总 (Global Summary)")
    st.write("计算当前整批验证集数据，生成 SHAP 特征汇总图。用于观察哪些特征对全局预测贡献最大，以及特征数值高低如何影响分类倾向。")
    
    if st.button("📊 生成全局特征影响图"):
        with st.spinner("系统正在对全量测试数据进行矩阵积分运算，请稍候..."):
            try:
                import shap
                pipeline = st.session_state['pipeline']
                model = pipeline.named_steps['classifier']
                X_transformed = pipeline[:-1].transform(st.session_state['X_new'])
                
                feature_selector = pipeline.named_steps['feature_selector']
                survived_indices = feature_selector.final_indices_
                survived_features = np.array(st.session_state['X_new'].columns)[survived_indices]
                
                model_name = type(model).__name__
                
                if "XGB" in model_name or "RandomForest" in model_name:
                    plt.clf() # 清空画板防止重影
                    explainer = shap.TreeExplainer(model)
                    shap_values = explainer(X_transformed)
                    shap_values.feature_names = list(survived_features)
                    
                    fig = plt.figure(figsize=(8, 6))
                    if len(shap_values.shape) == 3: # 多分类
                        shap.summary_plot(shap_values, X_transformed, feature_names=list(survived_features), show=False)
                    else: # 二分类
                        shap.plots.beeswarm(shap_values, show=False)
                    
                    st.pyplot(plt.gcf())
                    st.success("✅ 全局分析生成完毕。")
                else:
                    st.warning("⚠️ 平台说明：当前使用的 SVM 或 MLP 模型不支持快速全局 SHAP 渲染。")
                    
            except ImportError:
                st.error("❌ 缺失组件模块：您的环境中未安装 `shap` 依赖库。")
            except Exception as e:
                st.error(f"❌ 解析运算过程中发生计算异常: {e}")

    st.write("---")

    # 2. 单样本局部解析
    st.markdown("#### 🔬 单样本预测原理解析 (Local Explainer)")
    st.write("请选择特定的观测样本，系统将生成瀑布图（Waterfall Plot），展示各项特征数值是如何推导并得出当前预测分类的。")
    
    sample_opts = st.session_state['sample_ids']
    sel_sample = st.selectbox("请选择待解析样本:", options=sample_opts)
    
    if st.button("🔍 运行单样本解析"):
        with st.spinner("系统正在进行底层特征积分推导，请稍候..."):
            try:
                import shap
                idx = list(sample_opts).index(sel_sample)
                
                pipeline = st.session_state['pipeline']
                model = pipeline.named_steps['classifier']
                
                X_transformed = pipeline[:-1].transform(st.session_state['X_new'])
                sample_data = X_transformed[[idx]]
                
                feature_selector = pipeline.named_steps['feature_selector']
                survived_indices = feature_selector.final_indices_
                survived_features = np.array(st.session_state['X_new'].columns)[survived_indices]
                
                model_name = type(model).__name__
                
                if "XGB" in model_name or "RandomForest" in model_name:
                    plt.clf() # 清空画板防止重影
                    
                    explainer = shap.TreeExplainer(model)
                    shap_values = explainer(sample_data)
                    shap_values.feature_names = list(survived_features)
                    
                    if len(shap_values.shape) == 3: # 多分类
                        pred_class_idx = pipeline.predict(st.session_state['X_new'].values[[idx]])[0]
                        shap.plots.waterfall(shap_values[0, :, pred_class_idx], show=False)
                    else: # 二分类
                        shap.plots.waterfall(shap_values[0], show=False)
                        
                    st.pyplot(plt.gcf()) # 强行抓取底层的全局画板
                    st.success(f"✅ 解析完成。上方图表展示了样本【{sel_sample}】各变量对最终分类结果的正负向贡献。")
                    
                else:
                    st.warning("⚠️ 平台说明：当前训练底层使用的是 SVM 或 MLP 架构。因算力与算法兼容性限制，目前平台仅支持为 XGBoost 和 随机森林 生成 SHAP 瀑布图。")
                    
            except ImportError:
                st.error("❌ 缺失组件模块：您的环境中未安装 `shap` 依赖库。请在终端执行 `pip install shap`。")
            except Exception as e:
                st.error(f"❌ 解析运算过程中发生计算异常: {e}")

    # --- 阶段 4：结果导出 ---
    st.divider()
    st.header("📥 6. 预测结果与明细导出")
    csv_out = st.session_state['result_df'].to_csv(index=False).encode('utf-8-sig')
    st.download_button(
        label="📊 导出预测分类与概率分布表 (.csv)",
        data=csv_out,
        file_name="validation_predictions_prob.csv",
        mime="text/csv",
        help="导出完整的预测结果矩阵，其中包含系统给出的离散分类结论及对应的连续概率分布。"
    )
