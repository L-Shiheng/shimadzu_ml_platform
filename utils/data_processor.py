import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import SelectKBest, SelectFdr, SelectFwe, f_classif

class MassSpecFeatureSelector(BaseEstimator, TransformerMixin):
    """
    将自定义质谱特征筛选与去冗余代码，包装为标准的 sklearn 转换器。
    该转换器能在 fit 时记忆保留特征的索引，并在 transform 时应用这些索引。
    """
    def __init__(self, selection_method='fdr', n_features_to_select=100, 
                 alpha=0.05, remove_redundant=True, corr_threshold=0.9):
        self.selection_method = selection_method
        self.n_features_to_select = n_features_to_select
        self.alpha = alpha
        self.remove_redundant = remove_redundant
        self.corr_threshold = corr_threshold
        
        # 将被训练后赋值
        self.final_indices_ = None

    def fit(self, X, y):
        # 如果 X 是 DataFrame，转换为 numpy array 以保证运算兼容
        if hasattr(X, 'values'):
            X_array = X.values
        else:
            X_array = X
            
        n_features = X_array.shape[1]
        current_indices = np.arange(n_features)

        if self.selection_method is None or self.selection_method == 'None':
            self.final_indices_ = current_indices
            return self

        # --- 1. 统计检验筛选 ---
        if self.selection_method == 'kbest':
            k = min(self.n_features_to_select, n_features)
            selector = SelectKBest(f_classif, k=k)
        elif self.selection_method == 'fdr':
            selector = SelectFdr(f_classif, alpha=self.alpha)
        elif self.selection_method == 'fwe':
            selector = SelectFwe(f_classif, alpha=self.alpha)
        else:
            raise ValueError(f"不支持的 selection_method: {self.selection_method}")

        try:
            X_selected = selector.fit_transform(X_array, y)
            selected_mask = selector.get_support()
            current_indices = current_indices[selected_mask]
        except Exception as e:
            # 如果筛选过于严格导致没有特征剩下，跳过并返回原样
            print(f"统计检验失败，原因: {e}。保留所有特征。")
            self.final_indices_ = current_indices
            return self

        # --- 2. 消除共线性冗余特征 ---
        if self.remove_redundant and X_selected.shape[1] > 1:
            f_scores, _ = f_classif(X_selected, y)
            sorted_idx = np.argsort(f_scores)[::-1]
            
            selected_subset = []
            for idx in sorted_idx:
                if len(selected_subset) == 0:
                    selected_subset.append(idx)
                else:
                    corr_vals = np.abs([np.corrcoef(X_selected[:, idx], X_selected[:, sel])[0, 1] for sel in selected_subset])
                    if np.max(corr_vals) <= self.corr_threshold:
                        selected_subset.append(idx)
            
            self.final_indices_ = current_indices[selected_subset]
        else:
            self.final_indices_ = current_indices

        return self

    def transform(self, X):
        if self.final_indices_ is None:
            raise ValueError("该转换器还未被 fit 过！")
        
        if hasattr(X, 'values'):
            return X.values[:, self.final_indices_]
        return X[:, self.final_indices_]
