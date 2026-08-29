import os
import sys
import numpy as np
import pandas as pd
import shap

class TabularSHAPExplainer:
    def __init__(self, booster_model, feature_names):
        self.model = booster_model
        self.feature_names = feature_names
        # Load TreeExplainer
        self.explainer = shap.TreeExplainer(self.model)
        
    def explain(self, single_row_df):
        X = pd.DataFrame(index=single_row_df.index)
        for col in self.feature_names:
            if col in single_row_df.columns:
                X[col] = single_row_df[col]
            else:
                X[col] = np.nan
                
        # Cast category fields
        for col in X.columns:
            if X[col].dtype.name in ('object', 'category') or col in ('ProductCD', 'addr2', 'P_emaildomain'):
                X[col] = X[col].astype('category')
                
        shap_values = self.explainer.shap_values(X)
        if isinstance(shap_values, list):
            s_vals = shap_values[1][0] if len(shap_values) > 1 else shap_values[0][0]
        else:
            s_vals = shap_values[0] if len(shap_values.shape) > 1 else shap_values
            
        shap_dict = {}
        for feat, val in zip(self.feature_names, s_vals):
            if val > 0.0:
                shap_dict[feat] = float(val)
                
        return shap_dict
