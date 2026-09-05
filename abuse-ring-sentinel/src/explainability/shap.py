import os
import sys
import numpy as np
import pandas as pd

class TabularSHAPExplainer:
    def __init__(self, booster_model, feature_names):
        self.model = booster_model
        self.feature_names = feature_names
        
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
                
        # Native TreeSHAP via LightGBM C++ pred_contrib (Lundberg et al., 2020)
        # Returns [shap_feat_1, ..., shap_feat_n, base_value]
        contribs = self.model.predict(X, pred_contrib=True)
        if len(contribs.shape) > 1:
            s_vals = contribs[0][:len(self.feature_names)]
        else:
            s_vals = contribs[:len(self.feature_names)]
            
        shap_dict = {}
        for feat, val in zip(self.feature_names, s_vals):
            if val > 0.0:
                shap_dict[feat] = float(val)
                
        return shap_dict
