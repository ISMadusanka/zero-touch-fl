import xgboost as xgb
import shap
import pandas as pd
import numpy as np
import sys
import json
import os

# This script runs in a separate process to avoid OpenMP/LLVM conflicts with PyTorch
MODEL_PATH = "storage/surrogate_xgb.json"

def main():
    if len(sys.argv) < 2:
        print(json.dumps({"error": "No features provided"}))
        return

    try:
        feature_dict = json.loads(sys.argv[1])
        
        if not os.path.exists(MODEL_PATH):
             print(json.dumps({"error": f"Model file not found at {MODEL_PATH}"}))
             return

        model = xgb.XGBClassifier()
        model.load_model(MODEL_PATH)
        explainer = shap.TreeExplainer(model)
        
        # Ensure all required features exist for the XGBoost model
        # The model was trained with: layer_1_fl_trust, layer_2_cluster, layer_3_clipping, layer_4_is_trimmed, raw_norm
        
        df = pd.DataFrame([feature_dict])
        feature_order = [
            "layer_1_fl_trust", 
            "layer_2_cluster", 
            "layer_3_clipping", 
            "layer_4_is_trimmed", 
            "raw_norm"
        ]
        
        # Ensure all columns exist
        for col in feature_order:
            if col not in df.columns:
                df[col] = 0.0 # Default fallback
                
        df = df[feature_order]

        def get_risk_label(score):
            if score >= 0.8: return "CRITICAL", "Remove immediately"
            if score >= 0.6: return "DANGEROUS", "Quarantine / Partial reject"
            if score >= 0.4: return "SUSPICIOUS", "Lower aggregation weight"
            if score >= 0.2: return "WATCHLIST", "Increased monitoring"
            return "TRUSTED", "Full participation"

        prob = float(model.predict_proba(df)[0][1])
        global_label, global_action = get_risk_label(prob)
        
        report = {
            "risk_score": prob,
            "risk_label": global_label,
            "recommended_action": global_action,
            "layer_breakdown": {}
        }
        
        shap_values = explainer.shap_values(df)[0]
        
        for i, feat in enumerate(feature_order):
            val = float(df[feat].iloc[0])
            shap_val = float(shap_values[i])
            
            # Map SHAP contribution to a readable label
            normalized_contribution = 1 / (1 + np.exp(-shap_val))
            layer_label, _ = get_risk_label(normalized_contribution)
            
            # Descriptive logic
            if feat == "layer_1_fl_trust":
                desc = f"Trust score: {val:.4f}. "
            elif feat == "layer_2_cluster":
                desc = f"Cluster Anomaly Score: {val:.4f}. "
            elif feat == "layer_3_clipping":
                desc = f"Clipping Score: {val:.4f}. "
            elif feat == "layer_4_is_trimmed":
                desc = f"Trim Z-Score: {val:.4f}. "
            else:
                desc = f"Raw Norm: {val:.4f}."

            report["layer_breakdown"][feat] = {
                "value": val,
                "contribution_label": layer_label,
                "explanation": desc,
                "shap_score": shap_val
            }
            
        abs_shap = np.abs(shap_values)
        primary_feat = feature_order[np.argmax(abs_shap)]
        report["primary_factor"] = primary_feat
        
        print(json.dumps(report))
        
    except Exception as e:
        import traceback
        print(json.dumps({"error": str(e), "trace": traceback.format_exc()}))

if __name__ == "__main__":
    main()
