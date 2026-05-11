import xgboost as xgb
import shap
import pandas as pd
import numpy as np
import sys
import json

# This script runs in a separate process to avoid OpenMP/LLVM conflicts with PyTorch
MODEL_PATH = "storage/surrogate_xgb.json"

def main():
    if len(sys.argv) < 2:
        print(json.dumps({"error": "No features provided"}))
        return

    try:
        feature_dict = json.loads(sys.argv[1])
        
        if not sys.path[0]:
            sys.path[0] = "."
            
        model = xgb.XGBClassifier()
        model.load_model(MODEL_PATH)
        explainer = shap.TreeExplainer(model)
        
        # Map incoming scores to the model's expected feature keys
        if "layer_2_cluster_score" in feature_dict:
            feature_dict["layer_2_cluster"] = feature_dict.pop("layer_2_cluster_score")
        if "layer_3_clipping_score" in feature_dict:
            feature_dict["layer_3_clipping"] = feature_dict.pop("layer_3_clipping_score")
        if "layer_4_trim_score" in feature_dict:
            feature_dict["layer_4_is_trimmed"] = feature_dict.pop("layer_4_trim_score")

        df = pd.DataFrame([feature_dict])
        feature_order = [
            "layer_1_fl_trust", 
            "layer_2_cluster", 
            "layer_3_clipping", 
            "layer_4_is_trimmed", 
            "raw_norm"
        ]
        df = df[feature_order]

        # ------------------------------------------------------------------
        # 5-Tier Risk Scoring System
        # ------------------------------------------------------------------
        def get_risk_label(score):
            if score >= 0.8: return "CRITICAL", "Remove immediately"
            if score >= 0.6: return "DANGEROUS", "Quarantine / Partial reject"
            if score >= 0.4: return "SUSPICIOUS", "Lower aggregation weight"
            if score >= 0.2: return "WATCHLIST", "Increased monitoring"
            return "TRUSTED", "Full participation"

        prob = model.predict_proba(df)[0][1]
        global_label, global_action = get_risk_label(prob)
        
        report = {
            "risk_score": float(prob),
            "risk_label": global_label,
            "recommended_action": global_action,
            "layer_breakdown": {}
        }
        
        # ------------------------------------------------------------------
        # Layer-by-Layer Detailed Reasoning Engine
        # ------------------------------------------------------------------
        shap_values = explainer.shap_values(df)[0]
        
        for i, feat in enumerate(feature_order):
            val = feature_dict[feat]
            shap_val = shap_values[i]
            
            # Map SHAP contribution to a 0-1 scale for labeling consistency
            # (Using a sigmoid-like mapping for SHAP values)
            normalized_contribution = 1 / (1 + np.exp(-shap_val))
            layer_label, _ = get_risk_label(normalized_contribution)
            
            # Custom descriptive logic per layer
            if feat == "layer_1_fl_trust":
                desc = f"Trust score: {val:.4f}. "
                if val < 0.1: desc += "Severe directional misalignment."
                elif val < 0.5: desc += "Significant drift detected."
                else: desc += "Strong alignment with root."
                
            elif feat == "layer_2_cluster":
                desc = f"Cluster Anomaly Score: {val:.4f}. "
                if val > 3.0: desc += "Severe group outlier."
                elif val > 1.5: desc += "Suspicious deviation from group."
                else: desc += "Aligned with benign majority."
                
            elif feat == "layer_3_clipping":
                desc = f"Clipping Score: {val:.4f}. "
                if val > 2.0: desc += "Severe oversized influence detected."
                elif val > 1.3: desc += "Potentially suspicious magnitude."
                else: desc += "Normal update magnitude."
                
            elif feat == "layer_4_is_trimmed":
                desc = f"Trim Z-Score: {val:.4f}. "
                if val > 3.0: desc += "Extreme statistical outlier."
                elif val > 2.0: desc += "Significant deviation from mean."
                else: desc += "Within standard distribution."
                
            else: # raw_norm
                desc = f"L2 Norm: {val:.4f}."

            report["layer_breakdown"][feat] = {
                "value": val,
                "contribution_label": layer_label,
                "explanation": desc,
                "shap_score": float(shap_val)
            }
            
        # Identify Primary detection factor
        abs_shap = np.abs(shap_values)
        primary_feat = feature_order[np.argmax(abs_shap)]
        report["primary_factor"] = primary_feat
        
        # ------------------------------------------------------------------
        # LLM Reasoning Narrative: Synthesize evidence for the AI Agent
        # ------------------------------------------------------------------
        narrative_parts = []
        critical_count = 0
        
        for feat, details in report["layer_breakdown"].items():
            if details["contribution_label"] in ["CRITICAL", "DANGEROUS", "SUSPICIOUS"]:
                narrative_parts.append(f"- {feat}: {details['explanation']} (Level: {details['contribution_label']})")
                if details["contribution_label"] == "CRITICAL": critical_count += 1
                
        if not narrative_parts:
            report["security_narrative"] = "No significant anomalies detected across defense layers. Client behavior aligns with honest participation."
            report["threat_profile"] = "BENIGN_CONSISTENT"
        else:
            report["security_narrative"] = "The following anomalies were detected:\n" + "\n".join(narrative_parts)
            report["threat_profile"] = f"MULTI_LAYER_ANOMALY ({critical_count} Critical)" if critical_count > 1 else "SINGLE_LAYER_OUTLIER"

        print(json.dumps(report))
        
    except Exception as e:
        import traceback
        print(json.dumps({"error": str(e), "trace": traceback.format_exc()}))

if __name__ == "__main__":
    main()
