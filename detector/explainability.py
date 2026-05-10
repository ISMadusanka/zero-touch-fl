import json
import os
import logging
import subprocess
import sys

logger = logging.getLogger(__name__)

class ExplainabilityEngine:
    """Surrogate reasoning engine that runs in a separate process to avoid library conflicts."""

    def __init__(self, script_path="scripts/explain_update.py"):
        self.script_path = script_path
        self.python_exe = sys.executable

    # TODO [Step 2.2]: Implement incremental_train(new_features, labels) 
    # This would use model.fit(X, y, xgb_model=MODEL_PATH) for warm-start boosting.
    # For now, retraining is handled manually or skipped as per plan.

    def explain(self, feature_dict: dict) -> dict:
        """
        Calls the explain_update.py script in a separate process.
        """
        try:
            # Pass features as a JSON string
            cmd = [self.python_exe, self.script_path, json.dumps(feature_dict)]
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            
            # Parse the JSON output from the script
            return json.loads(result.stdout)
        except Exception as e:
            logger.error(f"ExplainabilityEngine: sub-process failed: {e}")
            if hasattr(e, 'stderr') and e.stderr:
                logger.error(f"Stderr: {e.stderr}")
            return {"error": f"Explainability sub-process failed: {str(e)}"}
