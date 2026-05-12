import json
import logging
import subprocess
import sys
import os

logger = logging.getLogger(__name__)

class ExplainabilityEngine:
    """Surrogate reasoning engine that runs in a separate process."""

    def __init__(self, script_path="scripts/explain_update.py"):
        self.script_path = script_path
        # Use the same python executable that is running the current process
        # or the venv one if we can detect it.
        self.python_exe = sys.executable

    def explain(self, feature_dict: dict) -> dict:
        """Calls the explain_update.py script in a separate process."""
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
