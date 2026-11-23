"""
Resume Analyzer - Main Application Entrypoint
Supports running via `python main.py` or `python app.py`.
"""

import os
from app import app

if __name__ == "__main__":
    is_debug = os.environ.get("FLASK_ENV") == "development" or os.environ.get("FLASK_DEBUG") == "1"
    port = int(os.environ.get("PORT", 5000))
    print(f"[*] Starting Resume Analyzer on http://localhost:{port} (Debug: {is_debug})")
    app.run(host="0.0.0.0", port=port, debug=is_debug)
