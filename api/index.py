"""
Vercel Serverless Function Entrypoint
Routes incoming requests to the Flask application instance.
"""

import os
import sys

# Ensure root directory is in sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import app
