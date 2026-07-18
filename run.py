"""
run.py
=======
Convenience launcher so you can start the whole app from the PROJECT ROOT
(instead of cd-ing into backend/ every time).

Usage:
    python run.py
"""
import os
import sys
import uvicorn

BACKEND_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "backend")
sys.path.append(BACKEND_DIR)
os.chdir(BACKEND_DIR)

if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
