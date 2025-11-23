# run_weeklyrecap.py
import os
import sys
from pipeline import run_pipeline  # UPDATED: was from weeklyrecap import run_pipeline

if __name__ == "__main__":
    # Usage: python run_weeklyrecap.py [optional path to config.yaml]
    base = os.path.dirname(os.path.abspath(__file__))
    cfg_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(base, "config.yaml")
    run_pipeline(cfg_path)
