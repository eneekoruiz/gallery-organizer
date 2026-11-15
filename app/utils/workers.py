import sys
from app.services.workers import run_background_task

def run_recluster():
    import subprocess
    subprocess.Popen([sys.executable, "recluster.py"])

def run_auto_merge():
    import subprocess
    subprocess.Popen([sys.executable, "auto_merge.py"])
    
def run_smart_clean(threshold=0.52):
    import deep_auto_classify_all
    run_background_task(deep_auto_classify_all.run_deep_classify, threshold=threshold)
