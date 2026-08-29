import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parents[1]/"src"))
import numpy as np
from phishintention.metrics import compute
def test_metrics_perfect():
 y=np.array([[1,0,0,0],[1,0,0,1]]);m=compute(y,y);assert m["micro_f1"]==1.0 and m["accuracy_by_complexity"]["2"]==1.0
