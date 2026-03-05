"""Save demo output to clean UTF-8 file."""
import sys, io
sys.stdout = io.TextIOWrapper(open('results_final/demo_results.txt', 'wb'), encoding='utf-8')
sys.stderr = sys.stdout
from real_world_demo import run_demo
run_demo()
sys.stdout.close()
