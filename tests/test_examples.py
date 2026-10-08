import os
import subprocess
import sys
from pathlib import Path

EX = Path(__file__).resolve().parents[1] / "examples"


def test_examples_run(tmp_path):
    env = dict(os.environ, OMP_NUM_THREADS="1")
    out = tmp_path / "demo.npz"
    subprocess.run([sys.executable, str(EX / "make_demo_data.py"), "--out", str(out), "--n-groups", "10"],
                   check=True, capture_output=True, text=True, env=env, timeout=300)
    r = subprocess.run([sys.executable, str(EX / "quickstart.py"), str(out)], capture_output=True, text=True,
                       env=env, timeout=600)
    assert r.returncode == 0, r.stderr
    assert "protocol=default" in r.stdout and "protocol=paper_2fold" in r.stdout
    assert "mahalanobis_l2: Delta =" in r.stdout and "knn_mean_cosine: Delta =" in r.stdout
