import json
import os
import subprocess
import sys

import numpy as np
import pytest

from crossfit_ood.cli import load_npz, main


@pytest.fixture()
def demo(tmp_path):
    p = str(tmp_path / "demo.npz")
    assert main(["demo", "--out", p, "--n-groups", "12"]) == 0
    return p


def test_demo_writes_documented_keys(demo):
    z = np.load(demo)
    for k in ("features_train", "groups_train", "features_id_eval", "groups_id_eval", "features_ood",
              "labels_train", "groups_ood"):
        assert k in z.files


def test_run_prints_summary_and_writes_json(demo, tmp_path, capsys):
    out = str(tmp_path / "rep.json")
    assert main(["run", demo, "--protocol", "paper_2fold", "--scorers", "mahalanobis", "--uncertainty", "bootstrap",
                 "--n-bootstrap", "50", "--out", out]) == 0
    txt = capsys.readouterr().out
    assert "protocol=paper_2fold" in txt and "K=2" in txt and "groups: train=12" in txt
    rep = json.load(open(out))
    assert "mahalanobis" in json.dumps(rep)


def test_run_default_protocol_jackknife(demo, capsys):
    assert main(["run", demo, "--protocol", "default", "--scorers", "knn_kth", "--n-splits", "3",
                 "--n-jackknife-blocks", "4"]) == 0
    txt = capsys.readouterr().out
    assert "protocol=default" in txt and "K=3" in txt and "uncertainty=jackknife" in txt


def test_key_renaming_and_missing_key(tmp_path, demo):
    z = dict(np.load(demo))
    z["X_tr"] = z.pop("features_train")
    p = str(tmp_path / "renamed.npz")
    np.savez(p, **z)
    with pytest.raises(SystemExit):
        load_npz(p)
    d = load_npz(p, {"features_train": "X_tr"})
    assert d["features_train"].shape[0] == len(d["groups_train"])


def test_python_dash_m(demo):
    env = dict(os.environ, OMP_NUM_THREADS="1")
    r = subprocess.run([sys.executable, "-m", "crossfit_ood", "run", demo, "--protocol", "paper_2fold", "--scorers", "vim",
                        "--uncertainty", "bootstrap", "--n-bootstrap", "20"],
                       capture_output=True, text=True, env=env, timeout=300)
    assert r.returncode == 0, r.stderr
    assert "protocol=paper_2fold" in r.stdout
