"""Step 2 and 4: test-set errors and parity plots for one or more models.

Run it first on the foundation model alone (zero-shot baseline), then again
with your trained model to compare the two.

Usage
  python 02_evaluate.py data/test.xyz --models mp:small
  python 02_evaluate.py data/test.xyz --models mp:small models/si_mace.model

Reported for each model
  * energy MAE / RMSE in meV/atom
  * energy MAE after removing a constant per-atom offset. A foundation model
    trained on PBE data with different energy references can be off by a
    constant amount and still have the right energy *differences*; this
    column separates the two effects.
  * force MAE / RMSE in meV/A
  * force MAE per config_type (shows where the model is weakest)
"""
import argparse
import collections
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from ase.io import read

from common import label, load_calculator


def evaluate(spec, frames):
    calc = load_calculator(spec)
    e_ref, e_pred, f_ref, f_pred, ctypes, f_ct = [], [], [], [], [], []
    for at in frames:
        n = len(at)
        x = at.copy()
        x.calc = calc
        e_ref.append(at.info["REF_energy"] / n)
        e_pred.append(x.get_potential_energy() / n)
        ctypes.append(at.info.get("config_type", "unknown"))
        if "REF_forces" in at.arrays:
            fr = at.arrays["REF_forces"]
            fp = x.get_forces()
            f_ref.append(fr.ravel())
            f_pred.append(fp.ravel())
            f_ct.append((ctypes[-1], np.abs(fp - fr).ravel()))
    e_ref, e_pred = np.array(e_ref), np.array(e_pred)
    f_ref, f_pred = np.concatenate(f_ref), np.concatenate(f_pred)
    de = e_pred - e_ref
    df = f_pred - f_ref
    by_ct = collections.defaultdict(list)
    for ct, err in f_ct:
        by_ct[ct].append(err)
    res = {
        "E_MAE_meV_atom": 1e3 * np.abs(de).mean(),
        "E_RMSE_meV_atom": 1e3 * np.sqrt((de ** 2).mean()),
        "E_offset_meV_atom": 1e3 * de.mean(),
        "E_MAE_offset_removed_meV_atom": 1e3 * np.abs(de - de.mean()).mean(),
        "F_MAE_meV_A": 1e3 * np.abs(df).mean(),
        "F_RMSE_meV_A": 1e3 * np.sqrt((df ** 2).mean()),
        "F_MAE_by_config_type_meV_A": {
            ct: 1e3 * float(np.concatenate(v).mean()) for ct, v in by_ct.items()
        },
    }
    return res, (e_ref, e_pred, f_ref, f_pred)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("test_xyz")
    p.add_argument("--models", nargs="+", required=True)
    p.add_argument("--out", default="results")
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)

    frames = read(a.test_xyz, index=":")
    print("Evaluating on %d test configurations" % len(frames))

    all_res, raw = {}, {}
    for spec in a.models:
        print("\n==> %s" % label(spec))
        res, data = evaluate(spec, frames)
        all_res[label(spec)] = res
        raw[label(spec)] = data
        for k, v in res.items():
            if not isinstance(v, dict):
                print("  %-32s %10.2f" % (k, v))
        print("  Force MAE by config_type (meV/A), worst first:")
        for ct, v in sorted(res["F_MAE_by_config_type_meV_A"].items(),
                            key=lambda kv: -kv[1]):
            print("    %-26s %8.1f" % (ct, v))

    with open(os.path.join(a.out, "test_errors.json"), "w") as fh:
        json.dump(all_res, fh, indent=1, default=float)

    # Parity plots: energy and force, one row per model
    n = len(raw)
    fig, axes = plt.subplots(n, 2, figsize=(9, 4.2 * n), squeeze=False)
    for row, (name, (er, ep, fr, fp)) in enumerate(raw.items()):
        r = all_res[name]
        ax = axes[row, 0]
        ax.scatter(er, ep, s=6, alpha=0.6)
        lim = [min(er.min(), ep.min()), max(er.max(), ep.max())]
        ax.plot(lim, lim, "k--", lw=0.8)
        ax.set_xlabel("DFT energy (eV/atom)")
        ax.set_ylabel("Model energy (eV/atom)")
        ax.set_title("%s\nE MAE %.1f meV/atom (offset-removed %.1f)"
                     % (name, r["E_MAE_meV_atom"],
                        r["E_MAE_offset_removed_meV_atom"]), fontsize=9)
        ax = axes[row, 1]
        ax.scatter(fr, fp, s=2, alpha=0.3)
        lim = [min(fr.min(), fp.min()), max(fr.max(), fp.max())]
        ax.plot(lim, lim, "k--", lw=0.8)
        ax.set_xlabel("DFT force component (eV/A)")
        ax.set_ylabel("Model force component (eV/A)")
        ax.set_title("F MAE %.0f meV/A" % r["F_MAE_meV_A"], fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(a.out, "parity.png"), dpi=150)
    print("\nSaved %s/test_errors.json and %s/parity.png" % (a.out, a.out))


if __name__ == "__main__":
    main()
