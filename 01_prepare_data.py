"""Step 1: inspect the Si GAP-18 dataset and write clean train/test files for MACE.

What it does
  * auto-detects the energy / force / virial-or-stress keys (GAP-18 uses
    dft_energy, dft_force, dft_virial; other copies may differ)
  * copies them to REF_energy, REF_forces, REF_stress (ASE sign convention,
    eV/A^3) so the DFT labels can never be confused with model predictions
  * pulls out the isolated-atom energy (E0) used as the energy reference
  * makes a test split stratified by config_type (every structure type is
    represented in the test set)
  * prints a summary of the data you are about to train on

Usage
  python 01_prepare_data.py path/to/gap18_training.xyz --out data
  (override detection with --energy-key / --forces-key / --virial-key / --stress-key)
"""
import argparse
import collections
import json
import os

import numpy as np
from ase.io import read, write

E_KEYS = ["dft_energy", "DFT_energy", "REF_energy", "energy", "free_energy"]
F_KEYS = ["dft_force", "dft_forces", "DFT_force", "REF_forces", "forces",
          "atomic_forces"]
V_KEYS = ["dft_virial", "DFT_virial", "REF_virial", "virial"]
S_KEYS = ["dft_stress", "REF_stress", "stress", "cauchy_stress"]
ISOLATED = {"isolated_atom", "isolatedatom", "isolated", "single_atom"}


def get_value(atoms, keys, per_atom=False):
    """Look for a label in info/arrays, then in an attached calculator."""
    for k in keys:
        store = atoms.arrays if per_atom else atoms.info
        if k in store:
            return k, np.array(store[k], dtype=float)
    if atoms.calc is not None:  # ASE moves energy/forces/stress here
        res = atoms.calc.results
        for k in keys:
            if k in res:
                return "calc:" + k, np.array(res[k], dtype=float)
    return None, None


def to_3x3(x):
    x = np.asarray(x, dtype=float).ravel()
    if x.size == 9:
        return x.reshape(3, 3)
    if x.size == 6:  # Voigt xx yy zz yz xz xy
        xx, yy, zz, yz, xz, xy = x
        return np.array([[xx, xy, xz], [xy, yy, yz], [xz, yz, zz]])
    raise ValueError("unexpected stress/virial shape %s" % (x.shape,))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("xyz")
    p.add_argument("--out", default="data")
    p.add_argument("--test-frac", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--energy-key")
    p.add_argument("--forces-key")
    p.add_argument("--virial-key")
    p.add_argument("--stress-key")
    a = p.parse_args()

    e_keys = [a.energy_key] if a.energy_key else E_KEYS
    f_keys = [a.forces_key] if a.forces_key else F_KEYS
    v_keys = [a.virial_key] if a.virial_key else V_KEYS
    s_keys = [a.stress_key] if a.stress_key else S_KEYS

    frames = read(a.xyz, index=":")
    print("Read %d configurations from %s" % (len(frames), a.xyz))
    print("info keys in first frame  :", sorted(frames[0].info))
    print("array keys in first frame :", sorted(frames[0].arrays))

    clean, isolated, skipped = [], [], 0
    used = collections.Counter()
    for at in frames:
        ek, e = get_value(at, e_keys)
        fk, f = get_value(at, f_keys, per_atom=True)
        vk, v = get_value(at, v_keys)
        sk, s = get_value(at, s_keys)
        ctype = str(at.info.get("config_type", "unknown"))
        if e is None:
            skipped += 1
            continue
        used["energy:" + ek] += 1

        new = at.copy()
        new.calc = None
        for k in list(new.info):
            if k != "config_type":
                del new.info[k]
        for k in list(new.arrays):
            if k not in ("numbers", "positions"):
                del new.arrays[k]
        new.info["config_type"] = ctype
        new.info["REF_energy"] = float(e)

        if len(at) == 1 and (ctype.lower() in ISOLATED or not at.pbc.any()):
            isolated.append(new)
            continue

        if f is not None:
            new.arrays["REF_forces"] = f.reshape(len(at), 3)
            used["forces:" + fk] += 1
        if at.pbc.all():
            if v is not None:  # stress = -virial / V  (ASE convention)
                new.info["REF_stress"] = (-to_3x3(v) / at.get_volume()).ravel()
                used["stress-from-virial:" + vk] += 1
            elif s is not None:
                new.info["REF_stress"] = to_3x3(s).ravel()
                used["stress:" + sk] += 1
        clean.append(new)

    print("\nLabels used:", dict(used))
    if skipped:
        print("Skipped %d frames with no energy label" % skipped)

    # Energy reference (E0) from the isolated atom
    e0 = {}
    for at in isolated:
        e0[int(at.numbers[0])] = at.info["REF_energy"]
    if e0:
        print("Isolated-atom energies (E0):", e0)
    else:
        print("WARNING: no isolated atom found; train with --E0s average")

    # Stratified split by config_type
    rng = np.random.default_rng(a.seed)
    groups = collections.defaultdict(list)
    for at in clean:
        groups[at.info["config_type"]].append(at)
    train, test = [], []
    print("\n%-28s %6s %6s %6s %12s" % ("config_type", "n", "train", "test",
                                       "E/atom (eV)"))
    for ct in sorted(groups):
        g = groups[ct]
        idx = rng.permutation(len(g))
        n_test = int(round(a.test_frac * len(g))) if len(g) >= 5 else 0
        test += [g[i] for i in idx[:n_test]]
        train += [g[i] for i in idx[n_test:]]
        epa = np.mean([x.info["REF_energy"] / len(x) for x in g])
        print("%-28s %6d %6d %6d %12.3f" % (ct, len(g), len(g) - n_test,
                                           n_test, epa))

    fmax = np.concatenate([np.abs(x.arrays["REF_forces"]).ravel()
                           for x in clean if "REF_forces" in x.arrays])
    n_stress = sum("REF_stress" in x.info for x in clean)
    print("\nTotal: %d train, %d test" % (len(train), len(test)))
    print("Atoms per config: %d-%d" % (min(map(len, clean)),
                                       max(map(len, clean))))
    print("|force component|: mean %.2f, max %.1f eV/A"
          % (fmax.mean(), fmax.max()))
    print("Configs with stress: %d" % n_stress)
    if n_stress:
        smax = max(np.abs(x.info["REF_stress"]).max() for x in clean
                   if "REF_stress" in x.info)
        print("Max |stress| = %.3f eV/A^3 (%.0f GPa). If this looks wildly off,"
              " check the stress units of your copy." % (smax, smax * 160.2177))

    os.makedirs(a.out, exist_ok=True)
    # Isolated atoms go into the training file too; MACE recognises them
    # by config_type=IsolatedAtom. E0s are also saved explicitly.
    for at in isolated:
        at.info["config_type"] = "IsolatedAtom"
    write(os.path.join(a.out, "train.xyz"), isolated + train, format="extxyz")
    write(os.path.join(a.out, "test.xyz"), test, format="extxyz")
    with open(os.path.join(a.out, "E0s.json"), "w") as fh:
        json.dump({str(k): v for k, v in e0.items()}, fh, indent=1)
    print("\nWrote %s/train.xyz, %s/test.xyz, %s/E0s.json" % ((a.out,) * 3))


if __name__ == "__main__":
    main()
