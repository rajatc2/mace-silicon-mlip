"""Step 5: physical validation with MD (ASE), for one or more models.

Tests
  1. Equation of state of diamond Si -> lattice constant a0, bulk modulus B0
  2. NVE run of the crystal (216 atoms, ~300 K) -> energy drift
     (a model with rough or non-conservative forces drifts or explodes here)
  3. NVT crystal at 300 K -> RDF
  4. Liquid Si at 1800 K (melted at 3000 K, liquid density) -> RDF,
     coordination number, self-diffusion coefficient
  5. Reference liquid RDF computed from the DFT liquid snapshots in your
     dataset (config_type containing "liq"), overlaid for direct comparison

Usage
  python 04_md_validate.py --models models/si_mace_fast.model mp:small \
                           --ref-xyz data/train.xyz
  python 04_md_validate.py --models mp:small --quick   # 1-minute smoke test

Run on a GPU. On CPU the full run takes hours; use --quick there.
"""
import argparse
import json
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from ase import units
from ase.build import bulk
from ase.eos import EquationOfState
from ase.io import read
from ase.md.langevin import Langevin
from ase.md.velocitydistribution import (MaxwellBoltzmannDistribution,
                                         Stationary, ZeroRotation)
from ase.md.verlet import VelocityVerlet

from common import label, load_calculator

M_SI = 28.0855          # amu
RHO_LIQ = 2.57          # g/cm^3, liquid Si near the melting point
A_EXP, B_EXP = 5.431, 98.0  # experimental a0 (A) and B0 (GPa), room T


# ---------------------------------------------------------------- analysis
def perp_widths(cell):
    v = abs(np.linalg.det(cell))
    return [v / np.linalg.norm(np.cross(cell[i], cell[j]))
            for i, j in ((1, 2), (2, 0), (0, 1))]


def rdf(frames, rmax, nbins=120):
    """Average g(r) over frames (minimum-image distances)."""
    edges = np.linspace(0, rmax, nbins + 1)
    hist = np.zeros(nbins)
    norm = 0.0
    for at in frames:
        d = at.get_all_distances(mic=True)
        n = len(at)
        d = d[np.triu_indices(n, k=1)]
        h, _ = np.histogram(d, bins=edges)
        hist += 2 * h                     # count each pair for both atoms
        norm += n * n / at.get_volume()   # N * density
    shell = 4 / 3 * np.pi * (edges[1:] ** 3 - edges[:-1] ** 3)
    r = 0.5 * (edges[1:] + edges[:-1])
    return r, hist / (norm * shell)


def coordination(r, g, rho, rcut):
    m = r < rcut
    dr = r[1] - r[0]
    return float(np.sum(4 * np.pi * r[m] ** 2 * g[m] * rho * dr))


# ---------------------------------------------------------------- tests
def eos_test(calc):
    a_guess = 5.43
    vols, ens = [], []
    for s in np.linspace(0.94, 1.06, 13):
        at = bulk("Si", "diamond", a=a_guess * s, cubic=True)
        at.calc = calc
        vols.append(at.get_volume())
        ens.append(at.get_potential_energy())
    eos = EquationOfState(vols, ens, eos="birchmurnaghan")
    v0, e0, b = eos.fit()
    return {"a0_A": v0 ** (1 / 3), "B0_GPa": b / units.GPa,
            "E0_eV_atom": e0 / 8}, (np.array(vols) / 8, np.array(ens) / 8)


def nve_test(calc, a0, steps, seed):
    at = bulk("Si", "diamond", a=a0, cubic=True).repeat(3)
    at.calc = calc
    MaxwellBoltzmannDistribution(at, temperature_K=600,
                                 rng=np.random.default_rng(seed))
    Stationary(at)
    ZeroRotation(at)
    dyn = VelocityVerlet(at, timestep=1.0 * units.fs)
    etot, temp = [], []

    def rec():
        etot.append(at.get_total_energy() / len(at))
        temp.append(at.get_temperature())
    dyn.attach(rec, interval=1)
    dyn.run(steps)
    t_ps = np.arange(len(etot)) * 1e-3
    slope = np.polyfit(t_ps, etot, 1)[0]
    return {"drift_meV_atom_ps": 1e3 * slope,
            "Etot_std_meV_atom": 1e3 * float(np.std(etot)),
            "mean_T_K": float(np.mean(temp[len(temp) // 4:])),
            "max_force_end_eV_A": float(np.abs(at.get_forces()).max())
            }, (t_ps, np.array(etot))


def nvt(at, calc, T, steps, sample_every=0, seed=0):
    at.calc = calc
    dyn = Langevin(at, timestep=1.0 * units.fs, temperature_K=T,
                   friction=0.02 / units.fs, rng=np.random.default_rng(seed))
    frames, pos = [], []
    if sample_every:
        def grab():
            frames.append(at.copy())
            pos.append(at.get_positions().copy())  # ASE MD does not wrap
        dyn.attach(grab, interval=sample_every)
    dyn.run(steps)
    return frames, pos


def crystal_rdf_test(calc, a0, steps, seed):
    at = bulk("Si", "diamond", a=a0, cubic=True).repeat(3)
    MaxwellBoltzmannDistribution(at, temperature_K=300,
                                 rng=np.random.default_rng(seed))
    nvt(at, calc, 300, steps // 3, seed=seed)                 # equilibrate
    frames, _ = nvt(at, calc, 300, steps, sample_every=max(1, steps // 50),
                    seed=seed + 1)
    return rdf(frames, rmax=min(6.0, 0.5 * min(perp_widths(at.cell)) - 0.05))


def liquid_test(calc, melt, equil, prod, seed):
    at = bulk("Si", "diamond", a=5.43, cubic=True).repeat(3)
    vol = len(at) * M_SI / units.kg * 1e3 / RHO_LIQ * 1e24  # g / (g/cm3) -> A^3
    at.set_cell(np.eye(3) * vol ** (1 / 3), scale_atoms=True)
    MaxwellBoltzmannDistribution(at, temperature_K=3000,
                                 rng=np.random.default_rng(seed))
    nvt(at, calc, 3000, melt, seed=seed)
    nvt(at, calc, 1800, equil, seed=seed + 1)
    every = max(1, prod // 250)
    frames, pos = nvt(at, calc, 1800, prod, sample_every=every, seed=seed + 2)
    rmax = min(6.0, 0.5 * min(perp_widths(at.cell)) - 0.05)
    r, g = rdf(frames[len(frames) // 5:], rmax)
    rho = len(at) / at.get_volume()
    pos = np.array(pos)
    disp = pos - pos[0]
    disp -= disp.mean(axis=1, keepdims=True)          # remove COM drift
    msd = (disp ** 2).sum(-1).mean(-1)                # A^2
    t_fs = np.arange(len(msd)) * every
    half = len(msd) // 2
    slope = np.polyfit(t_fs[half:], msd[half:], 1)[0]  # A^2/fs
    D = slope / 6 * 0.1                                # 1 A^2/fs = 0.1 cm^2/s
    return {"rdf_first_peak_A": float(r[np.argmax(g)]),
            "CN_r_lt_3.0A": coordination(r, g, rho, 3.0),
            "D_cm2_s": float(D),
            "max_force_end_eV_A": float(np.abs(at.get_forces()).max())
            }, (r, g), (t_fs * 1e-3, msd)


def reference_liquid_rdf(xyz):
    frames = [f for f in read(xyz, index=":")
              if "liq" in str(f.info.get("config_type", "")).lower()
              and f.pbc.all()]
    if not frames:
        return None
    rmax = min(6.0, min(0.5 * min(perp_widths(f.cell)) for f in frames) - 0.05)
    r, g = rdf(frames, rmax, nbins=int(120 * rmax / 6))
    rho = np.mean([len(f) / f.get_volume() for f in frames])
    return {"n_frames": len(frames),
            "rdf_first_peak_A": float(r[np.argmax(g)]),
            "CN_r_lt_3.0A": coordination(r, g, rho, 3.0)}, (r, g)


# ---------------------------------------------------------------- main
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--models", nargs="+", required=True)
    p.add_argument("--ref-xyz", help="dataset file to take DFT liquid RDF from")
    p.add_argument("--out", default="results")
    p.add_argument("--quick", action="store_true", help="tiny runs, smoke test")
    p.add_argument("--seed", type=int, default=1)
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)

    f = 50 if a.quick else 1
    n_nve, n_cry = 2000 // f, 3000 // f
    n_melt, n_eq, n_prod = 3000 // f, 2000 // f, 5000 // f

    results, curves = {}, {}
    ref = reference_liquid_rdf(a.ref_xyz) if a.ref_xyz else None
    if ref:
        results["DFT reference (dataset liquid snapshots)"] = ref[0]
        print("Reference liquid RDF from %d DFT snapshots" % ref[0]["n_frames"])

    for spec in a.models:
        name = label(spec)
        print("\n==> %s" % name)
        calc = load_calculator(spec)
        t0 = time.time()
        res = {}
        res["eos"], eosc = eos_test(calc)
        print("  EOS: a0 = %.4f A, B0 = %.1f GPa"
              % (res["eos"]["a0_A"], res["eos"]["B0_GPa"]))
        res["nve"], nvec = nve_test(calc, res["eos"]["a0_A"], n_nve, a.seed)
        print("  NVE: drift = %.3f meV/atom/ps, <T> = %.0f K"
              % (res["nve"]["drift_meV_atom_ps"], res["nve"]["mean_T_K"]))
        crc = crystal_rdf_test(calc, res["eos"]["a0_A"], n_cry, a.seed)
        res["liquid_1800K"], lic, msdc = liquid_test(calc, n_melt, n_eq,
                                                      n_prod, a.seed)
        print("  Liquid: first peak %.2f A, CN %.2f, D = %.2e cm^2/s"
              % (res["liquid_1800K"]["rdf_first_peak_A"],
                 res["liquid_1800K"]["CN_r_lt_3.0A"],
                 res["liquid_1800K"]["D_cm2_s"]))
        res["wall_time_s"] = time.time() - t0
        results[name] = res
        curves[name] = dict(eos=eosc, nve=nvec, crystal=crc, liquid=lic,
                            msd=msdc)

    results["notes"] = {
        "experiment": "a0 = %.3f A, B0 ~ %.0f GPa (room T). GGA DFT typically "
                      "gives a0 ~0.5-1%% larger and B0 ~10%% smaller; compare "
                      "the model first with its own DFT reference." % (A_EXP, B_EXP),
        "quick_mode": a.quick,
    }
    with open(os.path.join(a.out, "md_validation.json"), "w") as fh:
        json.dump(results, fh, indent=1, default=float)

    # --------------------------------------------------- plots
    fig, ax = plt.subplots(2, 2, figsize=(11, 8))
    for name, c in curves.items():
        v, e = c["eos"]
        ax[0, 0].plot(v, e - e.min(), "o-", ms=3, label=name)
        t, et = c["nve"]
        ax[0, 1].plot(t, 1e3 * (et - et[0]), lw=0.8, label=name)
        ax[1, 0].plot(*c["crystal"], label=name)
        ax[1, 1].plot(*c["liquid"], label=name)
    if ref:
        ax[1, 1].plot(*ref[1], "k--", lw=1.5, label="DFT snapshots (dataset)")
    ax[0, 0].set(xlabel="Volume (A^3/atom)", ylabel="E - Emin (eV/atom)",
                 title="Equation of state, diamond Si")
    ax[0, 1].set(xlabel="Time (ps)", ylabel="Etot - Etot(0) (meV/atom)",
                 title="NVE energy conservation, 216 atoms")
    ax[1, 0].set(xlabel="r (A)", ylabel="g(r)", title="Crystal, 300 K")
    ax[1, 1].set(xlabel="r (A)", ylabel="g(r)",
                 title="Liquid, 1800 K, %.2f g/cm^3" % RHO_LIQ)
    for x in ax.ravel():
        x.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(a.out, "md_validation.png"), dpi=150)

    fig, x = plt.subplots(figsize=(5, 3.5))
    for name, c in curves.items():
        x.plot(*c["msd"], label=name)
    x.set(xlabel="Time (ps)", ylabel="MSD (A^2)", title="Liquid Si, 1800 K")
    x.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(a.out, "liquid_msd.png"), dpi=150)
    print("\nSaved md_validation.json, md_validation.png, liquid_msd.png in",
          a.out)


if __name__ == "__main__":
    main()
