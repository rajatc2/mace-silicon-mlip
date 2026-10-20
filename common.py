"""Shared helpers: loading a MACE model (trained file or foundation model)."""
import torch


def default_device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_calculator(spec, device=None, dtype="float32"):
    """Return an ASE calculator.

    spec:
      "mp:small" / "mp:medium" / "mp:large"  -> MACE-MP-0 foundation model
      path/to/file.model                      -> your own trained MACE model
    float64 is used by default because it gives better energy conservation in MD.
    """
    device = device or default_device()
    if spec.startswith("mp:"):
        from mace.calculators import mace_mp
        return mace_mp(model=spec.split(":", 1)[1], device=device,
                       default_dtype=dtype)
    from mace.calculators import MACECalculator
    return MACECalculator(model_paths=spec, device=device, default_dtype=dtype)


def label(spec):
    """Short name for plots and tables."""
    if spec.startswith("mp:"):
        return "MACE-MP-0 (" + spec.split(":", 1)[1] + ", zero-shot)"
    return spec.split("/")[-1].replace(".model", "")
