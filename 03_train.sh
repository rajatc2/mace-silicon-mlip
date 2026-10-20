#!/bin/bash
# Step 3: train a MACE potential on the prepared Si data.
#
# Usage:   bash 03_train.sh            # "fast" preset (default)
#          PRESET=better bash 03_train.sh
#          DEVICE=cpu EPOCHS=2 bash 03_train.sh   # smoke test only
#
# On a SLURM GPU node, wrap this in your usual job script, e.g.
#   #SBATCH --gres=gpu:1  --time=03:00:00
#
# Watch the log (logs/*.txt). After 3-5 epochs you know the time per epoch;
# if the full run won't fit your budget, stop it and lower EPOCHS.

set -euo pipefail
cd "$(dirname "$0")"

PRESET=${PRESET:-fast}
DEVICE=${DEVICE:-cuda}
NAME=${NAME:-si_mace_${PRESET}}

if [ "$PRESET" = "fast" ]; then
    # Invariant-only messages (L=0): quickest to train, still a real MACE model
    CHANNELS=64; MAX_L=0; EPOCHS=${EPOCHS:-150}; SWA_START=${SWA_START:-110}
else
    # Equivariant (L=1) messages: more accurate, roughly 2-3x slower
    CHANNELS=128; MAX_L=1; EPOCHS=${EPOCHS:-200}; SWA_START=${SWA_START:-150}
fi

# Isolated-atom energy from 01_prepare_data.py, e.g. {14: -158.54}
E0S=$(python -c "import json;d=json.load(open('data/E0s.json'));print('{'+','.join(f'{k}:{v}' for k,v in d.items())+'}' if d else 'average')")
echo "Using E0s = $E0S"

mace_run_train \
    --name="$NAME" \
    --seed=123 \
    --device="$DEVICE" \
    --default_dtype=float32 \
    --train_file=data/train.xyz \
    --valid_fraction=0.1 \
    --test_file=data/test.xyz \
    --energy_key=REF_energy \
    --forces_key=REF_forces \
    --stress_key=REF_stress \
    --E0s="$E0S" \
    --model=MACE \
    --num_channels=$CHANNELS \
    --max_L=$MAX_L \
    --correlation=3 \
    --r_max=5.0 \
    --loss=stress \
    --energy_weight=1.0 \
    --forces_weight=10.0 \
    --stress_weight=1.0 \
    --batch_size=5 \
    --valid_batch_size=10 \
    --lr=0.01 \
    --max_num_epochs=$EPOCHS \
    --swa --start_swa=$SWA_START \
    --swa_energy_weight=1000 --swa_forces_weight=100 \
    --ema --ema_decay=0.99 \
    --amsgrad \
    --scheduler_patience=10 \
    --patience=40 \
    --eval_interval=1 \
    --error_table=PerAtomMAE \
    --model_dir=models \
    --checkpoints_dir=checkpoints \
    --results_dir=results \
    --log_dir=logs \
    --save_cpu

# Notes
# * --swa switches to a high energy weight for the last epochs ("stage two");
#   this usually improves energies a lot at little cost to forces.
# * Training in float32 is faster; the MD scripts load the model in float64.
# * Final models: models/${NAME}.model (use this one) and *_stagetwo.model.
echo "Done. Model files:"; ls -1 models/ | grep "$NAME" || true
