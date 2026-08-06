#!/bin/bash
# =============================================================================
# Sequential Runner — 11 experiments, ONE GPU, ONE at a time
# =============================================================================
# Usage: tmux new -s train_all && bash scripts/run_all.sh
# =============================================================================
set -e
cd /root/autodl-tmp/mri_deep
git pull

echo "===== Runner | $(date) ====="
echo -n "Baseline: "
python -c 'from training.config import check_exist; print(check_exist("/root/autodl-tmp/ResUNet_model"))'

experiments=(
    "lb=0.1|train_enhanced.py --lambda_b 0.1|/root/autodl-tmp/ResUNet_Enhanced_lb0.1_model"
    "lb=0.3|train_enhanced.py --lambda_b 0.3|/root/autodl-tmp/ResUNet_Enhanced_lb0.3_model"
    "lb=0.5|train_enhanced.py --lambda_b 0.5|/root/autodl-tmp/ResUNet_Enhanced_lb0.5_model"
    "Edge concat|train_v2_edge.py --fusion concat --edge_type sobel|/root/autodl-tmp/ResUNet_Edge_concat_sobel_model"
    "Edge add|train_v2_edge.py --fusion add --edge_type sobel|/root/autodl-tmp/ResUNet_Edge_add_sobel_model"
    "Edge laplacian|train_v2_edge.py --fusion concat --edge_type laplacian|/root/autodl-tmp/ResUNet_Edge_concat_laplacian_model"
    "Edge random|train_v2_edge.py --fusion concat --edge_type random|/root/autodl-tmp/ResUNet_Edge_concat_random_model"
    "FG Sampling|train_fg_sampling.py|/root/autodl-tmp/ResUNet_FG_Sampling_model"
    "CC Dice|train_cc_dice.py|/root/autodl-tmp/ResUNet_CCDice_model"
    "PM Dice|train_loss_ablation.py --mode pm --pm_gamma 2.0|/root/autodl-tmp/ResUNet_PMDice_model"
)

total=${#experiments[@]}
run=0; skip=0; fail=0

for exp in "${experiments[@]}"; do
    IFS='|' read -r NAME SCRIPT_ARGS CKPT_DIR <<< "$exp"
    NAME=$(echo "$NAME" | xargs)
    SCRIPT_ARGS=$(echo "$SCRIPT_ARGS" | xargs)
    CKPT_DIR=$(echo "$CKPT_DIR" | xargs)

    echo ""
    echo "===== [$((run+skip+1))/$total] $NAME ====="
    echo "  $(date '+%H:%M:%S')"

    if ls "$CKPT_DIR"/best_model_*.pth 1>/dev/null 2>&1; then
        echo "  [SKIP] $(ls "$CKPT_DIR"/best_model_*.pth | tail -1)"
        skip=$((skip+1))
        continue
    fi

    echo "  [RUN]"
    S=$(date +%s)
    if python scripts/$SCRIPT_ARGS; then
        E=$(date +%s)
        echo "  [DONE] $(( (E-S)/60 )) min"
        run=$((run+1))
    else
        echo "  [FAIL]"
        fail=$((fail+1))
    fi
done

echo ""
echo "===== DONE | Ran:$run Skip:$skip Fail:$fail | $(date) ====="
