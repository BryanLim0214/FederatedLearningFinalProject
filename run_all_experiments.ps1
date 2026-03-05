echo "==================================================="
echo " Starting Full FedRankX Synthetic Experiment Suite"
echo "==================================================="

echo ""
echo "[1/4] Running FedRankX (3 seeds)..."
python -m src.experiments.run_fedrankx --n_rounds 30 --top_k 15

echo ""
echo "[2/4] Running all Baseline methods (3 seeds)..."
python -m src.experiments.run_baselines --n_rounds 30

echo ""
echo "[3/4] Running Ablation Studies..."
python -m src.experiments.ablation --ablation all

echo ""
echo "[4/4] Generating Paper Tables and Figures..."
python -m src.evaluate

echo ""
echo "==================================================="
echo " All Experiments Complete!"
echo "==================================================="
