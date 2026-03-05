echo "# RankBridge" > README.md
echo "A Federated Learning Framework for Cross-Domain Clustering via Ordinal Feature Importance." >> README.md
echo "## Paper Results and Summary" >> README.md
echo "Please see \`RankBridge_Paper_Summary.md\` for the exhaustive methodology, testing datasets, and claims." >> README.md
echo "## Generated Graphs" >> README.md
echo "The high-resolution plots verifying RankBridge's performance are located in \`data/plots/\`" >> README.md

git init
git add .
git commit -m "Initial commit of RankBridge paper code, results, and advanced plots"
git branch -M main
git remote add origin https://github.com/BryanLim0214/FederatedLearningFinalProject.git
git push -u origin main --force
