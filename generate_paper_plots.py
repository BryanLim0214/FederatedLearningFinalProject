import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Professional academic styling
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("paper", font_scale=1.5)
plt.rcParams['font.family'] = 'serif'

RESULTS_DIR = 'data/plots'
Path(RESULTS_DIR).mkdir(parents=True, exist_ok=True)

COLORS = {
    'RankBridge': '#2ecc71', # Bright Green
    'LocalOnly': '#95a5a6',  # Gray
    'FedClust': '#3498db',   # Blue
    'FedAvg': '#e74c3c',     # Red
    'IFCA': '#9b59b6',       # Purple
    'RandomCluster': '#f1c40f' # Yellow
}

def load_results():
    with open('results_real/all_results.json', 'r') as f:
        return json.load(f)

def load_history(method, seed=42):
    path = f'results_real/{method.lower()}_seed_{seed}/{method.lower()}'
    if method == 'FedAvg': path += '_standard'
    path += '.json'
    
    # rankbridge uses custom name
    if method == 'RankBridge' or method == 'FedRankX':
        path = f'results_real/fedrankx_seed_{seed}/fedrankx_kendall_ward_k30_r30.json'
        
    try:
        with open(path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Warning: Could not load history for {method} seed {seed} at {path}")
        return None

def plot_main_results(all_results):
    methods = [m for m in all_results.keys() if m != 'FedRankX']
    methods = ['RankBridge'] + methods # Rename FedRankX to RankBridge
    
    f1_means, f1_errs = [], []
    auc_means, auc_errs = [], []
    
    for m in methods:
        data_key = 'FedRankX' if m == 'RankBridge' else m
        f1_means.append(all_results[data_key]['f1_mean'])
        f1_errs.append(all_results[data_key]['f1_std'])
        auc_means.append(all_results[data_key]['auc_mean'])
        auc_errs.append(all_results[data_key]['auc_std'])
        
    x = np.arange(len(methods))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 7))
    rects1 = ax.bar(x - width/2, f1_means, width, yerr=f1_errs, label='F1 Score', color='#2c3e50', capsize=5)
    rects2 = ax.bar(x + width/2, auc_means, width, yerr=auc_errs, label='AUC', color='#e67e22', capsize=5)

    ax.set_ylabel('Score', fontweight='bold')
    ax.set_title('Performance on Heterogeneous Multi-Domain Data', fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=45, ha='right')
    ax.legend(loc='upper right')
    ax.set_ylim(0, 1.05)

    # Add text labels on bars
    for rects in [rects1, rects2]:
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.3f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    plt.savefig(f'{RESULTS_DIR}/01_main_results.png', dpi=300)
    plt.close()

def plot_convergence():
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot RankBridge
    all_rb_f1 = []
    for seed in [42, 123, 99]:
        hist = load_history('RankBridge', seed)
        if hist and 'f1_history' in hist:
            all_rb_f1.append([val for rnd, val in hist['f1_history']])
    
    if all_rb_f1:
        rounds = np.arange(1, len(all_rb_f1[0])+1)
        mean_rb = np.mean(all_rb_f1, axis=0)
        std_rb = np.std(all_rb_f1, axis=0)
        ax.plot(rounds, mean_rb, label='RankBridge', color=COLORS['RankBridge'], linewidth=3)
        ax.fill_between(rounds, mean_rb - std_rb, mean_rb + std_rb, color=COLORS['RankBridge'], alpha=0.2)

    # Plot FedAvg
    all_fa_f1 = []
    for seed in [42, 123, 99]:
        hist = load_history('FedAvg', seed)
        if hist and 'f1_history' in hist:
            all_fa_f1.append([val for rnd, val in hist['f1_history']])
            
    if all_fa_f1:
        rounds = np.arange(1, len(all_fa_f1[0])+1)
        mean_fa = np.mean(all_fa_f1, axis=0)
        std_fa = np.std(all_fa_f1, axis=0)
        ax.plot(rounds, mean_fa, label='FedAvg', color=COLORS['FedAvg'], linewidth=3, linestyle='--')
        ax.fill_between(rounds, mean_fa - std_fa, mean_fa + std_fa, color=COLORS['FedAvg'], alpha=0.2)

    ax.set_xlabel('Communication Round', fontweight='bold')
    ax.set_ylabel('Global F1 Score', fontweight='bold')
    ax.set_title('Convergence Stability in Extreme Non-IID Settings', fontweight='bold')
    ax.legend()
    ax.set_ylim(0, 1.0)
    plt.tight_layout()
    plt.savefig(f'{RESULTS_DIR}/02_convergence.png', dpi=300)
    plt.close()

def plot_clustering_quality():
    fig, ax = plt.subplots(figsize=(10, 6))
    
    hist = load_history('RankBridge', seed=42) # NMI is deterministic based on data
    if hist and 'round_cluster_history' in hist:
        rounds = [item['round'] for item in hist['round_cluster_history']]
        nmi = [item['nmi'] for item in hist['round_cluster_history']]
        ari = [item['ari'] for item in hist['round_cluster_history']]
        
        ax.plot(rounds, nmi, label='NMI (Normalized Mutual Information)', marker='o', color='#2980b9', linewidth=2)
        ax.plot(rounds, ari, label='ARI (Adjusted Rand Index)', marker='s', color='#8e44ad', linewidth=2)
        
        ax.set_xlabel('Communication Round', fontweight='bold')
        ax.set_ylabel('Clustering Score', fontweight='bold')
        ax.set_title('RankBridge Latent Organizational Discovery Quality', fontweight='bold')
        ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)
        ax.legend()
        ax.set_ylim(-0.1, 1.1)
        plt.tight_layout()
        plt.savefig(f'{RESULTS_DIR}/03_clustering_quality.png', dpi=300)
    plt.close()

def plot_client_distributions():
    # Load final client metrics from LocalOnly vs RankBridge
    
    data = []
    
    # Load RankBridge client stats
    hist_rb = load_history('RankBridge', seed=42)
    if hist_rb and 'metrics_distributed' in hist_rb and 'f1' in hist_rb['metrics_distributed']:
        # The list of F1s is not standard FL_engine output, but we can approximate or use LocalOnly vs Avg
        pass # Skipping individual client violin plot as fl_engine aggregates results, we don't have per-client JSONs saved

if __name__ == '__main__':
    print("Generating RankBridge Paper Plots...")
    all_res = load_results()
    plot_main_results(all_res)
    plot_convergence()
    plot_clustering_quality()
    print(f"Plots saved to {RESULTS_DIR}/")
