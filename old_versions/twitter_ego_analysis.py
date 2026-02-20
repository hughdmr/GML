#!/usr/bin/env python3
"""
twitter_ego_analysis.py

Convert the functionality of Untitled-1.ipynb into a single runnable Python script.
"""

import os
import json
from pathlib import Path
from math import pi
import argparse
import warnings

import numpy as np
import pandas as pd
import networkx as nx
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

import matplotlib
# Use non-interactive backend so plots are always saved in headless environments
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def safe_tight_layout():
    """Call plt.tight_layout() but suppress UserWarning by falling back to subplots_adjust."""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            plt.tight_layout()
    except UserWarning:
        try:
            plt.subplots_adjust(top=0.95, bottom=0.05, left=0.05, right=0.95, hspace=0.3, wspace=0.3)
        except Exception:
            pass


sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)


def find_ego_network_ids(twitter_path: str):
    path = Path(twitter_path)
    if not path.exists():
        print(f"Directory {twitter_path} not found!")
        return []
    ids = []
    for f in path.iterdir():
        if f.suffix == ".edges":
            ids.append(f.stem)
    ids.sort()
    print(f"Found {len(ids)} ego networks")
    return ids


def load_ego_network(node_id: str, twitter_path: str = "twitter_ego/twitter"):
    data = {}
    base = Path(twitter_path)
    edges_file = base / f"{node_id}.edges"
    feat_file = base / f"{node_id}.feat"
    egofeat_file = base / f"{node_id}.egofeat"
    circles_file = base / f"{node_id}.circles"

    if edges_file.exists():
        try:
            data['edges'] = pd.read_csv(edges_file, sep=r"\s+", header=None, names=["source", "target"], engine='python')
        except Exception:
            data['edges'] = pd.read_csv(edges_file, sep=" ", header=None, names=["source", "target"])
    else:
        data['edges'] = pd.DataFrame(columns=["source", "target"])

    if feat_file.exists():
        data['feat'] = pd.read_csv(feat_file, sep=r"\s+", header=None, engine='python')
    else:
        data['feat'] = None

    if egofeat_file.exists():
        data['egofeat'] = pd.read_csv(egofeat_file, sep=r"\s+", header=None, engine='python')
    else:
        data['egofeat'] = None

    circles = {}
    if circles_file.exists():
        with open(circles_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if not parts:
                    continue
                circles[parts[0]] = parts[1:] if len(parts) > 1 else []
    data['circles'] = circles
    return data


def build_ego_graph(node_id: str, network_data: dict):
    G = nx.DiGraph()
    edges = network_data.get('edges', pd.DataFrame(columns=["source", "target"]))
    if not edges.empty:
        # ensure numeric or string consistency
        G.add_edges_from(edges.values)
    G.add_node(node_id)
    neighbors = set()
    if not edges.empty:
        neighbors = set(edges["source"].tolist()) | set(edges["target"].tolist())
    for n in neighbors:
        G.add_edge(node_id, n)
    return G


def compute_basic_stats(G: nx.DiGraph):
    stats_out = {
        'num_nodes': G.number_of_nodes(),
        'num_edges': G.number_of_edges(),
        'density': nx.density(G),
        'is_connected': nx.is_weakly_connected(G),
    }
    try:
        stats_out['num_scc'] = nx.number_strongly_connected_components(G)
        stats_out['num_wcc'] = nx.number_weakly_connected_components(G)
    except Exception:
        stats_out['num_scc'] = 0
        stats_out['num_wcc'] = 0
    return stats_out


def compute_degree_stats(G: nx.DiGraph):
    in_degrees = [d for _, d in G.in_degree()]
    out_degrees = [d for _, d in G.out_degree()]
    if len(in_degrees) == 0:
        in_degrees = [0]
    if len(out_degrees) == 0:
        out_degrees = [0]
    stats_out = {
        'mean_in_degree': float(np.mean(in_degrees)),
        'median_in_degree': float(np.median(in_degrees)),
        'max_in_degree': int(np.max(in_degrees)),
        'std_in_degree': float(np.std(in_degrees)),
        'mean_out_degree': float(np.mean(out_degrees)),
        'median_out_degree': float(np.median(out_degrees)),
        'max_out_degree': int(np.max(out_degrees)),
        'std_out_degree': float(np.std(out_degrees)),
    }
    return stats_out, in_degrees, out_degrees


def compute_clustering_stats(G: nx.DiGraph):
    G_undirected = G.to_undirected()
    try:
        avg_clustering = nx.average_clustering(G_undirected) if G_undirected.number_of_nodes() > 0 else 0.0
    except Exception:
        avg_clustering = 0.0
    try:
        transitivity = nx.transitivity(G_undirected)
    except Exception:
        transitivity = 0.0
    clustering_coeffs = list(nx.clustering(G_undirected).values()) if G_undirected.number_of_nodes() > 0 else []
    stats_out = {
        'avg_clustering': float(avg_clustering),
        'transitivity': float(transitivity),
        'max_clustering': float(np.max(clustering_coeffs)) if clustering_coeffs else 0.0,
        'min_clustering': float(np.min(clustering_coeffs)) if clustering_coeffs else 0.0,
        'std_clustering': float(np.std(clustering_coeffs)) if clustering_coeffs else 0.0
    }
    return stats_out, clustering_coeffs


def safe_zscore(arr: np.ndarray):
    arr = np.asarray(arr, dtype=float)
    mean = np.nanmean(arr)
    std = np.nanstd(arr)
    if std == 0 or np.isnan(std):
        return np.zeros_like(arr)
    return (arr - mean) / std


def identify_outliers(df: pd.DataFrame, columns: list, threshold: float = 2.0):
    outliers = {}
    for col in columns:
        if col not in df.columns:
            continue
        col_vals = df[col].astype(float).values
        z_scores = np.abs(safe_zscore(col_vals))
        mask = z_scores > threshold
        if mask.any():
            nets = df.index[mask].tolist()
            vals = df.loc[nets, col].tolist()
            zs = z_scores[mask].tolist()
            outliers[col] = {'networks': nets, 'values': vals, 'z_scores': zs}
    return outliers


def cluster_networks(df: pd.DataFrame, features: list, k: int = 3):
    X = df[features].astype(float).values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    if X_scaled.shape[0] < 2:
        df['cluster'] = 0
        return df, None
    k = min(k, X_scaled.shape[0])
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_scaled)
    df['cluster'] = labels
    return df, kmeans


def plot_dashboard(comprehensive_df: pd.DataFrame, save_dir: Path = None):
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

    ax1 = fig.add_subplot(gs[0, 0])
    ax1.bar(range(len(comprehensive_df)), comprehensive_df['density'], color='steelblue', alpha=0.7, edgecolor='black')
    ax1.set_title('Network Density Comparison')

    ax2 = fig.add_subplot(gs[0, 1])
    ax2.bar(range(len(comprehensive_df)), comprehensive_df['avg_degree'], color='coral', alpha=0.7, edgecolor='black')
    ax2.set_title('Average Degree Comparison')

    ax3 = fig.add_subplot(gs[0, 2])
    ax3.bar(range(len(comprehensive_df)), comprehensive_df['avg_clustering'], color='forestgreen', alpha=0.7, edgecolor='black')
    ax3.set_title('Clustering Coefficient Comparison')

    ax4 = fig.add_subplot(gs[1, 0])
    ax4.bar(range(len(comprehensive_df)), comprehensive_df['num_scc'], color='purple', alpha=0.7, edgecolor='black')
    ax4.set_title('Strongly Connected Components')

    ax5 = fig.add_subplot(gs[1, 1])
    ax5.scatter(comprehensive_df['num_nodes'], comprehensive_df['num_edges'], s=100, alpha=0.6, c=range(len(comprehensive_df)), cmap='viridis')
    ax5.set_title('Network Size: Nodes vs Edges')

    ax6 = fig.add_subplot(gs[1, 2])
    ax6.scatter(comprehensive_df['avg_degree'], comprehensive_df['avg_clustering'], s=100, alpha=0.6, c=range(len(comprehensive_df)), cmap='viridis')
    ax6.set_title('Degree vs Clustering')

    metrics_for_boxplot = [
        ('density', 'Density', 'steelblue'),
        ('avg_degree', 'Average Degree', 'coral'),
        ('avg_clustering', 'Clustering Coefficient', 'forestgreen')
    ]

    for idx, (metric, title, color) in enumerate(metrics_for_boxplot):
        ax = fig.add_subplot(gs[2, idx])
        bp = ax.boxplot([comprehensive_df[metric].dropna().values], vert=True, patch_artist=True)
        for patch in bp['boxes']:
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        ax.set_xticklabels(['All Networks'])
        ax.set_title(f'{title} Distribution')
        y = comprehensive_df[metric].values
        x = np.random.normal(1, 0.04, size=len(y))
        ax.scatter(x, y, alpha=0.4, color='black', s=30)

    plt.suptitle('Comprehensive Network Metrics Dashboard', fontsize=16, fontweight='bold', y=0.995)
    safe_tight_layout()
    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)
        out = save_dir / "dashboard.png"
        plt.savefig(out, dpi=150, bbox_inches='tight')
    else:
        plt.show()
    plt.close(fig)


def plot_outliers(comprehensive_df: pd.DataFrame, metrics_to_check: list, outliers: dict, save_dir: Path = None):
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    for idx, metric in enumerate(metrics_to_check):
        ax = axes[idx]
        if metric not in comprehensive_df.columns:
            ax.set_visible(False)
            continue
        values = comprehensive_df[metric].values
        indices = range(len(values))
        ax.scatter(indices, values, s=100, alpha=0.6, color='steelblue', label='Normal')
        if metric in outliers:
            outlier_indices = [list(comprehensive_df.index).index(net) for net in outliers[metric]['networks']]
            outlier_values = outliers[metric]['values']
            ax.scatter(outlier_indices, outlier_values, s=200, alpha=0.8, color='red', marker='*', label='Outlier', edgecolors='black', linewidths=2)
            for oidx, oval in zip(outlier_indices, outlier_values):
                ax.annotate(f'{comprehensive_df.index[oidx]}', xy=(oidx, oval), xytext=(5, 5), textcoords='offset points', fontsize=8, bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
        mean_val = np.nanmean(values)
        std_val = np.nanstd(values)
        ax.axhline(mean_val, color='green', linestyle='--', linewidth=2, alpha=0.7, label='Mean')
        ax.axhline(mean_val + 1.5 * std_val, color='orange', linestyle=':', linewidth=1.5, alpha=0.5)
        ax.axhline(mean_val - 1.5 * std_val, color='orange', linestyle=':', linewidth=1.5, alpha=0.5)
        ax.set_xlabel('Network Index')
        ax.set_ylabel(metric.replace('_', ' ').title())
        ax.set_title(f'{metric.replace("_", " ").title()} - Outlier Detection')
        ax.legend(loc='best')
    safe_tight_layout()
    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)
        out = save_dir / "outliers.png"
        plt.savefig(out, dpi=150, bbox_inches='tight')
    else:
        plt.show()
    plt.close(fig)


def pca_visualization(X_scaled: np.ndarray, labels: np.ndarray, index_labels: list, save_dir: Path = None):
    if X_scaled is None or X_scaled.shape[0] < 2:
        return
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, s=200, alpha=0.6, cmap='viridis', edgecolors='black', linewidths=2)
    for i, net_id in enumerate(index_labels):
        plt.annotate(net_id, (X_pca[i, 0], X_pca[i, 1]), fontsize=10, ha='center', va='center', bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
    plt.title('Network Clustering Visualization (PCA)')
    plt.colorbar(scatter, label='Cluster')
    plt.grid(alpha=0.3)
    safe_tight_layout()
    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)
        out = save_dir / "pca_clusters.png"
        plt.savefig(out, dpi=150, bbox_inches='tight')
    else:
        plt.show()
    plt.close()


def radar_chart_by_cluster(comprehensive_df: pd.DataFrame, clustering_features: list, save_dir: Path = None):
    if 'cluster' not in comprehensive_df.columns:
        return
    optimal_k = int(comprehensive_df['cluster'].nunique())
    if optimal_k == 0:
        return
    normalized_df = comprehensive_df.copy()
    for col in clustering_features:
        min_val = normalized_df[col].min()
        max_val = normalized_df[col].max()
        if max_val > min_val:
            normalized_df[col] = (normalized_df[col] - min_val) / (max_val - min_val)
        else:
            normalized_df[col] = 0.5
    categories = [c.replace('_', ' ').title() for c in clustering_features]
    N = len(categories)
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]

    fig, axes = plt.subplots(1, optimal_k, figsize=(6 * optimal_k, 6), subplot_kw=dict(projection='polar'))
    if optimal_k == 1:
        axes = [axes]
    for i in range(optimal_k):
        ax = axes[i]
        cluster_df = normalized_df[normalized_df['cluster'] == i]
        if cluster_df.empty:
            cluster_mean = [0.5] * N
        else:
            cluster_mean = cluster_df[clustering_features].mean().values.tolist()
        cluster_mean += cluster_mean[:1]
        ax.plot(angles, cluster_mean, 'o-', linewidth=2, label=f'Cluster {i}')
        ax.fill(angles, cluster_mean, alpha=0.25)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories, size=10)
        ax.set_ylim(0, 1)
        ax.set_title(f'Cluster {i}\n({len(cluster_df)} networks)', size=14, weight='bold', pad=20)
        ax.grid(True)
    safe_tight_layout()
    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)
        out = save_dir / "radar_clusters.png"
        plt.savefig(out, dpi=150, bbox_inches='tight')
    else:
        plt.show()
    plt.close(fig)


def main(twitter_path: str, output_dir: str = None, do_plots: bool = True):
    ego_network_ids = find_ego_network_ids(twitter_path)
    all_networks = {}
    for node_id in ego_network_ids:
        all_networks[node_id] = load_ego_network(node_id, twitter_path=twitter_path)

    graphs = {}
    for node_id, network_data in all_networks.items():
        graphs[node_id] = build_ego_graph(node_id, network_data)

    network_stats = {}
    for node_id, G in graphs.items():
        network_stats[node_id] = compute_basic_stats(G)

    stats_df = pd.DataFrame(network_stats).T
    stats_df.index.name = 'network_id'

    degree_stats = {}
    degree_distributions = {}
    for node_id, G in graphs.items():
        stats, in_deg, out_deg = compute_degree_stats(G)
        degree_stats[node_id] = stats
        degree_distributions[node_id] = {'in': in_deg, 'out': out_deg}
    degree_stats_df = pd.DataFrame(degree_stats).T

    clustering_stats = {}
    clustering_distributions = {}
    for node_id, G in graphs.items():
        stats, coeffs = compute_clustering_stats(G)
        clustering_stats[node_id] = stats
        clustering_distributions[node_id] = coeffs
    clustering_df = pd.DataFrame(clustering_stats).T

    comprehensive_df = stats_df.copy()
    comprehensive_df = comprehensive_df.join(degree_stats_df)
    comprehensive_df = comprehensive_df.join(clustering_df)
    comprehensive_df['avg_degree'] = (comprehensive_df['mean_in_degree'] + comprehensive_df['mean_out_degree']) / 2

    # Ensure index is string
    comprehensive_df.index = comprehensive_df.index.map(str)

    # Outlier detection
    metrics_to_check = ['num_nodes', 'num_edges', 'density', 'avg_degree', 'avg_clustering', 'transitivity']
    outliers = identify_outliers(comprehensive_df, metrics_to_check, threshold=1.5)

    # Clustering
    clustering_features = ['num_nodes', 'num_edges', 'density', 'avg_degree', 'avg_clustering', 'transitivity']
    if len(comprehensive_df) >= 2:
        optimal_k = min(3, max(2, len(comprehensive_df) // 2))
        comprehensive_df, kmeans = cluster_networks(comprehensive_df, clustering_features, k=optimal_k)
    else:
        comprehensive_df['cluster'] = 0
        kmeans = None

    # Distance from mean classification
    mean_values = comprehensive_df[clustering_features].mean()
    std_values = comprehensive_df[clustering_features].std().replace(0, 1e-8)
    distances = []
    for idx in comprehensive_df.index:
        network_values = comprehensive_df.loc[idx, clustering_features]
        z_scores = (network_values - mean_values) / std_values
        distance = np.sqrt((z_scores ** 2).sum())
        distances.append(distance)
    comprehensive_df['distance_from_mean'] = distances
    comprehensive_df['is_typical'] = comprehensive_df['distance_from_mean'] < np.nanmedian(comprehensive_df['distance_from_mean'])

    typical_networks = comprehensive_df[comprehensive_df['is_typical']].index.tolist()
    unusual_networks = comprehensive_df[~comprehensive_df['is_typical']].index.tolist()

    # Print concise summary
    print("=" * 80)
    print("FINAL SUMMARY REPORT")
    print("=" * 80)
    print(f"Total ego networks analyzed: {len(ego_network_ids)}")
    print(f"Network IDs: {ego_network_ids}")
    if not comprehensive_df.empty:
        print(f"Average nodes: {comprehensive_df['num_nodes'].mean():.2f} | Average edges: {comprehensive_df['num_edges'].mean():.2f}")
        print(f"Average density: {comprehensive_df['density'].mean():.4f}")
        print(f"Average clustering: {comprehensive_df['avg_clustering'].mean():.4f}")
        print(f"Typical networks ({len(typical_networks)}): {typical_networks}")
        print(f"Unusual networks ({len(unusual_networks)}): {unusual_networks}")
    total_outliers = set()
    for metric, data in outliers.items():
        total_outliers.update(data['networks'])
    print(f"Networks with outlier metrics: {sorted(total_outliers)}")
    print("=" * 80)

    # If no output dir provided, save plots into a "plots" folder inside the twitter_path
    out_dir = Path(output_dir) if output_dir else Path(twitter_path) / "plots"
    if do_plots:
        try:
            plot_dashboard(comprehensive_df, save_dir=out_dir)
            plot_outliers(comprehensive_df, metrics_to_check, outliers, save_dir=out_dir)
            if kmeans is not None:
                X_scaled = StandardScaler().fit_transform(comprehensive_df[clustering_features].astype(float).values)
                pca_visualization(X_scaled, comprehensive_df['cluster'].values, comprehensive_df.index.tolist(), save_dir=out_dir)
            radar_chart_by_cluster(comprehensive_df, clustering_features, save_dir=out_dir)
        except Exception as e:
            print(f"Plotting skipped due to error: {e}")

    # Save summary CSVs if output_dir provided
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)
        comprehensive_df.to_csv(out_dir / "comprehensive_metrics.csv")
        with open(out_dir / "outliers.json", "w") as f:
            json.dump(outliers, f, indent=2)
    return comprehensive_df, outliers


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze Twitter ego networks (convert from notebook).")
    parser.add_argument("--data", "-d", type=str, default="twitter_ego/twitter", help="Path to twitter ego files")
    parser.add_argument("--out", "-o", type=str, default=None, help="Output directory to save plots and CSVs")
    parser.add_argument("--no-plots", action="store_true", help="Disable plotting (useful for headless environments)")
    args = parser.parse_args()
    main(args.data, output_dir=args.out, do_plots=not args.no_plots)