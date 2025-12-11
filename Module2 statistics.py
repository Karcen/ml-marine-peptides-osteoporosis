import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from pathlib import Path
import warnings

# --- 配置 (Configuration) ---
# 定义默认输入文件名
INPUT_FILENAME = "sequence_properties.csv"

# 输出目录
OUTPUT_DIR = Path("outputs/chapter2_statistics")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

warnings.filterwarnings('ignore')

# 绘图风格设置 (学术发表级)
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 14,
    'axes.titleweight': 'bold',
    'figure.figsize': (10, 6),
    'savefig.format': 'pdf',
    'savefig.dpi': 300
})


def load_data():
    """智能加载数据：尝试多个路径"""
    # 1. 尝试当前目录
    path = Path(INPUT_FILENAME)
    if path.exists():
        print(f"Loading data from: {path.resolve()}")
        return pd.read_csv(path)

    # 2. 尝试 Chapter 1 生成的目录
    path = Path("outputs/chapter1_databases") / INPUT_FILENAME
    if path.exists():
        print(f"Loading data from: {path.resolve()}")
        return pd.read_csv(path)

    print(f"Error: 找不到文件 {INPUT_FILENAME}。请确保 Module 1 或 3 已运行。")
    return None


# =============================================================================
# 1. 基础统计数据生成 (保存 CSV)
# =============================================================================

def save_descriptive_stats(df):
    """计算并保存描述性统计 (均值, 标准差等)"""
    print(">>> [Data] 计算描述性统计...")
    cols = ['length', 'molecular_weight', 'charge_ph7', 'hydrophobicity',
            'hydrophobic_ratio', 'isoelectric_point', 'aromaticity', 'instability_index', 'composite_score']

    valid_cols = [c for c in cols if c in df.columns]
    desc = df[valid_cols].describe().T
    desc = desc[['mean', 'std', 'min', '50%', 'max']]
    desc.columns = ['Mean', 'Std', 'Min', 'Median', 'Max']

    save_path = OUTPUT_DIR / "descriptive_statistics.csv"
    desc.to_csv(save_path)
    print(f"   已保存 CSV: {save_path}")


# =============================================================================
# Chapter 1 图表与数据 (数据概览)
# =============================================================================

def plot_fig1_sequence_length(df):
    """Figure 1.1: 序列长度分布"""
    print(">>> [Ch1] 绘制 Figure 1.1: 序列长度分布...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    sns.histplot(df['length'], kde=True, ax=ax1, color='#1f77b4', bins=20)
    ax1.set_title('Sequence Length Histogram')
    ax1.set_xlabel('Length (aa)')  # 新增: 单位
    ax1.text(0.05, 0.95, f'n={len(df)}', transform=ax1.transAxes)  # 新增: 样本量标注

    sns.boxplot(y=df['length'], ax=ax2, color='#aec7e8')
    ax2.set_title('Sequence Length Boxplot')
    ax2.text(0.05, 0.95, f'n={len(df)}', transform=ax2.transAxes)  # 新增: 样本量标注

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "sequence_length_distribution.pdf")
    plt.close()


def plot_fig2_amino_acid(df):
    """Figure 1.2: 氨基酸组成分析 + 保存 CSV"""
    print(">>> [Ch1] 绘制 Figure 1.2: 氨基酸组成...")

    aa_cols = [c for c in df.columns if c.startswith('aa_')]
    if not aa_cols: return

    # 计算均值
    aa_means = df[aa_cols].mean().sort_values(ascending=False)
    aa_labels = [c.replace('aa_', '') for c in aa_means.index]

    # --- 保存 CSV ---
    aa_df = pd.DataFrame({'Amino_Acid': aa_labels, 'Average_Count': aa_means.values})
    aa_df.to_csv(OUTPUT_DIR / "amino_acid_composition.csv", index=False)
    print(f"   已保存 CSV: {OUTPUT_DIR}/amino_acid_composition.csv")

    # 绘图
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    sns.barplot(x=aa_labels, y=aa_means.values, ax=ax1, palette='viridis')
    ax1.set_title('Average Amino Acid Composition')
    ax1.errorbar(range(len(aa_labels)), aa_means.values, yerr=aa_df['Average_Count'].std(), fmt='none', capsize=5)  # 新增: 误差条
    ax1.text(0.05, 0.95, f'n={len(df)}', transform=ax1.transAxes)  # 新增: 样本量

    subset = df[aa_cols].head(50).copy()
    subset.columns = [c.replace('aa_', '') for c in subset.columns]
    sns.heatmap(subset, ax=ax2, cmap='YlGnBu', cbar_kws={'label': 'Count'})
    ax2.set_title('Composition Heatmap (Top 50 Sequences)')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "amino_acid_composition.pdf")
    plt.close()


def plot_fig3_physicochemical(df):
    """Figure 1.3: 理化性质分布 (从 Module 6 迁移优化的功能)"""
    print(">>> [Ch1] 绘制 Figure 1.3: 理化性质分布...")
    cols = ['molecular_weight', 'hydrophobicity', 'charge_ph7',
            'hydrophobic_ratio', 'isoelectric_point', 'aromaticity']

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    for i, col in enumerate(cols):
        if col in df.columns:
            sns.histplot(df[col], kde=True, ax=axes[i], color='#2ca02c', alpha=0.6)
            axes[i].set_title(col.replace('_', ' ').title())
            axes[i].set_xlabel('')  # 新增: 单位在标题
            axes[i].text(0.05, 0.95, f'n={len(df)}', transform=axes[i].transAxes)  # 新增: 样本量

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "physicochemical_properties.pdf")
    plt.close()


def plot_fig4_pca(df):
    """
    Figure 1.4: 增强版 PCA 分析 (包含 Score Plot, Loading Plot 和 Scree Plot)
    """
    print(">>> [Ch1] 绘制 Figure 1.4: 增强版 PCA 分析 (Scores & Loadings)...")

    # 1. 准备数据
    # 选择用于 PCA 的数值型特征 (排除掉非物理性质的列)
    features = ['length', 'molecular_weight', 'charge_ph7', 'hydrophobicity',
                'isoelectric_point', 'aromaticity', 'instability_index', 'hydrophobic_ratio']

    # 确保列存在
    valid_features = [f for f in features if f in df.columns]

    # 去除缺失值并进行标准化 (Standardization is crucial for PCA)
    data_clean = df[valid_features].dropna()
    if len(data_clean) == 0:
        print("Error: 没有足够的数据进行 PCA 分析")
        return

    X = data_clean.values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 2. 执行 PCA
    pca = PCA(n_components=None)  # 先计算所有成分以查看方差
    pca.fit(X_scaled)

    # 获取解释方差
    exp_var = pca.explained_variance_ratio_
    cum_var = np.cumsum(exp_var)

    # 重新拟合用于绘图的前2个主成分
    pca_2d = PCA(n_components=2)
    coords = pca_2d.fit_transform(X_scaled)
    loadings = pca_2d.components_.T * np.sqrt(pca_2d.explained_variance_)  # 计算载荷因子

    # ==========================================
    # 3. 保存详细数据 (CSV)
    # ==========================================

    # (A) 保存 PCA 坐标 (Scores) - 用于看样本分布
    pca_score_df = pd.DataFrame(coords, columns=['PC1', 'PC2'])
    pca_score_df['id'] = df.loc[data_clean.index, 'id']  # 拼回 ID
    # 拼回一些关键属性用于后续作图着色 (可选)
    pca_score_df['Label_Hydrophobicity'] = df.loc[data_clean.index, 'hydrophobicity']
    pca_score_df.to_csv(OUTPUT_DIR / "pca_scores_coordinates.csv", index=False)

    # (B) 保存 载荷矩阵 (Loadings) - 用于看特征贡献
    # 这张表告诉你：PC1 到底代表什么物理意义？
    loading_df = pd.DataFrame(loadings, columns=['PC1_Loading', 'PC2_Loading'], index=valid_features)
    loading_df['Magnitude'] = np.sqrt(loading_df['PC1_Loading'] ** 2 + loading_df['PC2_Loading'] ** 2)  # 向量长度
    loading_df = loading_df.sort_values('Magnitude', ascending=False)
    loading_df.to_csv(OUTPUT_DIR / "pca_loadings.csv", index=False)

    # --- 绘图: 使用 Gridspec 布局 ---
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(2, 2, width_ratios=[1, 1], height_ratios=[1, 1.5])

    # --- 子图 1: 碎石图 (Scree Plot) ---
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.bar(range(1, len(exp_var) + 1), exp_var, alpha=0.6, color='#4c72b0', label='Individual')
    ax1.plot(range(1, len(cum_var) + 1), cum_var, 'r-o', label='Cumulative', linewidth=2)
    ax1.axhline(y=0.85, color='gray', linestyle='--', alpha=0.5)
    ax1.text(len(exp_var), 0.86, '85% Threshold', va='bottom', ha='right', fontsize=9, color='gray')

    ax1.set_xlabel('Principal Component')
    ax1.set_ylabel('Explained Variance Ratio')
    ax1.set_title(f'Scree Plot (PC1+PC2 = {cum_var[1]:.1%} Variance)', fontsize=14)
    ax1.legend(loc='center right')
    ax1.grid(True, linestyle='--', alpha=0.5)
    ax1.text(0.05, 0.95, f'n={len(data_clean)}', transform=ax1.transAxes)  # 新增: 样本量

    # --- 子图 2: 得分图 (Score Plot) ---
    # 展示样本分布，颜色映射疏水性 (或其他关键指标)
    ax2 = fig.add_subplot(gs[0, 1])
    sc = ax2.scatter(coords[:, 0], coords[:, 1],
                     c=data_clean['hydrophobicity'],  # 这里可以用 molecular_weight 或其他
                     cmap='viridis', alpha=0.7, edgecolors='w', s=60)

    ax2.set_xlabel(f'PC1 ({exp_var[0]:.1%} var)')
    ax2.set_ylabel(f'PC2 ({exp_var[1]:.1%} var)')
    ax2.set_title('PCA Score Plot (Colored by Hydrophobicity)', fontsize=14)
    plt.colorbar(sc, ax=ax2, label='Hydrophobicity')
    ax2.text(0.05, 0.95, f'n={len(data_clean)}', transform=ax2.transAxes)  # 新增: 样本量

    # 绘制原点十字线
    ax2.axhline(0, color='grey', linestyle='--', linewidth=0.8)
    ax2.axvline(0, color='grey', linestyle='--', linewidth=0.8)

    # --- 子图 3: 载荷图 (Biplot / Loadings Plot) ---
    # 这是解释 PCA 含义的核心图
    ax3 = fig.add_subplot(gs[1, :])  # 占据下方整行

    # 1. 先画样本点背景 (淡色)
    ax3.scatter(coords[:, 0], coords[:, 1], alpha=0.2, color='gray', s=10, label='Peptides')

    # 2. 画特征向量 (箭头)
    scaling_factor = np.max(np.abs(coords)) * 0.8  # 缩放箭头以适应图表

    texts = []
    for i, feature in enumerate(valid_features):
        # 获取载荷坐标
        v1 = loadings[i, 0]
        v2 = loadings[i, 1]

        # 简单的归一化以便绘图可见
        # 注意：这里为了可视化，通常会放大载荷向量
        arrow_x = v1 * 5  # 系数可根据实际数据范围调整，或者使用自动缩放
        arrow_y = v2 * 5

        ax3.arrow(0, 0, arrow_x, arrow_y, color='#d62728', alpha=0.8, head_width=0.1, linewidth=1.5)
        texts.append(ax3.text(arrow_x * 1.1, arrow_y * 1.1, feature, color='#d62728', fontsize=12, fontweight='bold'))

    ax3.set_xlabel(f'PC1 ({exp_var[0]:.1%}) - Main Variation')
    ax3.set_ylabel(f'PC2 ({exp_var[1]:.1%}) - Secondary Variation')
    ax3.set_title('PCA Biplot: Which properties drive the distribution?', fontsize=14)
    ax3.grid(True, linestyle='--', alpha=0.5)
    ax3.legend(loc='upper right')  # 新增: 图例

    # 尽量让标签不重叠 (如果没有 adjustText 库，这行只是普通的 text)
    # try:
    #     from adjustText import adjust_text
    #     adjust_text(texts, ax=ax3)
    # except ImportError:
    #     pass

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "pca_analysis.pdf")
    plt.close()

# =============================================================================
# Chapter 2 图表与数据 (统计验证)
# =============================================================================

def plot_fig5_normality(df):
    """Figure 2.1: 正态性检验 + 保存 CSV"""
    print(">>> [Ch2] 绘制 Figure 2.1: 正态性检验...")
    cols = ['length', 'molecular_weight', 'charge_ph7',
            'hydrophobicity', 'hydrophobic_ratio', 'isoelectric_point']

    # --- 计算并保存 CSV ---
    norm_results = []
    for col in cols:
        if col in df.columns:
            stat, p = stats.shapiro(df[col].dropna())
            norm_results.append({
                'Variable': col,
                'W_Statistic': stat,
                'p_value': p,
                'Normal': 'Yes' if p > 0.05 else 'No'
            })
    pd.DataFrame(norm_results).to_csv(OUTPUT_DIR / "normality_test_results.csv", index=False)
    print(f"   已保存 CSV: {OUTPUT_DIR}/normality_test_results.csv")

    # 绘图
    fig, axes = plt.subplots(6, 2, figsize=(12, 20))
    for i, col in enumerate(cols):
        if col not in df.columns: continue
        sns.histplot(df[col], kde=True, ax=axes[i, 0], color='#1f77b4', stat='density')
        mu, std = stats.norm.fit(df[col])
        xmin, xmax = axes[i, 0].get_xlim()
        x = np.linspace(xmin, xmax, 100)
        axes[i, 0].plot(x, stats.norm.pdf(x, mu, std), 'r--', label='Normal Fit')  # 新增: 图例
        axes[i, 0].set_title(f'{col} Distribution')
        axes[i, 0].legend()

        stats.probplot(df[col], dist="norm", plot=axes[i, 1])
        axes[i, 1].set_title(f'{col} Q-Q Plot')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "distribution_comparison.pdf")
    plt.close()


def plot_fig6_correlation(df):
    """Figure 2.3: 相关性分析 + 保存 CSV"""
    print(">>> [Ch2] 绘制 Figure 2.3: 相关性热图...")
    cols = ['length', 'molecular_weight', 'charge_ph7', 'hydrophobicity',
            'hydrophobic_ratio', 'isoelectric_point', 'aromaticity', 'composite_score']

    valid_cols = [c for c in cols if c in df.columns]
    corr = df[valid_cols].corr()

    # --- 保存 CSV (合并冗余表) ---
    corr.to_csv(OUTPUT_DIR / "correlation_matrix.csv")
    print(f"   已保存 CSV: {OUTPUT_DIR}/correlation_matrix.csv")

    # 新增: 统计测试 (Pearson r and p-value)
    corr_stats = pd.DataFrame(index=valid_cols, columns=valid_cols)
    for i in valid_cols:
        for j in valid_cols:
            if i != j:
                r, p = stats.pearsonr(df[i], df[j])
                corr_stats.loc[i, j] = f"{r:.2f} (p={p:.2e})"
    corr_stats.to_csv(OUTPUT_DIR / "correlation_stats.csv")

    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='RdBu_r', center=0, square=True)
    plt.title('Correlation Matrix')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "correlation_heatmap.pdf")
    plt.close()


def plot_fig7_size_bias(df):
    """
    Figure 2.4: 尺寸偏差分析 (Size Bias Analysis)
    从 Module 6 迁移过来的关键功能。
    展示 线性加权评分 (Composite Score) 与 分子量 (MW) 的强相关性，
    以此论证为什么要引入 ML 和配体效率 (LE)。
    """
    print(">>> [Ch2] 绘制 Figure 2.4: 尺寸偏差分析 (Size Bias)...")

    if 'composite_score' not in df.columns or 'molecular_weight' not in df.columns:
        print("   Skipping: 缺少 composite_score 或 molecular_weight 列")
        return

    plt.figure(figsize=(8, 6))

    # 散点图 + 回归线
    sns.regplot(data=df, x='molecular_weight', y='composite_score',
                scatter_kws={'alpha': 0.5, 's': 20, 'color': '#4c72b0'},  # Blue
                line_kws={'color': '#c44e52'})  # Red

    # 计算相关系数
    r, p = stats.pearsonr(df['molecular_weight'], df['composite_score'])  # 新增: 统计测试

    plt.title(f'Size Bias: Score vs. MW (r = {r:.2f}, p={p:.2e})', fontweight='bold')
    plt.xlabel('Molecular Weight (Da)')  # 新增: 单位
    plt.ylabel('Linear Composite Score')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.text(0.05, 0.95, f'n={len(df)}', transform=plt.gca().transAxes)  # 新增: 样本量

    # 添加注释
    plt.text(0.05, 0.9, f"Positive Correlation (r={r:.2f})\nindicates Size Bias",
             transform=plt.gca().transAxes, fontsize=12,
             bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "size_bias_analysis.pdf")
    plt.close()


# --- 主程序 ---
def main():
    df = load_data()
    if df is not None:
        print(f"成功加载 {len(df)} 条数据。")

        save_descriptive_stats(df)

        plot_fig1_sequence_length(df)
        plot_fig2_amino_acid(df)
        plot_fig3_physicochemical(df)
        plot_fig4_pca(df)

        plot_fig5_normality(df)
        plot_fig6_correlation(df)

        # 新增的关键分析
        plot_fig7_size_bias(df)

        print(f"\n全部完成！所有图片和CSV数据已保存至: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()