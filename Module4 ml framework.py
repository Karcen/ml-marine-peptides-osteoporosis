# pattern_score = 0.1 * cv_trend + 0.255 + 0.039 * sensitivity + 0.386 * mcc + 0.118 * auc

import multiprocessing as mp

try:
    mp.set_start_method('forkserver', force=True)
except RuntimeError:
    pass

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
import time
from scipy import stats
from collections import defaultdict
import copy
import json
import pickle  # Added for saving models and scaler

# --- Core Configuration ---
RANDOM_STATE = 42
TEST_SIZE = 0.2
EXTERNAL_TEST_SIZE = 0.1
CV_FOLDS = 5
N_JOBS = -1
REPEATS = 5

# Paths
INPUT_FILE = Path("sequence_properties.csv")
OUTPUT_DIR = Path("outputs/chapter3_ml")
APPENDIX_DIR = OUTPUT_DIR / "appendix_plots"
INDIVIDUAL_DIR = OUTPUT_DIR / "individual_models"
COMBINED_DIR = OUTPUT_DIR / "combined_plots"
LEARNING_CURVES_DIR = OUTPUT_DIR / "learning_curves"
HYBRID_DIR = OUTPUT_DIR / "hybrid_model"
RANKINGS_DIR = OUTPUT_DIR / "rankings"
MODELS_DIR = OUTPUT_DIR / "saved_models"  # New directory for saved models

for d in [OUTPUT_DIR, APPENDIX_DIR, INDIVIDUAL_DIR, COMBINED_DIR,
          LEARNING_CURVES_DIR, HYBRID_DIR, RANKINGS_DIR, MODELS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

warnings.filterwarnings('ignore')

# --- Imports ---
from sklearn.model_selection import (
    train_test_split, cross_val_score, learning_curve,
    ShuffleSplit, GridSearchCV, StratifiedKFold, KFold, cross_val_predict
)
from sklearn.preprocessing import RobustScaler, StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, matthews_corrcoef, classification_report,
    roc_auc_score, roc_curve, precision_recall_curve
)
from sklearn.base import BaseEstimator, ClassifierMixin, clone

# Classification models
from sklearn.linear_model import LogisticRegression, RidgeClassifier, SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier,
    AdaBoostClassifier, BaggingClassifier, HistGradientBoostingClassifier,
    VotingClassifier, StackingClassifier
)
from sklearn.svm import SVC, LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis

# PyTorch
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR

# Extension libraries
try:
    from xgboost import XGBClassifier

    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False
    print("Note: xgboost not installed, skipping this model.")

try:
    from lightgbm import LGBMClassifier

    LGBM_AVAILABLE = True
except ImportError:
    LGBM_AVAILABLE = False
    print("Note: lightgbm not installed, skipping this model.")

try:
    from catboost import CatBoostClassifier

    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False
    print("Note: catboost not installed, skipping this model.")

try:
    import shap

    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

# Plot style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 11,
    'figure.dpi': 150,
    'savefig.dpi': 300
})


# ============================================================================
#                        UTILITY FUNCTIONS
# ============================================================================

def levenshtein_distance(s1, s2):
    """计算两个字符串的编辑距离"""
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)
    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    return previous_row[-1]


def filter_similar_sequences(df, threshold=0.8):
    """过滤相似度过高的序列"""
    sequences = df['sequence'].tolist()
    to_keep = [True] * len(sequences)
    for i in range(len(sequences)):
        for j in range(i + 1, len(sequences)):
            if not to_keep[j]:
                continue
            dist = levenshtein_distance(sequences[i], sequences[j])
            sim = 1 - dist / max(len(sequences[i]), len(sequences[j]))
            if sim > threshold:
                to_keep[j] = False
    return df.iloc[to_keep].reset_index(drop=True)


# Amino acid encoding
AA_TO_IDX = {aa: i for i, aa in enumerate('ACDEFGHIKLMNPQRSTVWY')}
PAD_IDX = len(AA_TO_IDX)


def encode_sequences(sequences, max_len=100):
    """将氨基酸序列编码为数值张量"""
    encoded = np.zeros((len(sequences), max_len), dtype=int) + PAD_IDX
    for i, seq in enumerate(sequences):
        seq = seq.upper()[:max_len]
        for j, aa in enumerate(seq):
            if aa in AA_TO_IDX:
                encoded[i, j] = AA_TO_IDX[aa]
    return encoded


# ============================================================================
#                    CLASSIFICATION METRICS CALCULATOR
# ============================================================================

class ClassificationMetrics:
    """计算分类指标：Accuracy, Sensitivity, Specificity, Precision, F1, MCC"""

    @staticmethod
    def calculate_all(y_true, y_pred, y_prob=None):
        """计算所有分类指标"""
        metrics = {}

        # Basic metrics
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        metrics['precision'] = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        metrics['recall'] = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        metrics['f1'] = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        metrics['mcc'] = matthews_corrcoef(y_true, y_pred)

        # Confusion matrix based metrics
        cm = confusion_matrix(y_true, y_pred)

        # For binary classification
        if len(cm) == 2:
            tn, fp, fn, tp = cm.ravel()
            metrics['sensitivity'] = tp / (tp + fn) if (tp + fn) > 0 else 0
            metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0
        else:
            # For multi-class, use macro average
            metrics['sensitivity'] = metrics['recall']
            specificities = []
            for i in range(len(cm)):
                tn = np.sum(cm) - np.sum(cm[i, :]) - np.sum(cm[:, i]) + cm[i, i]
                fp = np.sum(cm[:, i]) - cm[i, i]
                spec = tn / (tn + fp) if (tn + fp) > 0 else 0
                specificities.append(spec)
            metrics['specificity'] = np.mean(specificities)

        # AUC if probabilities available
        if y_prob is not None:
            try:
                if len(np.unique(y_true)) == 2:
                    metrics['auc'] = roc_auc_score(y_true, y_prob[:, 1] if y_prob.ndim > 1 else y_prob)
                else:
                    metrics['auc'] = roc_auc_score(y_true, y_prob, multi_class='ovr', average='weighted')
            except:
                metrics['auc'] = np.nan
        else:
            metrics['auc'] = np.nan

        return metrics

    @staticmethod
    def format_metrics(metrics, precision=4):
        """格式化指标输出"""
        return {k: round(v, precision) if isinstance(v, float) else v
                for k, v in metrics.items()}


# ============================================================================
#                    TRAINING METRICS TRACKER
# ============================================================================

class TrainingMetricsTracker:
    """跟踪所有模型的训练指标
    用于判断最佳模型（Training Score下降 + CV Score上升）
    """

    def __init__(self):
        self.metrics = {}
        self.learning_curves_data = {}

    def add_model_metrics(self, name, train_sizes, train_scores, cv_scores,
                          learning_rate=None, losses=None, test_metrics=None):
        """添加模型的训练指标"""

        train_scores = np.array(train_scores)
        cv_scores = np.array(cv_scores)

        self.metrics[name] = {
            'train_sizes': np.array(train_sizes),
            'train_scores_mean': np.mean(train_scores, axis=1) if train_scores.ndim > 1 else train_scores,
            'train_scores_std': np.std(train_scores, axis=1) if train_scores.ndim > 1 else np.zeros_like(train_scores),
            'cv_scores_mean': np.mean(cv_scores, axis=1) if cv_scores.ndim > 1 else cv_scores,
            'cv_scores_std': np.std(cv_scores, axis=1) if cv_scores.ndim > 1 else np.zeros_like(cv_scores),
            'learning_rate': learning_rate,
            'losses': losses,
            'test_metrics': test_metrics or {},
        }

        train_mean = self.metrics[name]['train_scores_mean']
        cv_mean = self.metrics[name]['cv_scores_mean']

        self.metrics[name]['train_score_trend'] = self._calculate_trend(train_mean)
        self.metrics[name]['cv_score_trend'] = self._calculate_trend(cv_mean)
        self.metrics[name]['is_best_pattern'] = self._check_best_pattern(train_mean, cv_mean)
        self.metrics[name]['quality_score'] = self._calculate_quality_score(name)

        self.learning_curves_data[name] = {
            'train_sizes': self.metrics[name]['train_sizes'].tolist(),
            'train_scores_mean': self.metrics[name]['train_scores_mean'].tolist(),
            'train_scores_std': self.metrics[name]['train_scores_std'].tolist(),
            'cv_scores_mean': self.metrics[name]['cv_scores_mean'].tolist(),
            'cv_scores_std': self.metrics[name]['cv_scores_std'].tolist(),
        }

    def _calculate_trend(self, scores):
        """计算得分趋势"""
        if len(scores) < 2:
            return 0
        x = np.arange(len(scores))
        slope, _, _, _, _ = stats.linregress(x, scores)
        return slope

    def _check_best_pattern(self, train_scores, cv_scores):
        """
        检查是否满足最佳模式：
        - Training Score 随数据增加而下降或稳定
        - CV Score 随数据增加而上升或稳定
        - 两者差距收敛
        """
        if len(train_scores) < 3 or len(cv_scores) < 3:
            return False

        train_trend = self._calculate_trend(train_scores)
        cv_trend = self._calculate_trend(cv_scores)

        # 条件1: 训练得分下降或微小上升
        train_decreasing = train_trend <= 0.02

        # 条件2: CV得分上升或稳定
        cv_increasing = cv_trend >= -0.01

        # 条件3: 最终泛化间隙合理
        final_gap = abs(train_scores[-1] - cv_scores[-1])
        gap_reasonable = final_gap < 0.20

        # 条件4: 间隙收敛
        mid = len(train_scores) // 2
        early_gap = np.mean(np.abs(train_scores[:mid] - cv_scores[:mid]))
        late_gap = np.mean(np.abs(train_scores[mid:] - cv_scores[mid:]))
        gap_converging = late_gap <= early_gap * 1.1

        return train_decreasing and cv_increasing and gap_reasonable and gap_converging

    def _calculate_quality_score(self, name):
        """计算模型综合质量分数"""
        m = self.metrics.get(name)
        if m is None:
            return 0

        cv_final = m['cv_scores_mean'][-1] if len(m['cv_scores_mean']) > 0 else 0
        train_final = m['train_scores_mean'][-1] if len(m['train_scores_mean']) > 0 else 0

        gap = abs(train_final - cv_final)
        gap_penalty = max(0, 1 - gap * 2)
        cv_trend_bonus = 1 + max(0, m['cv_score_trend']) * 10
        stability_bonus = 1 / (1 + np.mean(m['cv_scores_std']))

        quality = cv_final * gap_penalty * cv_trend_bonus * stability_bonus

        if m['is_best_pattern']:
            quality *= 1.2

        return max(0, quality)

    def get_best_pattern_models(self):
        """获取满足最佳模式的模型"""
        return [name for name, m in self.metrics.items() if m['is_best_pattern']]

    def get_top_models_by_quality(self, n=5):
        """获取质量分数最高的前n个模型"""
        sorted_models = sorted(self.metrics.items(),
                               key=lambda x: x[1]['quality_score'],
                               reverse=True)
        return [name for name, _ in sorted_models[:n]]

    def save_learning_curves_csv(self, save_dir):
        """保存所有学习曲线数据到CSV"""
        for name, data in self.learning_curves_data.items():
            df = pd.DataFrame({
                'train_size': data['train_sizes'],
                'train_score_mean': data['train_scores_mean'],
                'train_score_std': data['train_scores_std'],
                'cv_score_mean': data['cv_scores_mean'],
                'cv_score_std': data['cv_scores_std'],
            })
            df.to_csv(save_dir / f"{name}_learning_curve.csv", index=False)


# ============================================================================
#                    LOSS AND LEARNING RATE TRACKER
# ============================================================================

class LossLRTracker:
    """记录所有模型的损失和学习率"""

    def __init__(self):
        self.records = {}

    def add_record(self, name, epochs, train_losses, val_losses=None,
                   learning_rates=None):
        """添加模型的损失记录"""
        self.records[name] = {
            'epochs': list(range(1, len(train_losses) + 1)),
            'train_loss': train_losses,
            'val_loss': val_losses if val_losses else [np.nan] * len(train_losses),
            'learning_rate': learning_rates if learning_rates else [np.nan] * len(train_losses),
        }

    def save_to_csv(self, save_dir):
        """保存所有记录到CSV"""
        for name, record in self.records.items():
            df = pd.DataFrame(record)
            df.to_csv(save_dir / f"{name}_loss_lr.csv", index=False)

        summary_data = []
        for name, record in self.records.items():
            valid_val = [v for v in record['val_loss'] if not np.isnan(v)]
            summary_data.append({
                'model': name,
                'final_train_loss': record['train_loss'][-1] if record['train_loss'] else np.nan,
                'final_val_loss': valid_val[-1] if valid_val else np.nan,
                'min_train_loss': min(record['train_loss']) if record['train_loss'] else np.nan,
                'min_val_loss': min(valid_val) if valid_val else np.nan,
                'final_lr': record['learning_rate'][-1] if record['learning_rate'] else np.nan,
            })
        pd.DataFrame(summary_data).to_csv(save_dir / "all_models_loss_summary.csv", index=False)

    def plot_individual(self, save_dir):
        """为每个模型绘制单独的损失曲线PDF"""
        for name, record in self.records.items():
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))

            ax1 = axes[0]
            epochs = record['epochs']
            ax1.plot(epochs, record['train_loss'], 'b-', label='Train Loss', linewidth=2)
            if record['val_loss'] and not all(np.isnan(record['val_loss'])):
                ax1.plot(epochs, record['val_loss'], 'r-', label='Val Loss', linewidth=2)
            ax1.set_xlabel('Epoch', fontsize=12)
            ax1.set_ylabel('Loss', fontsize=12)
            ax1.set_title(f'{name} - Loss Curve', fontsize=14, fontweight='bold')
            ax1.legend(fontsize=10)
            ax1.grid(True, alpha=0.3)

            ax2 = axes[1]
            if record['learning_rate'] and not all(np.isnan(record['learning_rate'])):
                ax2.plot(epochs, record['learning_rate'], 'g-', linewidth=2)
                ax2.set_ylabel('Learning Rate', fontsize=12)
            else:
                ax2.text(0.5, 0.5, 'Learning Rate\nNot Available',
                         ha='center', va='center', transform=ax2.transAxes, fontsize=14)
            ax2.set_xlabel('Epoch', fontsize=12)
            ax2.set_title(f'{name} - Learning Rate', fontsize=14, fontweight='bold')
            ax2.grid(True, alpha=0.3)

            plt.tight_layout()
            plt.savefig(save_dir / f"{name}_loss_lr.pdf", dpi=300, bbox_inches='tight')
            plt.close()

    def plot_combined(self, save_dir, group_size=5):
        """将4-6个模型的损失曲线组合到一个PDF"""
        model_names = list(self.records.keys())
        num_groups = (len(model_names) + group_size - 1) // group_size

        for g in range(num_groups):
            start_idx = g * group_size
            end_idx = min((g + 1) * group_size, len(model_names))
            group_models = model_names[start_idx:end_idx]

            n_models = len(group_models)
            n_cols = min(3, n_models)
            n_rows = (n_models + n_cols - 1) // n_cols

            fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows))
            if n_rows == 1 and n_cols == 1:
                axes = np.array([[axes]])
            elif n_rows == 1:
                axes = axes.reshape(1, -1)
            elif n_cols == 1:
                axes = axes.reshape(-1, 1)

            for idx, name in enumerate(group_models):
                row, col = idx // n_cols, idx % n_cols
                ax = axes[row, col]
                record = self.records[name]

                epochs = record['epochs']
                ax.plot(epochs, record['train_loss'], 'b-', label='Train', linewidth=1.5)
                if record['val_loss'] and not all(np.isnan(record['val_loss'])):
                    ax.plot(epochs, record['val_loss'], 'r-', label='Val', linewidth=1.5)

                ax.set_xlabel('Epoch', fontsize=10)
                ax.set_ylabel('Loss', fontsize=10)
                ax.set_title(name, fontsize=11, fontweight='bold')
                ax.legend(fontsize=8)
                ax.grid(True, alpha=0.3)

            for idx in range(len(group_models), n_rows * n_cols):
                row, col = idx // n_cols, idx % n_cols
                axes[row, col].set_visible(False)

            plt.tight_layout()
            plt.savefig(save_dir / f"combined_loss_group_{g + 1}.pdf", dpi=300, bbox_inches='tight')
            plt.close()


# ============================================================================
#                    PyTorch MODELS
# ============================================================================

class PyTorchClassifier(BaseEstimator, ClassifierMixin):
    """PyTorch分类器包装器"""

    def __init__(self, model_class, model_kwargs=None, epochs=100, lr=0.001,
                 batch_size=32, device='cpu', use_scheduler=True):
        self.model_class = model_class
        self.model_kwargs = model_kwargs or {}
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.device = device
        self.use_scheduler = use_scheduler

        self.model = None
        self.train_losses = []
        self.val_losses = []
        self.learning_rates = []
        self.classes_ = None

    def fit(self, X, y, val_X=None, val_y=None):
        self.train_losses = []
        self.val_losses = []
        self.learning_rates = []

        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)

        input_dim = X.shape[1] if len(X.shape) > 1 else 1
        self.model = self.model_class(input_dim=input_dim, n_classes=n_classes,
                                      **self.model_kwargs).to(self.device)

        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
        y_tensor = torch.tensor(y, dtype=torch.long).to(self.device)

        if val_X is not None and val_y is not None:
            val_X_tensor = torch.tensor(val_X, dtype=torch.float32).to(self.device)
            val_y_tensor = torch.tensor(val_y, dtype=torch.long).to(self.device)

        optimizer = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=1e-5)
        criterion = nn.CrossEntropyLoss()

        if self.use_scheduler:
            scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5,
                                          patience=10, verbose=False)

        dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
        loader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size,
                                             shuffle=True)

        for epoch in range(self.epochs):
            self.model.train()
            epoch_loss = 0
            for batch_X, batch_y in loader:
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            avg_train_loss = epoch_loss / len(loader)
            self.train_losses.append(avg_train_loss)
            self.learning_rates.append(optimizer.param_groups[0]['lr'])

            if val_X is not None:
                self.model.eval()
                with torch.no_grad():
                    val_outputs = self.model(val_X_tensor)
                    val_loss = criterion(val_outputs, val_y_tensor).item()
                self.val_losses.append(val_loss)

                if self.use_scheduler:
                    scheduler.step(val_loss)

        return self

    def predict(self, X):
        self.model.eval()
        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            outputs = self.model(X_tensor)
            _, predicted = torch.max(outputs, 1)
        return predicted.cpu().numpy()

    def predict_proba(self, X):
        self.model.eval()
        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            outputs = self.model(X_tensor)
            probs = torch.softmax(outputs, dim=1)
        return probs.cpu().numpy()


# Neural Network Architectures
class SimpleMLP(nn.Module):
    def __init__(self, input_dim, n_classes, hidden_dims=(128, 64)):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.ReLU(),
                nn.Dropout(0.3)
            ])
            prev_dim = h_dim
        layers.append(nn.Linear(prev_dim, n_classes))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class DeepMLP(nn.Module):
    def __init__(self, input_dim, n_classes, hidden_dims=(256, 128, 64, 32)):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.LeakyReLU(0.1),
                nn.Dropout(0.4)
            ])
            prev_dim = h_dim
        layers.append(nn.Linear(prev_dim, n_classes))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class ResidualMLP(nn.Module):
    def __init__(self, input_dim, n_classes, hidden_dim=128, n_blocks=3):
        super().__init__()
        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU()
        )

        self.res_blocks = nn.ModuleList([
            self._make_block(hidden_dim) for _ in range(n_blocks)
        ])

        self.output_layer = nn.Linear(hidden_dim, n_classes)

    def _make_block(self, dim):
        return nn.Sequential(
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim)
        )

    def forward(self, x):
        x = self.input_layer(x)
        for block in self.res_blocks:
            residual = x
            x = block(x)
            x = x + residual
            x = torch.relu(x)
        return self.output_layer(x)


class AttentionMLP(nn.Module):
    def __init__(self, input_dim, n_classes, hidden_dim=128):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.Tanh(),
            nn.Linear(hidden_dim // 4, 1)
        )
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc3 = nn.Linear(hidden_dim // 2, n_classes)
        self.dropout = nn.Dropout(0.3)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim // 2)

    def forward(self, x):
        x = torch.relu(self.bn1(self.fc1(x)))
        attn_weights = torch.softmax(self.attention(x), dim=0)
        x = x * attn_weights
        x = self.dropout(x)
        x = torch.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        return self.fc3(x)


# ============================================================================
#                    HybridPeptideNet - 智能混合模型
# ============================================================================

class HybridPeptideNet(BaseEstimator, ClassifierMixin):
    """
    HybridPeptideNet: 智能混合模型

    策略：
    1. 只选择满足"Training Score下降 + CV Score上升"的模型
    2. 使用质量分数加权的Stacking策略
    3. 动态调整组合权重避免性能下降
    4. 特征级别的模型选择
    """

    def __init__(self, base_models=None, meta_learner=None,
                 metrics_tracker=None, use_quality_weights=True,
                 min_models=3, max_models=10):
        self.base_models = base_models or {}
        self.meta_learner = meta_learner
        self.metrics_tracker = metrics_tracker
        self.use_quality_weights = use_quality_weights
        self.min_models = min_models
        self.max_models = max_models

        self.selected_models = {}
        self.model_weights = {}
        self.stacking_model = None
        self.voting_model = None
        self.classes_ = None
        self.best_strategy = None

    def _select_models(self):
        """智能选择要组合的模型"""
        if self.metrics_tracker is None:
            return list(self.base_models.keys())[:self.max_models]

        best_pattern_models = self.metrics_tracker.get_best_pattern_models()

        if len(best_pattern_models) < self.min_models:
            top_quality = self.metrics_tracker.get_top_models_by_quality(self.max_models)
            for m in top_quality:
                if m not in best_pattern_models:
                    best_pattern_models.append(m)
                if len(best_pattern_models) >= self.min_models:
                    break

        selected = self.metrics_tracker.get_top_models_by_quality(self.max_models)
        selected = [m for m in selected if m in self.base_models]

        return selected

    def _calculate_weights(self, selected_models):
        """计算模型权重"""
        if not self.use_quality_weights or self.metrics_tracker is None:
            return {m: 1.0 / len(selected_models) for m in selected_models}

        weights = {}
        total_quality = 0
        for m in selected_models:
            q = self.metrics_tracker._calculate_quality_score(m)
            weights[m] = q
            total_quality += q

        if total_quality > 0:
            weights = {m: w / total_quality for m, w in weights.items()}
        else:
            weights = {m: 1.0 / len(selected_models) for m in selected_models}

        return weights

    def fit(self, X, y, X_val=None, y_val=None):
        """训练混合模型"""
        self.classes_ = np.unique(y)

        selected_names = self._select_models()
        print(f"\n[HybridPeptideNet] Selected {len(selected_names)} models: {selected_names}")

        if len(selected_names) == 0:
            raise ValueError("No models selected for HybridPeptideNet!")

        self.model_weights = self._calculate_weights(selected_names)
        print(f"[HybridPeptideNet] Model weights: {self.model_weights}")

        self.selected_models = {}
        for name in selected_names:
            if name in self.base_models:
                model = clone(self.base_models[name])
                model.fit(X, y)
                self.selected_models[name] = model

        strategies = {}

        # 策略1: 加权投票
        estimators = [(name, self.selected_models[name]) for name in self.selected_models]
        weights_list = [self.model_weights[name] for name in self.selected_models]

        try:
            voting = VotingClassifier(
                estimators=estimators,
                voting='soft',
                weights=weights_list
            )
            voting.estimators_ = [self.selected_models[name] for name in self.selected_models]
            voting.le_ = LabelEncoder().fit(y)
            voting.classes_ = self.classes_
            strategies['weighted_voting'] = voting
        except Exception as e:
            print(f"[HybridPeptideNet] Weighted voting failed: {e}")

        # 策略2: Stacking
        try:
            meta = self.meta_learner or XGBClassifier(n_estimators=100, random_state=RANDOM_STATE) if XGB_AVAILABLE else RandomForestClassifier(random_state=RANDOM_STATE)

            stacking = StackingClassifier(
                estimators=[(name, clone(self.base_models[name])) for name in selected_names[:5]],
                final_estimator=meta,
                cv=3,
                n_jobs=N_JOBS,
                passthrough=True
            )
            stacking.fit(X, y)
            strategies['stacking'] = stacking
        except Exception as e:
            print(f"[HybridPeptideNet] Stacking failed: {e}")

        # 策略3: 简单平均投票
        try:
            simple_voting = VotingClassifier(
                estimators=estimators,
                voting='soft'
            )
            simple_voting.estimators_ = [self.selected_models[name] for name in self.selected_models]
            simple_voting.le_ = LabelEncoder().fit(y)
            simple_voting.classes_ = self.classes_
            strategies['simple_voting'] = simple_voting
        except Exception as e:
            print(f"[HybridPeptideNet] Simple voting failed: {e}")

        # 在验证集上评估
        if X_val is not None and y_val is not None:
            best_score = -1
            best_strategy_name = None

            for strategy_name, strategy_model in strategies.items():
                try:
                    y_pred = strategy_model.predict(X_val)
                    score = f1_score(y_val, y_pred, average='weighted')
                    print(f"[HybridPeptideNet] {strategy_name} validation F1: {score:.4f}")

                    if score > best_score:
                        best_score = score
                        best_strategy_name = strategy_name
                except Exception as e:
                    print(f"[HybridPeptideNet] {strategy_name} evaluation failed: {e}")

            self.best_strategy = best_strategy_name
            self.stacking_model = strategies.get(best_strategy_name)
        else:
            self.best_strategy = 'stacking' if 'stacking' in strategies else list(strategies.keys())[0]
            self.stacking_model = strategies.get(self.best_strategy)

        print(f"[HybridPeptideNet] Best strategy: {self.best_strategy}")

        return self

    def predict(self, X):
        if self.stacking_model is not None:
            return self.stacking_model.predict(X)
        else:
            predictions = []
            for name, model in self.selected_models.items():
                pred = model.predict(X)
                predictions.append(pred)

            predictions = np.array(predictions)
            final_pred = []
            for i in range(predictions.shape[1]):
                counts = np.bincount(predictions[:, i].astype(int), minlength=len(self.classes_))
                final_pred.append(np.argmax(counts))

            return np.array(final_pred)

    def predict_proba(self, X):
        if self.stacking_model is not None and hasattr(self.stacking_model, 'predict_proba'):
            return self.stacking_model.predict_proba(X)
        else:
            proba_sum = None
            total_weight = 0

            for name, model in self.selected_models.items():
                if hasattr(model, 'predict_proba'):
                    proba = model.predict_proba(X)
                    weight = self.model_weights.get(name, 1.0)

                    if proba_sum is None:
                        proba_sum = proba * weight
                    else:
                        proba_sum += proba * weight
                    total_weight += weight

            if proba_sum is not None and total_weight > 0:
                return proba_sum / total_weight
            else:
                n_samples = X.shape[0]
                n_classes = len(self.classes_)
                return np.ones((n_samples, n_classes)) / n_classes

    def get_model_contributions(self):
        return {
            'selected_models': list(self.selected_models.keys()),
            'weights': self.model_weights,
            'best_strategy': self.best_strategy,
        }


# ============================================================================
#                    MODEL DEFINITIONS
# ============================================================================

def get_all_models():
    """获取所有分类模型"""
    models = {}

    models['Baseline_MajorityClass'] = GaussianNB()

    # Linear Models
    models['LogisticRegression'] = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
    models['LogisticRegression_L1'] = LogisticRegression(penalty='l1', solver='saga',
                                                         max_iter=1000, random_state=RANDOM_STATE)
    models['LogisticRegression_L2'] = LogisticRegression(penalty='l2', max_iter=1000,
                                                         random_state=RANDOM_STATE)
    models['RidgeClassifier'] = RidgeClassifier(random_state=RANDOM_STATE)
    models['SGDClassifier'] = SGDClassifier(loss='log_loss', max_iter=1000,
                                            random_state=RANDOM_STATE)

    # Neighbors
    models['KNN_3'] = KNeighborsClassifier(n_neighbors=3)
    models['KNN_5'] = KNeighborsClassifier(n_neighbors=5)
    models['KNN_7'] = KNeighborsClassifier(n_neighbors=7)

    # SVM
    models['SVC_Linear'] = SVC(kernel='linear', probability=True, random_state=RANDOM_STATE)
    models['SVC_RBF'] = SVC(kernel='rbf', probability=True, random_state=RANDOM_STATE)
    models['SVC_Poly'] = SVC(kernel='poly', probability=True, random_state=RANDOM_STATE)
    models['LinearSVC'] = LinearSVC(dual='auto', max_iter=2000, random_state=RANDOM_STATE)

    # Naive Bayes
    models['GaussianNB'] = GaussianNB()

    # Discriminant Analysis
    models['LDA'] = LinearDiscriminantAnalysis()
    models['QDA'] = QuadraticDiscriminantAnalysis()

    # Trees
    models['DecisionTree'] = DecisionTreeClassifier(random_state=RANDOM_STATE)
    models['DecisionTree_Pruned'] = DecisionTreeClassifier(max_depth=10, min_samples_split=5,
                                                           random_state=RANDOM_STATE)

    # Ensemble - Bagging
    models['RandomForest'] = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE,
                                                    n_jobs=N_JOBS)
    models['RandomForest_Deep'] = RandomForestClassifier(n_estimators=200, max_depth=20,
                                                         random_state=RANDOM_STATE, n_jobs=N_JOBS)
    models['ExtraTrees'] = ExtraTreesClassifier(n_estimators=100, random_state=RANDOM_STATE,
                                                n_jobs=N_JOBS)
    models['Bagging'] = BaggingClassifier(n_estimators=50, random_state=RANDOM_STATE,
                                          n_jobs=N_JOBS)

    # Ensemble - Boosting
    models['GradientBoosting'] = GradientBoostingClassifier(n_estimators=100,
                                                            random_state=RANDOM_STATE)
    models['AdaBoost'] = AdaBoostClassifier(n_estimators=100, random_state=RANDOM_STATE)
    models['HistGradientBoosting'] = HistGradientBoostingClassifier(max_iter=100,
                                                                    random_state=RANDOM_STATE)

    if XGB_AVAILABLE:
        models['XGBoost'] = XGBClassifier(n_estimators=100, random_state=RANDOM_STATE,
                                          n_jobs=N_JOBS, verbosity=0, use_label_encoder=False,
                                          eval_metric='logloss')
        models['XGBoost_Deep'] = XGBClassifier(n_estimators=200, max_depth=8,
                                               random_state=RANDOM_STATE, n_jobs=N_JOBS,
                                               verbosity=0, use_label_encoder=False,
                                               eval_metric='logloss')

    if LGBM_AVAILABLE:
        models['LightGBM'] = LGBMClassifier(n_estimators=100, random_state=RANDOM_STATE,
                                            n_jobs=N_JOBS, verbose=-1)
        models['LightGBM_Deep'] = LGBMClassifier(n_estimators=200, max_depth=10,
                                                 random_state=RANDOM_STATE, n_jobs=N_JOBS,
                                                 verbose=-1)

    if CATBOOST_AVAILABLE:
        models['CatBoost'] = CatBoostClassifier(n_estimators=100, random_state=RANDOM_STATE,
                                                verbose=0)

    # Neural Networks (sklearn)
    models['MLP_Small'] = MLPClassifier(hidden_layer_sizes=(64,), max_iter=500,
                                        random_state=RANDOM_STATE, early_stopping=True)
    models['MLP_Medium'] = MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=500,
                                         random_state=RANDOM_STATE, early_stopping=True)
    models['MLP_Large'] = MLPClassifier(hidden_layer_sizes=(256, 128, 64), max_iter=500,
                                        random_state=RANDOM_STATE, early_stopping=True)
    models['MLP_Deep'] = MLPClassifier(hidden_layer_sizes=(256, 128, 64, 32), max_iter=500,
                                       random_state=RANDOM_STATE, early_stopping=True)

    return models


def get_pytorch_models():
    """获取PyTorch模型"""
    models = {}

    models['PyTorch_SimpleMLP'] = PyTorchClassifier(
        model_class=SimpleMLP, epochs=100, lr=0.001, batch_size=32
    )
    models['PyTorch_DeepMLP'] = PyTorchClassifier(
        model_class=DeepMLP, epochs=100, lr=0.001, batch_size=32
    )
    models['PyTorch_ResidualMLP'] = PyTorchClassifier(
        model_class=ResidualMLP, epochs=100, lr=0.001, batch_size=32
    )
    models['PyTorch_AttentionMLP'] = PyTorchClassifier(
        model_class=AttentionMLP, epochs=100, lr=0.001, batch_size=32
    )

    return models


# ============================================================================
#                    PLOTTING FUNCTIONS
# ============================================================================

def plot_learning_curve(estimator, name, X, y, save_path, cv=5):
    """绘制学习曲线"""
    cv_splitter = ShuffleSplit(n_splits=cv, test_size=0.2, random_state=RANDOM_STATE)

    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv_splitter, n_jobs=N_JOBS,
        train_sizes=np.linspace(0.2, 1.0, 5), scoring='accuracy'
    )

    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(train_sizes, train_mean, 'o-', color='#d62728', label='Training Score', linewidth=2)
    ax.plot(train_sizes, test_mean, 'o-', color='#2ca02c', label='Cross-Validation Score', linewidth=2)
    ax.fill_between(train_sizes, train_mean - train_std, train_mean + train_std,
                    alpha=0.15, color='#d62728')
    ax.fill_between(train_sizes, test_mean - test_std, test_mean + test_std,
                    alpha=0.15, color='#2ca02c')

    ax.set_title(f'Learning Curve: {name}', fontsize=14, fontweight='bold')
    ax.set_xlabel('Training Examples', fontsize=12)
    ax.set_ylabel('Accuracy Score', fontsize=12)
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, linestyle='--', alpha=0.6)

    train_trend = np.polyfit(range(len(train_mean)), train_mean, 1)[0]
    cv_trend = np.polyfit(range(len(test_mean)), test_mean, 1)[0]
    final_gap = abs(train_mean[-1] - test_mean[-1])

    info_text = f'Train trend: {train_trend:.4f}\nCV trend: {cv_trend:.4f}\nFinal gap: {final_gap:.4f}'
    ax.text(0.02, 0.02, info_text, transform=ax.transAxes, fontsize=9,
            verticalalignment='bottom', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()
    plt.savefig(save_path, format='pdf', dpi=300, bbox_inches='tight')
    plt.close()

    return train_sizes, train_scores, test_scores


def plot_confusion_matrix(y_true, y_pred, name, save_path, class_names=None):
    """绘制混淆矩阵"""
    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=class_names, yticklabels=class_names)
    ax.set_title(f'Confusion Matrix: {name}', fontsize=14, fontweight='bold')
    ax.set_xlabel('Predicted Label', fontsize=12)
    ax.set_ylabel('True Label', fontsize=12)

    plt.tight_layout()
    plt.savefig(save_path, format='pdf', dpi=300, bbox_inches='tight')
    plt.close()


def plot_roc_curve(y_true, y_prob, name, save_path, n_classes=2):
    """绘制ROC曲线"""
    fig, ax = plt.subplots(figsize=(8, 6))

    if n_classes == 2:
        fpr, tpr, _ = roc_curve(y_true, y_prob[:, 1] if y_prob.ndim > 1 else y_prob)
        auc = roc_auc_score(y_true, y_prob[:, 1] if y_prob.ndim > 1 else y_prob)
        ax.plot(fpr, tpr, linewidth=2, label=f'ROC (AUC = {auc:.4f})')
    else:
        for i in range(n_classes):
            y_true_bin = (y_true == i).astype(int)
            fpr, tpr, _ = roc_curve(y_true_bin, y_prob[:, i])
            auc = roc_auc_score(y_true_bin, y_prob[:, i])
            ax.plot(fpr, tpr, linewidth=2, label=f'Class {i} (AUC = {auc:.4f})')

    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title(f'ROC Curve: {name}', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, format='pdf', dpi=300, bbox_inches='tight')
    plt.close()


def plot_metrics_comparison(results_df, save_path, title='Model Comparison'):
    """绘制模型指标对比图"""
    metrics_cols = ['accuracy', 'sensitivity', 'specificity', 'precision', 'f1', 'mcc']
    available_cols = [c for c in metrics_cols if c in results_df.columns]

    if len(available_cols) == 0:
        return

    plot_df = results_df[['Model'] + available_cols].melt(
        id_vars='Model', var_name='Metric', value_name='Score'
    )

    fig, ax = plt.subplots(figsize=(16, 10))

    sns.barplot(data=plot_df, x='Model', y='Score', hue='Metric', ax=ax)

    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('Model', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    ax.legend(title='Metric', bbox_to_anchor=(1.02, 1), loc='upper left')
    ax.set_ylim(0, 1.1)
    ax.grid(True, axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, format='pdf', dpi=300, bbox_inches='tight')
    plt.close()


def plot_combined_learning_curves(models_data, save_path, group_size=6):
    """将多个模型的学习曲线组合到一个PDF"""
    model_names = list(models_data.keys())
    num_groups = (len(model_names) + group_size - 1) // group_size

    for g in range(num_groups):
        start_idx = g * group_size
        end_idx = min((g + 1) * group_size, len(model_names))
        group_models = model_names[start_idx:end_idx]

        n_models = len(group_models)
        n_cols = min(3, n_models)
        n_rows = (n_models + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows))
        if n_rows == 1 and n_cols == 1:
            axes = np.array([[axes]])
        elif n_rows == 1:
            axes = axes.reshape(1, -1)
        elif n_cols == 1:
            axes = axes.reshape(-1, 1)

        for idx, name in enumerate(group_models):
            row, col = idx // n_cols, idx % n_cols
            ax = axes[row, col]

            data = models_data[name]
            train_sizes = data['train_sizes']
            train_mean = data['train_scores_mean']
            train_std = data['train_scores_std']
            cv_mean = data['cv_scores_mean']
            cv_std = data['cv_scores_std']

            ax.plot(train_sizes, train_mean, 'o-', color='#d62728', label='Train', linewidth=1.5)
            ax.plot(train_sizes, cv_mean, 'o-', color='#2ca02c', label='CV', linewidth=1.5)
            ax.fill_between(train_sizes,
                            np.array(train_mean) - np.array(train_std),
                            np.array(train_mean) + np.array(train_std),
                            alpha=0.1, color='#d62728')
            ax.fill_between(train_sizes,
                            np.array(cv_mean) - np.array(cv_std),
                            np.array(cv_mean) + np.array(cv_std),
                            alpha=0.1, color='#2ca02c')

            ax.set_xlabel('Training Size', fontsize=10)
            ax.set_ylabel('Accuracy', fontsize=10)
            ax.set_title(name, fontsize=11, fontweight='bold')
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

        for idx in range(len(group_models), n_rows * n_cols):
            row, col = idx // n_cols, idx % n_cols
            axes[row, col].set_visible(False)

        plt.tight_layout()
        group_save_path = str(save_path).replace('.pdf', f'_group_{g + 1}.pdf')
        plt.savefig(group_save_path, dpi=300, bbox_inches='tight')
        plt.close()


# ============================================================================
#                    HYPERPARAMETER TUNING
# ============================================================================

def tune_hyperparams(model, name, X_train, y_train):
    """超参数调优"""
    pipe = Pipeline([('scaler', RobustScaler()), ('model', model)])
    param_grid = None

    if 'RandomForest' in name:
        param_grid = {'model__n_estimators': [50, 100, 200],
                      'model__max_depth': [None, 10, 20]}
    elif 'LogisticRegression' in name:
        param_grid = {'model__C': [0.1, 1.0, 10.0]}
    elif 'SVC' in name and 'Linear' not in name:
        param_grid = {'model__C': [0.1, 1, 10], 'model__gamma': ['scale', 'auto']}
    elif 'KNN' in name:
        param_grid = {'model__n_neighbors': [3, 5, 7, 10]}
    elif 'GradientBoosting' in name:
        param_grid = {'model__n_estimators': [50, 100], 'model__max_depth': [3, 5]}
    elif 'XGBoost' in name and XGB_AVAILABLE:
        param_grid = {'model__n_estimators': [50, 100], 'model__max_depth': [3, 6],
                      'model__learning_rate': [0.01, 0.1]}
    elif 'LightGBM' in name and LGBM_AVAILABLE:
        param_grid = {'model__n_estimators': [50, 100], 'model__max_depth': [3, 6],
                      'model__learning_rate': [0.01, 0.1]}
    elif 'MLP' in name:
        param_grid = {'model__alpha': [0.0001, 0.001],
                      'model__learning_rate_init': [0.001, 0.01]}

    if param_grid:
        try:
            grid = GridSearchCV(pipe, param_grid, cv=3, n_jobs=N_JOBS, scoring='accuracy')
            grid.fit(X_train, y_train)
            print(f"    Best params for {name}: {grid.best_params_}")
            return grid.best_estimator_
        except Exception as e:
            print(f"    Tuning failed for {name}: {e}")
            return pipe

    return pipe


# ============================================================================
#                    EXTRACT LOSS HISTORY
# ============================================================================

def get_loss_history(fitted, name, X_train, y_train, X_val, y_val):
    """提取模型的损失历史"""
    train_losses = []
    val_losses = []
    learning_rates = []

    if hasattr(fitted, 'named_steps'):
        inner = fitted.named_steps.get('model', fitted)
    else:
        inner = fitted

    if isinstance(inner, PyTorchClassifier):
        train_losses = inner.train_losses
        val_losses = inner.val_losses
        learning_rates = inner.learning_rates

    elif isinstance(inner, MLPClassifier):
        if hasattr(inner, 'loss_curve_'):
            train_losses = inner.loss_curve_
        if hasattr(inner, 'validation_scores_'):
            val_losses = [1 - s for s in inner.validation_scores_]

    elif isinstance(inner, (GradientBoostingClassifier, AdaBoostClassifier)):
        if hasattr(inner, 'train_score_'):
            train_losses = [1 - s for s in inner.train_score_]
        if hasattr(inner, 'staged_predict'):
            try:
                val_losses = []
                for pred in inner.staged_predict(X_val):
                    val_losses.append(1 - accuracy_score(y_val, pred))
            except:
                pass

    elif XGB_AVAILABLE and 'XGBoost' in name:
        if hasattr(inner, 'evals_result'):
            try:
                evals = inner.evals_result()
                if evals:
                    for key in evals:
                        if 'logloss' in evals[key]:
                            train_losses = evals[key]['logloss']
                            break
            except:
                pass

    elif LGBM_AVAILABLE and 'LightGBM' in name:
        if hasattr(inner, 'evals_result_'):
            evals = inner.evals_result_
            if 'training' in evals and 'binary_logloss' in evals['training']:
                train_losses = evals['training']['binary_logloss']

    return train_losses, val_losses, learning_rates


# ============================================================================
#                    RANKING SYSTEMS
# ============================================================================

class RankingSystem:
    """双重排名系统"""

    def __init__(self):
        self.all_results = []
        self.best_pattern_results = []

    def add_result(self, name, metrics, is_best_pattern=False,
                   train_trend=None, cv_trend=None):
        result = {
            'Model': name,
            **metrics,
            'is_best_pattern': is_best_pattern,
            'train_trend': train_trend,
            'cv_trend': cv_trend
        }
        self.all_results.append(result)

        if is_best_pattern:
            self.best_pattern_results.append(result)

    def get_general_ranking(self):
        """通用排名：基于Accuracy、F1、MCC"""
        df = pd.DataFrame(self.all_results)

        df['general_score'] = (
                df['accuracy'] * 0.3 +
                df['f1'] * 0.3 +
                df['mcc'].clip(lower=0) * 0.2 +
                df.get('auc', pd.Series([0.5] * len(df))).fillna(0.5) * 0.2
        )

        df = df.sort_values('general_score', ascending=False).reset_index(drop=True)
        df['Rank'] = range(1, len(df) + 1)

        return df

    def get_best_pattern_ranking(self):
        """最佳模式排名"""
        if not self.best_pattern_results:
            return pd.DataFrame()

        df = pd.DataFrame(self.best_pattern_results)

        df['pattern_score'] = (
                df['accuracy'] * 0.25 +
                df['f1'] * 0.25 +
                df['mcc'].clip(lower=0) * 0.15 +
                df.get('auc', pd.Series([0.5] * len(df))).fillna(0.5) * 0.15 +
                (-df['train_trend'].fillna(0)).clip(lower=0) * 0.1 +
                df['cv_trend'].fillna(0).clip(lower=0) * 0.1
        )

        df = df.sort_values('pattern_score', ascending=False).reset_index(drop=True)
        df['Rank'] = range(1, len(df) + 1)

        return df

    def save_rankings(self, save_dir):
        """保存排名结果"""
        general_df = self.get_general_ranking()
        cols_order = ['Rank', 'Model', 'accuracy', 'sensitivity', 'specificity',
                      'precision', 'f1', 'mcc', 'auc', 'general_score', 'is_best_pattern']
        cols_order = [c for c in cols_order if c in general_df.columns]
        general_df[cols_order].to_csv(save_dir / 'general_ranking.csv', index=False)

        best_df = self.get_best_pattern_ranking()
        if not best_df.empty:
            cols_order = ['Rank', 'Model', 'accuracy', 'sensitivity', 'specificity',
                          'precision', 'f1', 'mcc', 'auc', 'pattern_score',
                          'train_trend', 'cv_trend']
            cols_order = [c for c in cols_order if c in best_df.columns]
            best_df[cols_order].to_csv(save_dir / 'best_pattern_ranking.csv', index=False)

        return general_df, best_df


# ============================================================================
#                    MAIN FUNCTION
# ============================================================================

def create_classification_target(df, target_col='binding_affinity', n_classes=2):
    """创建分类目标变量"""
    if target_col in df.columns:
        y_continuous = df[target_col].values
    else:
        print(">>> Calculating binding affinity for classification...")
        try:
            from Bio.SeqUtils.ProtParam import ProteinAnalysis
            affinities = []
            for seq in df['sequence']:
                if len(seq) == 0:
                    affinities.append(0)
                    continue
                try:
                    analyzer = ProteinAnalysis(seq)
                    helix = analyzer.secondary_structure_fraction()[0]
                    turn = analyzer.secondary_structure_fraction()[2]
                    gravy = analyzer.gravy()
                    instability = analyzer.instability_index()
                    e_hydro = -15 * np.log1p(np.abs(gravy)) * np.sin(gravy * np.pi)
                    e_struct = -10 * (helix ** 2 - turn * 0.5) / (len(seq) / 100)
                    e_instab = 5 * np.tanh(instability / 50)
                    affinity = e_hydro + e_struct + e_instab
                    affinities.append(affinity)
                except:
                    affinities.append(0)
            y_continuous = np.array(affinities)
        except ImportError:
            print("   BioPython not available, using random target")
            y_continuous = np.random.randn(len(df))

    if n_classes == 2:
        threshold = np.median(y_continuous)
        y = (y_continuous > threshold).astype(int)
        class_names = ['Low Affinity', 'High Affinity']
    else:
        y = pd.qcut(y_continuous, q=n_classes, labels=False, duplicates='drop')
        class_names = [f'Class_{i}' for i in range(n_classes)]

    return y, class_names


def main():
    """主函数"""
    print("=" * 80)
    print("Module4 ML Framework Enhanced - HybridPeptideNet")
    print("Classification with: Accuracy, Sensitivity, Specificity, Precision, F1, MCC")
    print("=" * 80)

    # --- 1. 数据准备 ---
    print("\n>>> [1/6] Loading and preparing data...")

    if not INPUT_FILE.exists():
        print(f"Note: {INPUT_FILE} not found, creating synthetic data...")

        np.random.seed(RANDOM_STATE)
        n_samples = 500

        sequences = []
        for _ in range(n_samples):
            length = np.random.randint(10, 50)
            seq = ''.join(np.random.choice(list('ACDEFGHIKLMNPQRSTVWY'), length))
            sequences.append(seq)

        df = pd.DataFrame({
            'id': range(n_samples),
            'sequence': sequences,
            'length': [len(s) for s in sequences],
            'molecular_weight': np.random.uniform(1000, 5000, n_samples),
            'charge_ph7': np.random.uniform(-5, 5, n_samples),
            'hydrophobicity': np.random.uniform(-2, 2, n_samples),
            'isoelectric_point': np.random.uniform(4, 10, n_samples),
            'aromaticity': np.random.uniform(0, 0.3, n_samples),
            'instability_index': np.random.uniform(20, 60, n_samples),
        })
    else:
        df = pd.read_csv(INPUT_FILE)

    df = filter_similar_sequences(df, threshold=0.8)
    print(f"   Samples after filtering: {len(df)}")

    y, class_names = create_classification_target(df, n_classes=2)
    print(f"   Class distribution: {np.bincount(y)}")
    print(f"   Class names: {class_names}")

    exclude_cols = ['binding_affinity', 'id', 'sequence', 'description',
                    'Unnamed: 0', 'composite_score']
    feature_cols = [c for c in df.select_dtypes(include=[np.number]).columns
                    if c not in exclude_cols]
    print(f"   Features ({len(feature_cols)}): {feature_cols}")

    X = df[feature_cols].values

    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=0.15, random_state=RANDOM_STATE, stratify=y_train_val
    )

    print(f"   Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

    # --- 2. 初始化追踪器 ---
    metrics_tracker = TrainingMetricsTracker()
    loss_lr_tracker = LossLRTracker()
    ranking_system = RankingSystem()

    # --- 3. 训练所有模型 ---
    print("\n>>> [2/6] Training all models...")

    sklearn_models = get_all_models()
    pytorch_models = get_pytorch_models()
    all_models = {**sklearn_models, **pytorch_models}

    trained_models = {}
    results = []

    total = len(all_models)
    for i, (name, model) in enumerate(all_models.items(), 1):
        print(f"    [{i}/{total}] Training {name}...", end='')

        try:
            start_time = time.time()

            if name.startswith('PyTorch'):
                model_clone = copy.deepcopy(model)
                scaler = RobustScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_val_scaled = scaler.transform(X_val)
                X_test_scaled = scaler.transform(X_test)

                model_clone.fit(X_train_scaled, y_train, X_val_scaled, y_val)
                y_pred = model_clone.predict(X_test_scaled)
                y_prob = model_clone.predict_proba(X_test_scaled)

                loss_lr_tracker.add_record(
                    name,
                    epochs=len(model_clone.train_losses),
                    train_losses=model_clone.train_losses,
                    val_losses=model_clone.val_losses,
                    learning_rates=model_clone.learning_rates
                )

                fitted = model_clone
            else:
                pipe = tune_hyperparams(model, name, X_train, y_train)
                pipe.fit(X_train, y_train)
                y_pred = pipe.predict(X_test)

                if hasattr(pipe, 'predict_proba'):
                    try:
                        y_prob = pipe.predict_proba(X_test)
                    except:
                        y_prob = None
                else:
                    y_prob = None

                fitted = pipe

                train_losses, val_losses, lrs = get_loss_history(
                    fitted, name, X_train, y_train, X_val, y_val
                )
                if train_losses:
                    loss_lr_tracker.add_record(name, len(train_losses),
                                               train_losses, val_losses, lrs)

            trained_models[name] = fitted

            metrics = ClassificationMetrics.calculate_all(y_test, y_pred, y_prob)
            metrics = ClassificationMetrics.format_metrics(metrics)

            elapsed = time.time() - start_time
            print(
                f" Done ({elapsed:.1f}s) - Acc: {metrics['accuracy']:.4f}, F1: {metrics['f1']:.4f}, MCC: {metrics['mcc']:.4f}")

            try:
                train_sizes, train_scores, cv_scores = learning_curve(
                    clone(model) if not name.startswith('PyTorch') else model,
                    X_train_val, y_train_val, cv=3, n_jobs=N_JOBS,
                    train_sizes=np.linspace(0.2, 1.0, 5), scoring='accuracy'
                )

                metrics_tracker.add_model_metrics(
                    name, train_sizes, train_scores, cv_scores,
                    test_metrics=metrics
                )

                is_best_pattern = metrics_tracker.metrics[name]['is_best_pattern']
                train_trend = metrics_tracker.metrics[name]['train_score_trend']
                cv_trend = metrics_tracker.metrics[name]['cv_score_trend']
            except Exception as e:
                print(f"    Warning: Learning curve failed for {name}: {e}")
                is_best_pattern = False
                train_trend = None
                cv_trend = None

            ranking_system.add_result(name, metrics, is_best_pattern, train_trend, cv_trend)

            results.append({
                'Model': name,
                **metrics,
                'is_best_pattern': is_best_pattern
            })

            # Save individual model
            model_save_path = MODELS_DIR / f'{name}.pkl'
            with open(model_save_path, 'wb') as f:
                pickle.dump(fitted, f)

        except Exception as e:
            print(f" Error: {e}")
            continue

    # --- 4. 训练HybridPeptideNet ---
    print("\n>>> [3/6] Training HybridPeptideNet...")

    base_models_for_hybrid = {k: v for k, v in sklearn_models.items()
                              if k in trained_models}

    hybrid_model = HybridPeptideNet(
        base_models=base_models_for_hybrid,
        meta_learner=LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        metrics_tracker=metrics_tracker,
        use_quality_weights=True,
        min_models=3,
        max_models=10
    )

    try:
        scaler = RobustScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        X_test_scaled = scaler.transform(X_test)

        # Save scaler
        with open(HYBRID_DIR / 'scaler.pkl', 'wb') as f:
            pickle.dump(scaler, f)

        hybrid_model.fit(X_train_scaled, y_train, X_val_scaled, y_val)

        y_pred_hybrid = hybrid_model.predict(X_test_scaled)
        y_prob_hybrid = hybrid_model.predict_proba(X_test_scaled)

        hybrid_metrics = ClassificationMetrics.calculate_all(y_test, y_pred_hybrid, y_prob_hybrid)
        hybrid_metrics = ClassificationMetrics.format_metrics(hybrid_metrics)

        print(f"\n[HybridPeptideNet] Final Results:")
        print(f"   Accuracy:    {hybrid_metrics['accuracy']:.4f}")
        print(f"   Sensitivity: {hybrid_metrics['sensitivity']:.4f}")
        print(f"   Specificity: {hybrid_metrics['specificity']:.4f}")
        print(f"   Precision:   {hybrid_metrics['precision']:.4f}")
        print(f"   F1 Score:    {hybrid_metrics['f1']:.4f}")
        print(f"   MCC:         {hybrid_metrics['mcc']:.4f}")

        trained_models['HybridPeptideNet'] = hybrid_model
        ranking_system.add_result('HybridPeptideNet', hybrid_metrics,
                                  is_best_pattern=True, train_trend=None, cv_trend=None)
        results.append({
            'Model': 'HybridPeptideNet',
            **hybrid_metrics,
            'is_best_pattern': True
        })

        contributions = hybrid_model.get_model_contributions()
        with open(HYBRID_DIR / 'model_contributions.json', 'w') as f:
            json.dump(contributions, f, indent=2)

        # Save hybrid model
        with open(HYBRID_DIR / 'hybrid_model.pkl', 'wb') as f:
            pickle.dump(hybrid_model, f)

    except Exception as e:
        print(f"   Error training HybridPeptideNet: {e}")
        import traceback
        traceback.print_exc()

    # --- 5. 保存结果 ---
    print("\n>>> [4/6] Saving results...")

    results_df = pd.DataFrame(results)
    results_df.to_csv(OUTPUT_DIR / 'all_model_results.csv', index=False)

    general_ranking, best_pattern_ranking = ranking_system.save_rankings(RANKINGS_DIR)

    print(f"\n   === GENERAL RANKING (Top 10) ===")
    display_cols = ['Rank', 'Model', 'accuracy', 'sensitivity', 'specificity', 'precision', 'f1', 'mcc']
    display_cols = [c for c in display_cols if c in general_ranking.columns]
    print(general_ranking[display_cols].head(10).to_string(index=False))

    if not best_pattern_ranking.empty:
        print(f"\n   === BEST PATTERN RANKING (Training↓ + CV↑) ===")
        display_cols = ['Rank', 'Model', 'accuracy', 'f1', 'mcc', 'train_trend', 'cv_trend']
        display_cols = [c for c in display_cols if c in best_pattern_ranking.columns]
        print(best_pattern_ranking[display_cols].head(10).to_string(index=False))
    else:
        print("\n   No models satisfy the best pattern criteria (Train↓ + CV↑).")

    metrics_tracker.save_learning_curves_csv(LEARNING_CURVES_DIR)
    loss_lr_tracker.save_to_csv(INDIVIDUAL_DIR)

    # --- 6. 绘制图表 ---
    print("\n>>> [5/6] Generating plots...")

    print("   Plotting individual learning curves...")
    for name in list(metrics_tracker.learning_curves_data.keys())[:20]:
        data = metrics_tracker.learning_curves_data[name]

        fig, ax = plt.subplots(figsize=(10, 6))
        train_sizes = data['train_sizes']
        train_mean = data['train_scores_mean']
        train_std = data['train_scores_std']
        cv_mean = data['cv_scores_mean']
        cv_std = data['cv_scores_std']

        ax.plot(train_sizes, train_mean, 'o-', color='#d62728',
                label='Training Score', linewidth=2)
        ax.plot(train_sizes, cv_mean, 'o-', color='#2ca02c',
                label='CV Score', linewidth=2)
        ax.fill_between(train_sizes,
                        np.array(train_mean) - np.array(train_std),
                        np.array(train_mean) + np.array(train_std),
                        alpha=0.1, color='#d62728')
        ax.fill_between(train_sizes,
                        np.array(cv_mean) - np.array(cv_std),
                        np.array(cv_mean) + np.array(cv_std),
                        alpha=0.1, color='#2ca02c')

        ax.set_title(f'Learning Curve: {name}', fontsize=14, fontweight='bold')
        ax.set_xlabel('Training Examples', fontsize=12)
        ax.set_ylabel('Accuracy', fontsize=12)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

        if name in metrics_tracker.metrics:
            m = metrics_tracker.metrics[name]
            info = f"Train trend: {m['train_score_trend']:.4f}\n"
            info += f"CV trend: {m['cv_score_trend']:.4f}\n"
            info += f"Best pattern: {m['is_best_pattern']}"
            ax.text(0.02, 0.02, info, transform=ax.transAxes, fontsize=9,
                    verticalalignment='bottom',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        plt.tight_layout()
        plt.savefig(LEARNING_CURVES_DIR / f'{name}_learning_curve.pdf',
                    dpi=300, bbox_inches='tight')
        plt.close()

    print("   Plotting combined learning curves (groups of 6)...")
    plot_combined_learning_curves(
        metrics_tracker.learning_curves_data,
        COMBINED_DIR / 'combined_learning_curves.pdf',
        group_size=6
    )

    print("   Plotting individual loss curves...")
    loss_lr_tracker.plot_individual(INDIVIDUAL_DIR)

    print("   Plotting combined loss curves (groups of 5)...")
    loss_lr_tracker.plot_combined(COMBINED_DIR, group_size=5)

    print("   Plotting metrics comparison...")
    plot_metrics_comparison(results_df, OUTPUT_DIR / 'metrics_comparison.pdf',
                            title='Model Performance Comparison\n(Accuracy, Sensitivity, Specificity, Precision, F1, MCC)')

    if 'HybridPeptideNet' in trained_models:
        plot_confusion_matrix(y_test, y_pred_hybrid, 'HybridPeptideNet',
                              HYBRID_DIR / 'confusion_matrix.pdf', class_names)

        if y_prob_hybrid is not None:
            plot_roc_curve(y_test, y_prob_hybrid, 'HybridPeptideNet',
                           HYBRID_DIR / 'roc_curve.pdf', n_classes=len(class_names))

    top5 = general_ranking['Model'].head(5).tolist()
    for name in top5:
        if name in trained_models and name != 'HybridPeptideNet':
            model = trained_models[name]
            try:
                if name.startswith('PyTorch'):
                    y_pred = model.predict(X_test_scaled)
                else:
                    y_pred = model.predict(X_test)
                plot_confusion_matrix(y_test, y_pred, name,
                                      APPENDIX_DIR / f'{name}_confusion_matrix.pdf', class_names)
            except Exception as e:
                print(f"   Warning: Could not plot confusion matrix for {name}: {e}")

    # --- 7. 生成汇总报告 ---
    print("\n>>> [6/6] Generating summary report...")

    report = []
    report.append("=" * 80)
    report.append("HybridPeptideNet - Model Training Summary Report")
    report.append("=" * 80)
    report.append(f"\nDate: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    report.append(f"Total models trained: {len(trained_models)}")
    report.append(f"Total samples: {len(df)}")
    report.append(f"Features: {len(feature_cols)}")
    report.append(f"Classes: {class_names}")
    report.append(f"\nEvaluation Metrics: Accuracy, Sensitivity, Specificity, Precision, F1, MCC")

    report.append("\n" + "-" * 40)
    report.append("GENERAL RANKING (Top 10)")
    report.append("-" * 40)
    display_cols = ['Rank', 'Model', 'accuracy', 'sensitivity', 'specificity', 'precision', 'f1', 'mcc']
    display_cols = [c for c in display_cols if c in general_ranking.columns]
    report.append(general_ranking[display_cols].head(10).to_string(index=False))

    if not best_pattern_ranking.empty:
        report.append("\n" + "-" * 40)
        report.append("BEST PATTERN RANKING (Training↓ + CV↑)")
        report.append("-" * 40)
        display_cols = ['Rank', 'Model', 'accuracy', 'f1', 'mcc', 'train_trend', 'cv_trend']
        display_cols = [c for c in display_cols if c in best_pattern_ranking.columns]
        report.append(best_pattern_ranking[display_cols].to_string(index=False))

    if 'HybridPeptideNet' in trained_models:
        report.append("\n" + "-" * 40)
        report.append("HYBRIDPEPTIDENET DETAILS")
        report.append("-" * 40)
        contributions = hybrid_model.get_model_contributions()
        report.append(f"Selected models: {contributions['selected_models']}")
        report.append(f"Best strategy: {contributions['best_strategy']}")
        report.append("Model weights:")
        for m, w in contributions['weights'].items():
            report.append(f"  - {m}: {w:.4f}")

    report.append("\n" + "=" * 80)
    report.append("Output Files:")
    report.append("=" * 80)
    report.append(f"- Results CSV: {OUTPUT_DIR / 'all_model_results.csv'}")
    report.append(f"- General Ranking: {RANKINGS_DIR / 'general_ranking.csv'}")
    report.append(f"- Best Pattern Ranking: {RANKINGS_DIR / 'best_pattern_ranking.csv'}")
    report.append(f"- Learning Curves (individual): {LEARNING_CURVES_DIR}")
    report.append(f"- Learning Curves (combined): {COMBINED_DIR}")
    report.append(f"- Loss Curves (individual): {INDIVIDUAL_DIR}")
    report.append(f"- Loss Curves (combined): {COMBINED_DIR}")
    report.append(f"- Hybrid Model: {HYBRID_DIR}")

    report_text = "\n".join(report)
    print(report_text)

    with open(OUTPUT_DIR / 'training_report.txt', 'w') as f:
        f.write(report_text)

    # --- 8. 排序肽序列并输出CSV ---
    print("\n>>> [Extra] Sorting peptides by priority and saving to CSV...")
    try:
        # 选择最佳模型：优先HybridPeptideNet，否则使用排名最高的模型
        if 'HybridPeptideNet' in trained_models:
            best_model = trained_models['HybridPeptideNet']
            print("   Using HybridPeptideNet for priority scoring.")
        else:
            top_model_name = general_ranking['Model'].iloc[0]
            best_model = trained_models[top_model_name]
            print(f"   HybridPeptideNet not available. Using top model: {top_model_name} for priority scoring.")

        # 使用scaler缩放所有X
        X_all_scaled = scaler.transform(X)

        # 预测概率，高亲和力类（index 1）的概率作为优先级分数
        y_prob_all = best_model.predict_proba(X_all_scaled)
        df['priority_score'] = y_prob_all[:, 1]  # 高亲和力概率

        # 按优先级分数降序排序
        df_sorted = df.sort_values(by='priority_score', ascending=False).reset_index(drop=True)

        # 输出到CSV
        priority_csv_path = OUTPUT_DIR / 'peptides_priority.csv'
        df_sorted.to_csv(priority_csv_path, index=False)
        print(f"   Peptides sorted by priority and saved to: {priority_csv_path}")

    except Exception as e:
        print(f"   Error sorting peptides: {e}")

    print(f"\n>>> All completed! Results saved in: {OUTPUT_DIR.resolve()}")

    return trained_models, results_df, general_ranking, best_pattern_ranking


if __name__ == "__main__":
    trained_models, results_df, general_ranking, best_pattern_ranking = main()
