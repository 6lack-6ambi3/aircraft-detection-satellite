"""
Aircraft Detection in Satellite Imagery
Binary classification: aircraft vs background (20x20 px chips)
Classical: HOG + SVM / RandomForest / k-NN
Modern:    Deep MLP (CNN-equivalent) on raw pixels
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use a non-interactive backend for matplotlib
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import time, warnings, os
warnings.filterwarnings('ignore')

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve, f1_score, precision_score, accuracy_score
from sklearn.pipeline import Pipeline
from skimage import feature
from skimage.filters import gaussian
from skimage import exposure

import json

np.random.seed(42)
JSON_FILE = '/Users/air/Documents/python-projects/python files/Image_Recognition_Systems/archive/planesnet.json'
OUTDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'aircraft_detection_results')
os.makedirs(OUTDIR, exist_ok=True)

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

DARK = '#0D1B2A'; DARK2 = '#141E2E'; CYAN = '#00D4FF'; RED = '#FF6B6B'
ORG = '#FFA500'; GLD = '#FFD700'; GRN = '#00FF88'; WHT = '#FFFFFF'
COLORS = [RED, ORG, GLD, CYAN]

REQ_CLASSICAL = { 'accuracy': 0.85,
                  'sensitivity': 0.82,
                  'specificity': 0.80,
                  'f1_score': 0.80,
                  'auc': 0.85
                  }

REQ_MODERN = { 'accuracy': 0.94,
               'sensitivity': 0.92,
               'specificity': 0.90,
               'f1_score': 0.90,
               'auc': 0.94
               }

#DATASET
def load_dataset(json_file):
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    X_raw = np.array(data['data'], dtype = np.uint8)
    X_raw = X_raw.reshape(-1, 3, 20, 20)
    X_raw = X_raw.transpose(0, 2, 3, 1)
    X = X_raw.astype(np.float32) / 255.0

    y = np.array(data['labels'], dtype = np.uint32)
    n_aircraft = y.sum()
    n_background = (y == 0).sum()
    print(f" Total chips: {len(y):,}")
    print(f" Aircraft chips: {n_aircraft:,} ({n_aircraft/len(y)*100:.1f}%)")
    print(f" Background chips: {n_background:,} ({n_background/len(y)*100:.1f}%)")
    return X, y


def preprocessing(X):
    out = X.copy().astype(np.float32)
    for i in range(len(out)):
        mn, mx = out[i].min(), out[i].max()
        if mx -mn > 1e-6:
            out[i] = (out[i] - mn) / (mx- mn)
    return out

def extract_hog_features(X):
    features = []
    for img in X:
        gray = 0.299 * img[:,:,0] + \
               0.587 * img[:,:,1] + \
               0.114 * img[:,:,2]

        hog_feature = feature.hog(gray, orientations=8, pixels_per_cell=(4, 4), cells_per_block=(2, 2), feature_vector=True)
        color_feats = []
        for channel_idx in range(3):
            hist, _ = np.histogram(img[:,:,channel_idx], bins=8, range=(0.0, 1.0))
            hist = hist.astype(float) / (hist.sum() + 1e-7)
            color_feats.append(hist)
        color_feats = np.concatenate(color_feats)
        combined_feats = np.concatenate([hog_feature, color_feats])
        features.append(combined_feats)
    return np.array(features)

def compute_metrics(y_true, y_pred, y_prob, model_name="Model-1"):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    sensitivity = tp / (tp + fn + 1e-9)
    specificity = tn / (tn + fp + 1e-9)
    precision = precision_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    accuracy = accuracy_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_prob)

    print(f"\n {'-'*40}")
    print(f" {model_name}")
    print(f" Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f" Sensitivity: {sensitivity:.4f} <- don't miss aircraft!")
    print(f" Specificity: {specificity:.4f} <- reject background correctly")
    print(f" Precision: {precision:.4f}")
    print(f" F1 Score: {f1:.4f}")
    print(f" AUC-ROC: {auc:.4f}") 
    print(f" TP={tp}, FP={fp}, TN={tn}, FN={fn}")

    return { "accuracy": accuracy,
             "sensitivity": sensitivity,
             "specificity": specificity,
             "precision": precision,
             "f1_score": f1,
             "auc": auc,
             "y_pred": y_pred,
             "y_prob": y_prob,
             "cm": confusion_matrix(y_true, y_pred),
             'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn
             }

def train_classical_classifiers(X_train_hog, X_test_hog, y_train, y_test):
    classifiers = {
        'SVM (RBF)': Pipeline([('scaler', StandardScaler()),
                               ('clf', SVC(kernel='rbf', C=10,
                                           gamma='scale', probability=True,
                                           class_weight='balanced', random_state=RANDOM_SEED))]),

        'Random Forest': Pipeline([
            ('scaler', StandardScaler()),
            ('clf', RandomForestClassifier(n_estimators=200,
                                           max_depth=15,
                                           min_samples_leaf=2,
                                           class_weight='balanced',
                                           random_state=RANDOM_SEED,
                                           n_jobs=-1))
        ]),

        'k-NN': Pipeline([
            ('scaler', StandardScaler()),
            ('clf', KNeighborsClassifier(n_neighbors=7,
                                         metric='euclidean',
                                         weights='distance',
                                         n_jobs=-1))
        ]),
    }
    results = {}
    trained_clfs = {}

    for name, pipeline in classifiers.items():
        print(f"\nTraining {name}...")
        start_time = time.time()
        pipeline.fit(X_train_hog, y_train)
        elapsed_time = time.time() - start_time
        print(f" Training time: {elapsed_time:.2f} seconds")

        y_pred = pipeline.predict(X_test_hog)
        infr_time = (time.time() - start_time) / len(X_test_hog) * 1000
        y_prob = pipeline.predict_proba(X_test_hog)[:, 1]

        metrics = compute_metrics(y_test, y_pred, y_prob, model_name=f"Classical - {name}")
        metrics['training_time'] = elapsed_time
        metrics['inference_time_per_sample_ms'] = infr_time
        results[name] = metrics
        trained_clfs[name] = pipeline

        print(f" Train time: {elapsed_time:.2f}s | "
              f" Inference time/sample: {infr_time:.3f} ms/sample")
        
    return results, trained_clfs

def train_neural_network(X_train, X_test, X_val, y_train, y_val, y_test):
    print("\n Preparing data for MLP...")

    X_flat_train = X_train.reshape(len(X_train), -1)
    X_flat_val = X_val.reshape(len(X_val), -1)
    X_flat_test = X_test.reshape(len(X_test), -1)

    scaler = StandardScaler()
    X_flat_train = scaler.fit_transform(X_flat_train)
    X_flat_val = scaler.transform(X_flat_val)
    X_flat_test = scaler.transform(X_flat_test)

    n_background = (y_train == 0).sum()
    n_aircraft = (y_train == 1).sum()
    class_weight_ratio = n_background / n_aircraft
    print(f" Class imbalance ratio (background/aircraft): {class_weight_ratio:.2f}x")
    sample_weight = np.where(y_train == 1, class_weight_ratio, 1.0)

    mlp = MLPClassifier(
        hidden_layer_sizes = (512, 256, 128, 64),
        activation = 'relu',
        solver = 'adam',
        learning_rate_init= 0.001,
        learning_rate='adaptive',
        alpha=1e-4,
        batch_size=256,
        max_iter=1,
        warm_start=True,
        random_state=RANDOM_SEED,
        early_stopping=False 
    )

    N_EPOCHS = 50
    history = {'loss': [], 'val_acc': []}
    best_val_acc = 0.0
    patience = 10
    patience_counter = 0

    print(f" Training MLP for {N_EPOCHS} epochs...")
    t_train_start = time.time()

    for epoch in range(N_EPOCHS):
        mlp.max_iter = epoch + 1
        mlp.fit(X_flat_train, y_train)
        current_loss = mlp.loss_curve_[-1]
        history['loss'].append(current_loss)

        val_pred = mlp.predict(X_flat_val)
        val_acc = accuracy_score(y_val, val_pred)
        history['val_acc'].append(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
        else:
            patience_counter += 1

        if (epoch + 1) % 10 == 0:
            print(f" Epoch {epoch+1:3d} / {N_EPOCHS} | "
                  f" Loss: {current_loss:.4f} | "
                  f" Val Acc: {val_acc:.4f} | ")
            
        if patience_counter >= patience:
            print(f" Early stopping at epoch {epoch+1} due to no improvement in validation accuracy for {patience} epochs.")
            break

    training_time = time.time() - t_train_start

    t0 = time.time()
    y_pred = mlp.predict(X_flat_test)
    inference_time = (time.time() - t0) / len(X_flat_test) * 1000
    y_prob = mlp.predict_proba(X_flat_test)[:, 1]

    results_cnn = compute_metrics(y_test, y_pred, y_prob, model_name="Neural Network - Deep MLP")
    results_cnn['training_time'] = training_time
    results_cnn['inference_time_per_sample_ms'] = inference_time
    print(f" Train time: {training_time:.1f}s | "
          f" Inference time/sample: {inference_time:.4f} ms/sample")
    
    return results_cnn, history

def style_ax(ax):
    ax.set_facecolor(DARK2)
    ax.tick_params(colors=WHT)
    for spine in ax.spines.values():
        spine.set_color('#444444')

def save_fig(filename):
    plt.savefig(os.path.join(OUTDIR, filename), 
                dpi=150,
                bbox_inches='tight',
                facecolor=DARK,
                edgecolor='none')
    plt.close()

def plot_sample_chips(X, y):
    fig, axes = plt.subplots(4, 10, figsize=(20, 8))
    fig.patch.set_facecolor(DARK)

    aircraft_idx = np.where(y == 1)[0][:20]
    background_idx = np.where(y == 0)[0][:20]

    for i, idx in enumerate(aircraft_idx):
        r, c = i // 10, i % 10
        axes[r, c].imshow(X[idx])
        axes[r, c].axis('off')
        axes[r, c].set_title('Aircraft', color=CYAN, fontsize=7)

    for i, idx in enumerate(background_idx):
        r, c = 2 + i // 10, i % 10
        axes[r, c].imshow(X[idx])
        axes[r, c].axis('off')
        axes[r, c].set_title('Background', color=RED, fontsize=7)

    fig.suptitle('PlanesNet Sample Chips (20x20 px)',color=WHT, fontsize=16, fontweight='bold')
    plt.tight_layout()
    save_fig('01_sample_chips.png')

def plot_class_distribution(y_train, y_test):
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    fig.patch.set_facecolor(DARK)

    for ax, yy, title in zip(axes, [y_train, y_test], ['Training Set', 'Test Set']):
        style_ax(ax)
        labels = ['Background', 'Aircraft']
        counts = [(yy == 0).sum(), (yy == 1).sum()]
        colors = [RED, CYAN]
        bars = ax.bar(labels, counts, color=colors, alpha = 0.88)

        for bar, c in zip(bars, counts):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 10,
                f'{c:,}\n({c/len(yy)*100:.1f}%)',
                ha='center', va='bottom', color=WHT, fontsize=11
            )

        ax.set_title(title, color=WHT, fontsize=14, fontweight='bold')
        ax.set_ylabel('Number of Chips', color=WHT)
        ax.set_ylim(0, max(counts) * 1.25)
    
    fig.suptitle('Class Distribution in PlanesNet Dataset(25% Aircraft/ 75% Background)', color=WHT, fontsize=13, fontweight='bold')
    plt.tight_layout()
    save_fig('02_class_distribution.png')

def plot_preprocessing_comparison(X_raw, X_proc, y):
    fig, axes = plt.subplots(2, 8, figsize=(20, 5))
    fig.patch.set_facecolor(DARK)

    a_idx = np.where(y == 1)[0][:4]
    b_idx = np.where(y == 0)[0][:4]

    for col, (idx, label) in enumerate([(i, 'Aircraft') for i in a_idx] + [(i, 'Background') for i in b_idx]):
        axes[0, col].imshow(X_raw[idx])
        axes[0, col].axis('off')
        axes[0, col].set_title(f'{label}\nRaw [0,255]', color=CYAN, fontsize=8)

        axes[1, col].imshow(X_proc[idx])
        axes[1, col].axis('off')
        axes[1, col].set_title('Normalized [0,1]', color=CYAN, fontsize=8)

    fig.suptitle('Preprocessing: Per-chip Min-Max Normalization', color=WHT, fontsize=13, fontweight='bold')
    plt.tight_layout()
    save_fig('03_preprocessing.png')

def plot_hog_visualization(X, y):
    fig, axes = plt.subplots(3,6, figsize=(18,9))
    fig.patch.set_facecolor(DARK)

    a_idx = np.where(y == 1)[0][:3]
    b_idx = np.where(y == 0)[0][:3]
    samples = list(a_idx) + list(b_idx)
    labels = ['Aircraft'] * 3 + ['Background'] * 3

    for col, (idx, label) in enumerate(zip(samples, labels)):
        chip = X[idx]
        gray = 0.299 * chip[:,:,0] + 0.587 * chip[:,:,1] + 0.114 * chip[:,:,2]

        _, hog_image = feature.hog(gray, orientations=8, pixels_per_cell=(4,4), cells_per_block=(2,2), visualize=True)
        hog_image = exposure.rescale_intensity(hog_image, in_range=(0, hog_image.max() + 1e-9))

        label_color = CYAN if 'Aircraft' in label else RED
        axes[0, col].imshow(chip)
        axes[0, col].axis('off')
        axes[0, col].set_title(label, color=label_color, fontsize=8)

        axes[1, col].imshow(gray, cmap='gray')
        axes[1, col].axis('off')
        axes[1, col].set_title('Grayscale', color=WHT, fontsize=8)

        axes[2, col].imshow(hog_image, cmap='inferno')
        axes[2, col].axis('off')
        axes[2, col].set_title('HOG Visualization', color=label_color, fontsize=8)

    fig.suptitle('HOG Feature Extraction Visualization', color=WHT, fontsize=13, fontweight='bold')
    plt.tight_layout()
    save_fig('04_hog_visualization.png')

def plot_confusion_matrix(cm, title, filename):
    """
    Heatmap of the confusion matrix with cell annotations.
    Rows = actual class, Colums = predicted class.
    """
    fig, ax = plt.subplots(figsize=(5, 4))
    fig.patch.set_facecolor(DARK)
    style_ax(ax)

    im = ax.imshow(cm, cmap='Blues')
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(['Background', 'Aircraft'], color=WHT, fontsize=10)
    ax.set_yticklabels(['Background', 'Aircraft'], color=WHT, fontsize=10)
    ax.set_xlabel('Predicted Label', color=WHT, fontsize=11)
    ax.set_ylabel('True Label', color=WHT, fontsize=11)
    ax.set_title(title, color=WHT, fontsize=13, fontweight='bold')

    for i in range(2):
        for j in range(2):
            text_color = 'white' if cm[i, j] < cm.max() * 0.6 else 'black'
            ax.text(j, i, f'{cm[i,j]}', ha='center', va='center', color=text_color, fontsize=13, fontweight='bold')

    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    save_fig(filename)

def plot_training_history(history):
    """Two-panel plot showing MLP training loss and validation accuracy over epochs."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.patch.set_facecolor(DARK)
    for ax in axes:
        style_ax(ax)

    axes[0].plot(history['loss'], color=RED, lw=2, label='Training Loss')
    axes[0].set_title('MLP Training Loss (Binary Cross-Entropy)', color=WHT, fontsize=13, fontweight='bold')
    axes[0].set_xlabel('Epoch', color=WHT)
    axes[0].set_ylabel('Loss', color=WHT)
    axes[0].legend(labelcolor=WHT, framealpha=0.8)

    axes[1].plot(history['val_acc'], color=CYAN, lw=2, label='Validation Accuracy')
    axes[1].axhline(0.94, color=GLD, linestyle='--', lw=1.5, label='Target Accuracy (94%)')

    axes[1].set_ylim(0, 1.05)
    axes[1].set_title('MLP Validation Accuracy per Epoch', color=WHT, fontsize=13, fontweight='bold')
    axes[1].set_xlabel('Epoch', color=WHT)
    axes[1].set_ylabel('Accuracy', color=WHT)
    axes[1].legend(labelcolor=WHT, framealpha=0.8)

    plt.title('MLP Training History: Loss and Validation Accuracy', color=WHT, fontsize=13, fontweight='bold')
    plt.tight_layout()
    save_fig('07_mlp_training_history.png')

def plot_roc_curves(all_results, y_test):
    """
    ROC curve for all 4 models overlaid on one plot.
 
    The ROC curve shows how the trade-off between sensitivity (catching aircraft)
    and specificity (rejecting background) changes as we vary the decision threshold.
    A perfect classifier hugs the top-left corner (AUC = 1.0).
    A random classifier follows the diagonal (AUC = 0.5).
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    fig.patch.set_facecolor(DARK)
    style_ax(ax)
 
    for (name, res), color in zip(all_results.items(), COLORS):
        fpr, tpr, _ = roc_curve(y_test, res['y_prob'])
        ax.plot(fpr, tpr, color=color, lw=2,
                label=f"{name}  (AUC = {res['auc']:.3f})")
 
    # Diagonal = random classifier baseline
    ax.plot([0, 1], [0, 1], '--', color='#555555', lw=1, label='Random')
 
    ax.set_xlabel('False Positive Rate  (1 − Specificity)', color=WHT)
    ax.set_ylabel('True Positive Rate  (Sensitivity / Recall)', color=WHT)
    ax.set_title('ROC Curves — All Models', color=WHT, fontweight='bold', fontsize=13)
    ax.legend(framealpha=0.2, labelcolor=WHT)
    plt.tight_layout()
    save_fig('08_roc_curves.png')
 
 
def plot_metrics_comparison(all_results):
    """
    Grouped bar chart comparing all 5 metrics across all 4 models.
    Horizontal dashed lines show the minimum required thresholds.
    """
    metric_keys   = ['accuracy', 'sensitivity', 'specificity', 'f1_score', 'auc']
    metric_labels = ['Accuracy', 'Sensitivity\n(Recall)', 'Specificity', 'F1-Score', 'AUC-ROC']
 
    # Minimum thresholds (dashed lines on the chart)
    req_classical = [0.85, 0.82, 0.80, 0.80, 0.85]
    req_modern    = [0.94, 0.92, 0.90, 0.90, 0.94]
 
    x     = np.arange(len(metric_keys))
    width = 0.18   # width of each bar
 
    fig, ax = plt.subplots(figsize=(13, 6))
    fig.patch.set_facecolor(DARK)
    style_ax(ax)
 
    for i, (name, color) in enumerate(zip(all_results.keys(), COLORS)):
        values = [all_results[name][k] for k in metric_keys]
        bars = ax.bar(x + i * width, values, width, label=name,
                      color=color, alpha=0.88)
 
        # Add value labels above each bar
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.005,
                    f'{val:.2f}', ha='center', va='bottom',
                    fontsize=7, color=WHT)
 
    # Draw threshold reference lines for each metric group
    for xi, rc, rm in zip(x + width * 1.5, req_classical, req_modern):
        ax.hlines(rc, xi - width * 2, xi + width * 2,
                  colors=ORG, linestyles='--', lw=1.2, alpha=0.8)
        ax.hlines(rm, xi - width * 2, xi + width * 2,
                  colors=GRN, linestyles='--', lw=1.2, alpha=0.8)
 
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(metric_labels, color=WHT)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel('Score', color=WHT)
    ax.set_title('Performance Metrics — All Models vs Requirements',
                 color=WHT, fontweight='bold', fontsize=13)
 
    # Combine model legend with threshold legend
    h1 = mlines.Line2D([], [], color=ORG, linestyle='--', label='Classical threshold')
    h2 = mlines.Line2D([], [], color=GRN, linestyle='--', label='Modern threshold')
    handles, labels_list = ax.get_legend_handles_labels()
    ax.legend(handles=handles + [h1, h2], framealpha=0.2,
              labelcolor=WHT, loc='upper left')
 
    plt.tight_layout()
    save_fig('09_metrics_comparison.png')
 
 
def plot_speed_comparison(all_results):
    """Bar chart of inference time per sample in milliseconds."""
    names  = list(all_results.keys())
    speeds = [all_results[n]['inference_time_per_sample_ms'] for n in names]
 
    fig, ax = plt.subplots(figsize=(8, 5))
    fig.patch.set_facecolor(DARK)
    style_ax(ax)
 
    bars = ax.bar(names, speeds, color=COLORS, alpha=0.88)
    for bar, t in zip(bars, speeds):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.001,
                f'{t:.4f} ms', ha='center', va='bottom',
                color=WHT, fontsize=11, fontweight='bold')
 
    ax.set_ylabel('Inference time per sample (ms)', color=WHT)
    ax.set_title('Processing Speed — Inference per Chip',
                 color=WHT, fontweight='bold', fontsize=13)
    plt.tight_layout()
    save_fig('10_speed_comparison.png')
 
 
def plot_feature_importance(trained_clfs):
    """
    Bar chart of the top 30 most important features from Random Forest.
 
    Feature importance in Random Forest measures how much each feature
    reduces impurity (uncertainty) across all the trees.
    Higher = that feature was more useful for splitting decisions.
    Features 0–511 are HOG bins; features 512–535 are color histogram bins.
    """
    rf_pipeline = trained_clfs['Random Forest']
    importances = rf_pipeline.named_steps['clf'].feature_importances_
    top_k = 30
    top_idx = np.argsort(importances)[-top_k:]
 
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor(DARK)
    style_ax(ax)
 
    # Color bars differently: cyan for HOG features, orange for color features
    bar_colors = [CYAN if i < 512 else ORG for i in top_idx]
    ax.barh(range(top_k), importances[top_idx], color=bar_colors, alpha=0.85)
    ax.set_yticks(range(top_k))
    ax.set_yticklabels([f'{"HOG" if i<512 else "Color"} #{i}' for i in top_idx],
                       color=WHT, fontsize=8)
    ax.set_xlabel('Feature Importance (mean impurity decrease)', color=WHT)
    ax.set_title('Top 30 Feature Importances — Random Forest\n'
                 '(Cyan = HOG features, Orange = Color histogram features)',
                 color=WHT, fontweight='bold', fontsize=12)
 
    # Legend
    import matplotlib.patches as mpatches
    h1 = mpatches.Patch(color=CYAN, label='HOG gradient feature')
    h2 = mpatches.Patch(color=ORG,  label='Color histogram feature')
    ax.legend(handles=[h1, h2], framealpha=0.2, labelcolor=WHT)
 
    plt.tight_layout()
    save_fig('06_feature_importance.png')

def main():
    """
    Master function that orchestrates the complete pipeline end-to-end.
 
    Flow:
      1. Load PlanesNet data
      2. Preprocess
      3. Split into train/val/test
      4. Visualize dataset
      5. Extract HOG features
      6. Train & evaluate classical classifiers
      7. Train & evaluate neural network
      8. Generate all comparison plots
      9. Print final summary table
    """
    print("\n" + "=" * 70)
    print("   AIRCRAFT DETECTION IN SATELLITE IMAGERY")
    print("   PlanesNet Dataset — Real Planet Labs Satellite Imagery")
    print("=" * 70)
 
    # ── Step 1: Load Data ─────────────────────────────────────────────────
    print("\n[STEP 1] Loading PlanesNet dataset...")
    X, y = load_dataset(JSON_FILE)
    # X.shape = (32000, 20, 20, 3)  — 32000 normalized RGB chips
    # y.shape = (32000,)            — 0=background, 1=aircraft
 
    # Keep raw (unnormalized) copy for visualization comparison
    X_raw_for_viz = (X * 255).astype(np.uint8)
 
    # ── Step 2: Preprocess ────────────────────────────────────────────────
    print("\n[STEP 2] Applying per-chip normalization...")
    X_proc = preprocessing(X)
    # X_proc has the same shape but each chip's pixel values now span [0,1]
 
    # ── Step 3: Train / Validation / Test Split ───────────────────────────
    print("\n[STEP 3] Splitting dataset (70% train / 15% val / 15% test)...")
    #
    # We split twice:
    #   First:  70% train, 30% temp
    #   Second: 50% of temp = val (15%), 50% of temp = test (15%)
    #
    # stratify=y ensures each split has the same class ratio (~25% aircraft).
    # Without stratify, a random split might put most aircraft in one set.
    #
    X_train, X_temp, y_train, y_temp = train_test_split(
        X_proc, y, test_size=0.30, stratify=y, random_state=RANDOM_SEED
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=RANDOM_SEED
    )
 
    print(f"  Train: {len(y_train):,} chips  "
          f"({y_train.sum():,} aircraft / {(y_train==0).sum():,} background)")
    print(f"  Val:   {len(y_val):,} chips  "
          f"({y_val.sum():,} aircraft / {(y_val==0).sum():,} background)")
    print(f"  Test:  {len(y_test):,} chips  "
          f"({y_test.sum():,} aircraft / {(y_test==0).sum():,} background)")
 
    # ── Step 4: Visualize Dataset ─────────────────────────────────────────
    print("\n[STEP 4] Generating dataset visualizations...")
    plot_sample_chips(X_proc, y)
    plot_class_distribution(y_train, y_test)
    plot_preprocessing_comparison(X_raw_for_viz, X_proc, y)
    plot_hog_visualization(X_proc, y)
    print(f"  Saved to: {OUTDIR}/")
 
    # ── Step 5: Extract HOG Features ─────────────────────────────────────
    print("\n[STEP 5] Extracting HOG features (this may take a minute)...")
    t0 = time.time()
    X_train_hog = extract_hog_features(X_train)
    X_test_hog  = extract_hog_features(X_test)
    elapsed = time.time() - t0
    print(f"  Feature vector dimension: {X_train_hog.shape[1]}")
    print(f"  Extraction time: {elapsed:.1f}s for "
          f"{len(X_train) + len(X_test):,} chips")
 
    # ── Step 6: Classical Classifiers ────────────────────────────────────
    print("\n[STEP 6] Training classical classifiers (SVM / RF / k-NN)...")
    results_classical, trained_clfs = train_classical_classifiers(
        X_train_hog, X_test_hog, y_train, y_test
    )
 
    # Plot confusion matrices for classical models
    for name, res in results_classical.items():
        fname = 'cm_' + name.lower().replace(' ', '_').replace('(', '').replace(')', '').replace('/', '') + '.png'
        plot_confusion_matrix(res['cm'], f'Confusion Matrix — {name}', fname)
 
    # Feature importance from Random Forest
    plot_feature_importance(trained_clfs)
 
    # ── Step 7: Neural Network ────────────────────────────────────────────
    print("\n[STEP 7] Training Deep MLP Neural Network...")
    results_cnn, history = train_neural_network(
        X_train, X_test, X_val, y_train, y_val, y_test
    )
 
    plot_confusion_matrix(results_cnn['cm'],
                          'Confusion Matrix — Deep MLP', 'cm_mlp.png')
    plot_training_history(history)
 
    # ── Step 8: Comparative Visualizations ───────────────────────────────
    print("\n[STEP 8] Generating comparison plots...")
    all_results = {**results_classical, 'Deep MLP': results_cnn}
    plot_roc_curves(all_results, y_test)
    plot_metrics_comparison(all_results)
    plot_speed_comparison(all_results)
 
    # ── Step 9: Final Summary Table ───────────────────────────────────────
    print("\n" + "=" * 80)
    print("  FINAL RESULTS SUMMARY")
    print("=" * 80)
    print(f"\n  {'Model':<18} {'Acc':>7} {'Sens':>7} {'Spec':>7} "
          f"{'F1':>7} {'AUC':>7} {'ms/chip':>9}  Status")
    print("  " + "─" * 75)
 
    for name, res in all_results.items():
        req = REQ_MODERN if name == 'Deep MLP' else REQ_CLASSICAL
        meets = all(res[k] >= req[k] for k in req)
        status = "✓ PASS" if meets else "✗ REVIEW"

        print(f"  {name:<18} "
              f"{res['accuracy']:>7.3f} "
              f"{res['sensitivity']:>7.3f} "
              f"{res['specificity']:>7.3f} "
              f"{res['f1_score']:>7.3f} "
              f"{res['auc']:>7.3f} "
              f"{res['inference_time_per_sample_ms']:>9.4f}  {status}")
 
    print("\n  Classical minimum:  Acc>85% | Sens>82% | Spec>80% | F1>0.80 | AUC>0.85")
    print("  Modern minimum:     Acc>94% | Sens>92% | Spec>90% | F1>0.90 | AUC>0.94")
    print(f"\n  All outputs saved to: {OUTDIR}/")
    print("=" * 80)
 
 
# ── Entry point ───────────────────────────────────────────────────────────────
# This block only runs when you execute the file directly:
#   python aircraft_detection_.py
# It does NOT run if someone imports the file as a module.
if __name__ == '__main__':
    main()