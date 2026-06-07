import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    auc,
    precision_recall_curve,
)
from sklearn.preprocessing import label_binarize
from sklearn.tree import plot_tree

from src.preprocessing import prepare_dataset

# --- 1. Veriyi Yükleme ve Ön İşleme ---
df, _, feature_cols = prepare_dataset("cars_dataframe.csv")

hedef_sutun = "Price_Category"
sinif_isimleri = ["Dusuk", "Orta", "Yuksek", "Luks"]

X = df[feature_cols]
y = df[hedef_sutun].astype(str)

# --- 2. Eğitim / Test Bölmesi (%80 / %20) ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# --- 3. Model Eğitimi ---
rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf_model.fit(X_train, y_train)

y_pred = rf_model.predict(X_test)
y_pred_proba = rf_model.predict_proba(X_test)

# --- 4. Performans Ölçütleri ---
cm = confusion_matrix(y_test, y_pred, labels=sinif_isimleri)

accuracy = accuracy_score(y_test, y_pred)
f1_macro = f1_score(y_test, y_pred, average="macro")
f1_weighted = f1_score(y_test, y_pred, average="weighted")

# Sınıf bazlı sensitivity (recall) ve specificity
specificities = {}
for i, label in enumerate(sinif_isimleri):
    tp = cm[i, i]
    fp = cm[:, i].sum() - tp
    fn = cm[i, :].sum() - tp
    tn = cm.sum() - tp - fp - fn
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    specificities[label] = specificity

report = classification_report(y_test, y_pred, labels=sinif_isimleri, output_dict=True)

print("-" * 40)
print("--- Model Performans Sonuçları ---")
print(f"Accuracy (Doğruluk):     {accuracy:.4f}")
print(f"F-measure (macro):       {f1_macro:.4f}")
print(f"F-measure (weighted):    {f1_weighted:.4f}")
print("-" * 40)
for label in sinif_isimleri:
    r = report[label]
    print(
        f"{label:8s} | Sensitivity: {r['recall']:.4f} | "
        f"Specificity: {specificities[label]:.4f} | F1: {r['f1-score']:.4f}"
    )
print("-" * 40)

# --- 5. Görselleştirmeler ---
OUTPUT_DIR = "outputs/figures_simple"
import os
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Confusion Matrix
plt.figure(figsize=(8, 6))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=sinif_isimleri,
    yticklabels=sinif_isimleri,
)
plt.title("Random Forest - Karmaşıklık Matrisi")
plt.xlabel("Tahmin Edilen Kategori")
plt.ylabel("Gerçek Kategori")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/confusion_matrix.png", bbox_inches="tight")
plt.close()

# Feature Importance (Top 10)
feature_importances = pd.Series(rf_model.feature_importances_, index=X.columns)
top_features = feature_importances.nlargest(10)

plt.figure(figsize=(10, 6))
top_features.plot(kind="barh", color="teal")
plt.title("En Önemli 10 Özellik (Feature Importance)")
plt.xlabel("Önem Skoru")
plt.ylabel("Özellikler")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/feature_importance.png", bbox_inches="tight")
plt.close()

# ROC Eğrileri (One-vs-Rest, 4 sınıf)
# predict_proba sütunları model.classes_ sırasındadır (alfabetik: Dusuk, Luks, Orta, Yuksek)
sinif_sirasi = list(rf_model.classes_)
y_bin = label_binarize(y_test, classes=sinif_sirasi)
plt.figure(figsize=(8, 6))
for i, label in enumerate(sinif_sirasi):
    fpr, tpr, _ = roc_curve(y_bin[:, i], y_pred_proba[:, i])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, lw=2, label=f"{label} (AUC = {roc_auc:.4f})")
plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Random Forest - ROC Eğrileri")
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/roc_curve.png", bbox_inches="tight")
plt.close()

# Sınıf Dağılımı
plt.figure(figsize=(6, 4))
sns.countplot(x=y, order=sinif_isimleri, palette="Set2")
plt.title("Fiyat Kategorisi Dağılımı")
plt.xlabel("Kategori")
plt.ylabel("Frekans")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/sinif_dagilimi.png", bbox_inches="tight")
plt.close()

# Metrikler Çubuk Grafiği
plt.figure(figsize=(8, 6))
metrik_isimleri = ["Accuracy", "F1 (macro)", "F1 (weighted)"]
metrik_degerleri = [accuracy, f1_macro, f1_weighted]
sns.barplot(x=metrik_isimleri, y=metrik_degerleri, palette="viridis")
plt.ylim(0, 1.1)
plt.title("Model Değerlendirme Metrikleri")
for i, v in enumerate(metrik_degerleri):
    plt.text(i, v + 0.02, f"{v:.4f}", ha="center", fontweight="bold")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/metrikler.png", bbox_inches="tight")
plt.close()

# Precision-Recall (sınıf sırası model.classes_ ile uyumlu)
plt.figure(figsize=(8, 6))
for i, label in enumerate(sinif_sirasi):
    precision, recall, _ = precision_recall_curve(y_bin[:, i], y_pred_proba[:, i])
    plt.plot(recall, precision, lw=2, label=label)
plt.xlabel("Recall (Duyarlılık)")
plt.ylabel("Precision (Kesinlik)")
plt.title("Precision-Recall Eğrileri")
plt.legend()
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/precision_recall_curve.png", bbox_inches="tight")
plt.close()

# Örnek Karar Ağacı (ilk ağaç, derinlik sınırlı)
plt.figure(figsize=(20, 10))
plot_tree(
    rf_model.estimators_[0],
    feature_names=list(X.columns),
    class_names=sinif_isimleri,
    filled=True,
    rounded=True,
    max_depth=3,
)
plt.title("Random Forest İçinden Örnek Bir Karar Ağacı (max_depth=3)")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/ornek_karar_agaci.png", bbox_inches="tight")
plt.close()

print(f"\nTüm grafikler kaydedildi: {OUTPUT_DIR}/")
