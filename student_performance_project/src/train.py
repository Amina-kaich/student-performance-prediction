"""
Student Performance Prediction
ML Pipeline — Amina Kaich, ENIT 2025-2026
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# ─────────────────────────────────────────
# 1. CHARGEMENT DES DONNÉES
# ─────────────────────────────────────────
DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "StudentsPerformance.csv")
OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "..", "outputs")
os.makedirs(OUTPUT_PATH, exist_ok=True)

df = pd.read_csv(DATA_PATH)
print("✅ Dataset chargé :", df.shape)
print(df.head())

# ─────────────────────────────────────────
# 2. EXPLORATION & NETTOYAGE
# ─────────────────────────────────────────
print("\n📊 Statistiques descriptives :")
print(df.describe())

print("\n🔍 Valeurs manquantes :")
print(df.isnull().sum())

# Vérification — aucune valeur manquante dans ce dataset
assert df.isnull().sum().sum() == 0, "Des valeurs manquantes ont été détectées !"

# ─────────────────────────────────────────
# 3. FEATURE ENGINEERING
# ─────────────────────────────────────────

# Encodage des variables catégorielles (One-Hot Encoding)
df_encoded = pd.get_dummies(df, drop_first=True)
print("\n📐 Colonnes après encodage :", list(df_encoded.columns))

# Target : math score (note la plus difficile à prédire)
TARGET = "math score"
X = df_encoded.drop(columns=[TARGET])
y = df_encoded[TARGET]

# ─────────────────────────────────────────
# 4. SPLIT TRAIN / TEST
# ─────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"\n📦 Train : {X_train.shape[0]} | Test : {X_test.shape[0]}")

# ─────────────────────────────────────────
# 5. NORMALISATION
# ─────────────────────────────────────────
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ─────────────────────────────────────────
# 6. MODÈLES
# ─────────────────────────────────────────
models = {
    "Linear Regression (baseline)": LinearRegression(),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42)
}

results = {}

for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    results[name] = {"model": model, "y_pred": y_pred, "MSE": mse, "R2": r2}
    print(f"\n🤖 {name}")
    print(f"   MSE : {mse:.2f}")
    print(f"   R²  : {r2:.4f}")

# ─────────────────────────────────────────
# 7. VISUALISATIONS
# ─────────────────────────────────────────

# — Prédictions vs Réel (Random Forest)
best_model_name = "Random Forest"
y_pred_best = results[best_model_name]["y_pred"]

plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred_best, alpha=0.6, color="steelblue", edgecolors="white", linewidth=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--", lw=2, label="Idéal")
plt.xlabel("Valeurs réelles (math score)", fontsize=12)
plt.ylabel("Prédictions", fontsize=12)
plt.title("Prédictions vs Réel — Random Forest", fontsize=14)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_PATH, "predictions_vs_real.png"), dpi=150)
plt.close()
print("\n📈 Graphique sauvegardé : predictions_vs_real.png")

# — Feature Importance
rf = results["Random Forest"]["model"]
importances = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=True)

plt.figure(figsize=(10, 6))
importances.plot(kind="barh", color="steelblue")
plt.xlabel("Importance", fontsize=12)
plt.title("Feature Importance — Random Forest", fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_PATH, "feature_importance.png"), dpi=150)
plt.close()
print("📊 Graphique sauvegardé : feature_importance.png")

# — Distribution des scores
fig, axes = plt.subplots(1, 3, figsize=(14, 4))
for ax, col in zip(axes, ["math score", "reading score", "writing score"]):
    df[col].hist(ax=ax, bins=20, color="steelblue", edgecolor="white")
    ax.set_title(col.title())
    ax.set_xlabel("Score")
    ax.set_ylabel("Fréquence")
plt.suptitle("Distribution des scores par matière", fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_PATH, "score_distributions.png"), dpi=150)
plt.close()
print("📊 Graphique sauvegardé : score_distributions.png")

# — Comparaison des modèles
model_names = list(results.keys())
r2_scores = [results[m]["R2"] for m in model_names]
mse_scores = [results[m]["MSE"] for m in model_names]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
ax1.bar(model_names, r2_scores, color=["#90caf9", "#1565c0"])
ax1.set_title("R² Score (plus haut = mieux)")
ax1.set_ylim(0, 1)
for i, v in enumerate(r2_scores):
    ax1.text(i, v + 0.01, f"{v:.3f}", ha="center", fontweight="bold")

ax2.bar(model_names, mse_scores, color=["#ef9a9a", "#b71c1c"])
ax2.set_title("MSE (plus bas = mieux)")
for i, v in enumerate(mse_scores):
    ax2.text(i, v + 0.5, f"{v:.1f}", ha="center", fontweight="bold")

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_PATH, "model_comparison.png"), dpi=150)
plt.close()
print("📊 Graphique sauvegardé : model_comparison.png")

print("\n✅ Pipeline terminé avec succès !")
print(f"   Meilleur modèle : {best_model_name}")
print(f"   R² : {results[best_model_name]['R2']:.4f}")
print(f"   MSE : {results[best_model_name]['MSE']:.2f}")
