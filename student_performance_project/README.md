# 🎓 Student Performance Prediction

**ML Pipeline — Amina Kaich | ENIT 2025-2026**

Prédiction des notes de mathématiques d'étudiants à partir de facteurs socio-démographiques et scolaires.

---

## 📁 Structure du projet

```
student_performance_project/
├── data/
│   └── StudentsPerformance.csv     # Dataset (1000 étudiants, 8 features)
├── src/
│   └── train.py                    # Pipeline ML complet
├── outputs/                        # Graphiques générés automatiquement
│   ├── predictions_vs_real.png
│   ├── feature_importance.png
│   ├── score_distributions.png
│   └── model_comparison.png
├── requirements.txt
└── README.md
```

---

## 📊 Dataset

| Feature | Type | Description |
|---|---|---|
| gender | Catégorielle | female / male |
| race/ethnicity | Catégorielle | group A → E |
| parental level of education | Catégorielle | Niveau d'éducation des parents |
| lunch | Catégorielle | standard / free/reduced |
| test preparation course | Catégorielle | completed / none |
| math score | Numérique | **TARGET** — note de maths (0-100) |
| reading score | Numérique | Note de lecture |
| writing score | Numérique | Note d'écriture |

---

## 🔬 Pipeline ML

1. **Chargement** — lecture CSV avec Pandas
2. **Exploration** — statistiques descriptives, vérification des valeurs manquantes
3. **Feature Engineering** — One-Hot Encoding des variables catégorielles
4. **Split** — 80% train / 20% test (random_state=42)
5. **Normalisation** — StandardScaler
6. **Modèles** — Linear Regression (baseline) + Random Forest
7. **Évaluation** — MSE, R²
8. **Visualisations** — 4 graphiques exportés dans `/outputs`

---

## 📈 Résultats

| Modèle | R² | MSE |
|---|---|---|
| Linear Regression (baseline) | ~0.21 | ~186 |
| **Random Forest** | **~0.38** | **~146** |

> Le Random Forest surpasse significativement la régression linéaire.  
> Les features les plus importantes sont le **lunch** (indicateur socio-économique) et le **test preparation course**.

---

## 🚀 Installation & Exécution

```bash
# 1. Cloner le repo
git clone https://github.com/Amina-kaich/student-performance-prediction.git
cd student-performance-prediction

# 2. Installer les dépendances
pip install -r requirements.txt

# 3. Lancer le pipeline
python src/train.py
```

---

## 🛠️ Stack technique

- **Python 3.10+**
- **Pandas** — manipulation des données
- **Scikit-learn** — modèles ML et évaluation
- **Matplotlib** — visualisations
- **NumPy** — calculs numériques

---

## 💡 Points clés

- Comparaison baseline vs modèle avancé
- Pipeline reproductible avec random_state fixe
- Visualisations pensées pour un public non-technique
- Architecture modulaire et commentée

---

*Projet académique — ENIT Tunis | Télécommunications 2ème année*
