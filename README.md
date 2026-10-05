# Progetto 4 – Machine Learning per la Regressione

Pipeline completa di Machine Learning per la regressione applicata al **Multiclass Diabetes Dataset** (264 pazienti).
Il target è **HbA1c** (emoglobina glicata), stimata a partire da età, sesso, BMI, profilo lipidico e funzionalità renale.

## Contenuto

| File | Descrizione |
|---|---|
| `progetto_diabetes_regressione.ipynb` | Notebook con tutto il progetto e gli output |
| `Multiclass_Diabetes_Dataset.csv` | Dataset |
| `requirements.txt` | Librerie necessarie |

## Struttura del notebook

1. **Data Exploration & Preprocessing** – `describe()`, istogrammi, boxplot, scatter matrix, matrice di correlazione, `train_test_split`
2. **Confronto dei modelli** – Linear Regression, Decision Tree, Ridge, Lasso, KNN, SVR con 10-Fold Cross-Validation e metrica NMSE
3. **Hyperparameters Tuning** – Grid Search su `n_neighbors` del KNN e valutazione finale sul test set (MSE, R²)
4. **Analisi delle Performance** – Learning curves, bias e varianza
5. **Data Reduction** – PCA a due componenti e superficie di regressione nello spazio PC1-PC2

## Risultati principali

| Modello | NMSE (10-Fold CV) |
|---|---|
| Ridge | -4.06 |
| Linear Regression | -4.07 |
| SVR | -4.11 |
| KNN | -4.12 |
| Decision Tree | -4.94 |
| Lasso | -5.21 |
| **KNN ottimizzato (k = 8)** | **-3.81** |

Sul test set il KNN ottimizzato ottiene **MSE = 2.65** e **R² = 0.51**.

## Esecuzione

```bash
pip install -r requirements.txt
jupyter notebook progetto_diabetes_regressione.ipynb
```
