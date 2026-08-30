# Porto Seguro Safe Driver Prediction
# English below

Dieses Notebook fasst die zentralen Analysen und Modellierungsversuche des Projekts **Porto Seguro Safe Driver Prediction** zusammen. Ziel ist die Vorhersage, ob eine versicherte Person einen Schaden melden wird.

Das Notebook ergänzt das projektbegleitende Logbuch sowie die separate Prozess- und Ergebniszusammenfassung. Es konzentriert sich daher vor allem auf den relevanten, ausführbaren Code und die wichtigsten Visualisierungen.

## Datensatz

Verwendet wird der über OpenML verfügbare Datensatz **Porto Seguro Safe Driver Prediction**:

* OpenML Data ID: `42742`
* 595.212 Beobachtungen
* 57 Eingangsfeatures
* binäre Zielvariable `target`
* Anteil der positiven Klasse: ca. 3,64 %
* anonymisierte personenbezogene, regionale, fahrzeugbezogene und berechnete Features

In der OpenML-Version sind fehlende Werte als `NaN` codiert. Eine ID-Spalte ist nicht enthalten.

## Benötigte Bibliotheken

Das Notebook verwendet insbesondere:

* Python
* NumPy
* pandas
* Matplotlib
* Seaborn
* scikit-learn
* XGBoost
* LightGBM

## Inhalt des Notebooks

Das Notebook umfasst folgende Arbeitsschritte:

1. Laden und Aufbereiten des Datensatzes
2. Einteilung der Features in binäre, kategoriale und numerische Variablen
3. ausgewählte explorative Datenanalyse
4. Untersuchung der Klassenverteilung und fehlender Werte
5. logistische Regression als Baseline
6. logistische Regression mit PCA
7. Vergleich zweier Random-Forest-Konfigurationen
8. XGBoost-Experimente
9. LightGBM-Experimente
10. abschließender Modellvergleich

## Untersuchte Modelle

### Logistische Regression

Die logistische Regression dient als Baseline. Numerische Features werden imputiert und standardisiert. Kategoriale Features werden mittels One-Hot-Encoding verarbeitet.

Zusätzlich wird eine Variante getestet, bei der die numerischen Features vor der Modellierung durch eine PCA transformiert werden.

### Random Forest

Es werden zwei Modellvarianten verglichen:

* **Variante A:** keine Begrenzung der maximalen Baumtiefe
* **Variante B:** Begrenzung der Baumtiefe auf `max_depth=12`

Beide Varianten verwenden 300 Bäume, eine minimale Blattgröße von 50 und eine Gewichtung der Klassen.

### XGBoost und LightGBM

Für beide Boosting-Verfahren wird ein 2×2-Experiment durchgeführt:

* mit und ohne `ps_calc`-Features
* mit und ohne `scale_pos_weight`

Zusätzlich wird die Anzahl fehlender Werte je Beobachtung als Feature verwendet. Die kategorialen Features werden von XGBoost und LightGBM nativ verarbeitet.

## Evaluation

Alle Modelle werden mit einer fünffachen stratifizierten Cross-Validation evaluiert:

```python
StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)
```

Als Bewertungsmetriken dienen:

* ROC-AUC
* normalisierter Gini-Koeffizient
* Out-of-Fold-ROC-AUC
* Out-of-Fold-Gini

Der normalisierte Gini-Koeffizient wird aus der ROC-AUC berechnet:

```text
Gini = 2 × ROC-AUC − 1
```

Durch Out-of-Fold-Vorhersagen wird jede Beobachtung ausschließlich von einem Modell vorhergesagt, das diese Beobachtung während des Trainings nicht gesehen hat.

## Zentrale Ergebnisse

| Modell                 | Variante                        |   OOF-Gini |
| ---------------------- | ------------------------------- | ---------: |
| Random Forest          | Variante A                      |     0,2699 |
| LightGBM               | ohne `ps_calc`, ohne Gewichtung |     0,2676 |
| XGBoost                | ohne `ps_calc`, ohne Gewichtung |     0,2671 |
| Random Forest          | Variante B                      |     0,2638 |
| Logistische Regression | Baseline                        | ca. 0,2600 |
| Logistische Regression | PCA                             | ca. 0,2584 |

Random Forest Variante A erzielt das beste Einzelergebnis. Der Abstand zu LightGBM und XGBoost ist jedoch gering, sodass die drei baumbasierten Verfahren insgesamt eine vergleichbare Modellgüte zeigen.

Die PCA führt gegenüber der logistischen Baseline zu keiner Verbesserung. Bei XGBoost und LightGBM schneiden die Varianten ohne `ps_calc` und ohne Klassengewichtung am besten ab.

## Ausführung

Das Notebook sollte von oben nach unten ausgeführt werden. Der Datensatz wird während der Ausführung über OpenML geladen, weshalb eine Internetverbindung erforderlich ist.

Durch den festgelegten Zufallszustand

```python
RANDOM_STATE = 42
```

und identische stratifizierte Folds sind die Ergebnisse weitgehend reproduzierbar.

Die Cross-Validation der Random-Forest- und Boosting-Modelle kann aufgrund der Größe des Datensatzes einige Zeit in Anspruch nehmen.




## English
# Porto Seguro Safe Driver Prediction

This notebook summarizes the central analyses and modeling experiments conducted as part of the **Porto Seguro Safe Driver Prediction** project. The objective is to predict whether an insured person will file an insurance claim.

The notebook complements the project logbook and the separate process and results summary. It therefore focuses primarily on the relevant executable code and the most important visualizations.

## Dataset

The **Porto Seguro Safe Driver Prediction** dataset, available through OpenML, is used:

* OpenML Data ID: `42742`
* 595,212 observations
* 57 input features
* binary target variable `target`
* positive-class share: approximately 3.64%
* anonymized personal, regional, vehicle-related, and calculated features

In the OpenML version, missing values are encoded as `NaN`. The dataset does not contain an ID column.

## Notebook Contents

The notebook covers the following steps:

1. Loading and preparing the dataset
2. Dividing the features into binary, categorical, and numerical variables
3. Selected exploratory data analysis
4. Examination of the class distribution and missing values
5. Logistic regression as a baseline
6. Logistic regression with PCA
7. Comparison of two Random Forest configurations
8. XGBoost experiments
9. LightGBM experiments
10. Final model comparison

## Required Libraries

The notebook primarily uses:

* Python
* NumPy
* pandas
* Matplotlib
* Seaborn
* scikit-learn
* XGBoost
* LightGBM

## Models Evaluated

### Logistic Regression

Logistic regression serves as the baseline model. Numerical features are imputed and standardized. Categorical features are processed using one-hot encoding.

An additional variant is evaluated in which PCA is applied to the numerical features before modeling.

### Random Forest

Two model configurations are compared:

* **Variant A:** no limitation on the maximum tree depth
* **Variant B:** maximum tree depth limited to `max_depth=12`

Both variants use 300 trees, a minimum leaf size of 50, and class weighting.

### XGBoost and LightGBM

A 2×2 experiment is conducted for both boosting methods:

* with and without `ps_calc` features
* with and without `scale_pos_weight`

In addition, the number of missing values per observation is included as a feature. Categorical features are processed natively by XGBoost and LightGBM.

## Evaluation

All models are evaluated using five-fold stratified cross-validation:

```python
StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)
```

The following evaluation metrics are used:

* ROC-AUC
* normalized Gini coefficient
* out-of-fold ROC-AUC
* out-of-fold Gini

With out-of-fold predictions, each observation is predicted exclusively by a model that did not see that observation during training.

## Key Results

| Model               | Variant                                    |       OOF Gini |
| ------------------- | ------------------------------------------ | -------------: |
| Random Forest       | Variant A                                  |         0.2699 |
| LightGBM            | without `ps_calc`, without class weighting |         0.2676 |
| XGBoost             | without `ps_calc`, without class weighting |         0.2671 |
| Random Forest       | Variant B                                  |         0.2638 |
| Logistic Regression | Baseline                                   | approx. 0.2600 |
| Logistic Regression | PCA                                        | approx. 0.2584 |

Random Forest Variant A achieves the best individual result. However, the difference compared with LightGBM and XGBoost is small, indicating that the three tree-based methods provide broadly comparable model performance.

PCA does not improve performance compared with the logistic regression baseline. For both XGBoost and LightGBM, the variants without `ps_calc` features and without class weighting achieve the best results.

## Execution

The notebook should be executed from top to bottom. The dataset is loaded from OpenML during execution, so an internet connection is required.

The fixed random state `RANDOM_STATE = 42` and identical stratified folds make the results largely reproducible.

Due to the size of the dataset, cross-validation for the Random Forest and boosting models may take some time.

