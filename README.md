# Who Is The Killer?

Identifying which of 8 serial killers committed each crime, using classical pattern recognition and machine learning with scikit-learn.

*Course project for Pattern Recognition, Department of Informatics, University of Piraeus (2025–26).*

![Crime locations by killer](figures/spatial_distribution.png)

## The problem

The "Piraeus Vice" dataset has 4,800 crimes. Each crime has the time, place, victim, weather, weapon and scene type, and the solved crimes also have the killer (1–8). The task is to train models on the solved crimes and name the most likely killer for each unsolved one.

| Split | Use |
|---|---|
| TRAIN | fit the models |
| VAL | compare the models |
| TEST | unsolved crimes, predicted at the end |

**Features:** 8 continuous (`hour_float`, `latitude`, `longitude`, `victim_age`, `temp_c`, `humidity`, `dist_precinct_km`, `pop_density`) and 4 categorical (`weapon_code`, `scene_type`, `weather`, `vic_gender`).

## Approach

1. **Exploratory analysis:** class balance, crime locations, feature distributions.
2. **Gaussian MLE per killer:** mean and covariance of the continuous features for each class.
3. **Generative vs discriminative models:** Gaussian Naive Bayes, compared with Logistic Regression, an MLP (100 → 50 neurons) and an SVM with an RBF kernel. Continuous features are standardised and categorical ones one-hot encoded, inside scikit-learn pipelines.
4. **Unsupervised view:** PCA for visualisation, then k-means with k = 8, to see whether the killers form natural clusters without using the labels.
5. **Final prediction:** the best model is retrained on TRAIN + VAL and predicts the TEST crimes.

## Results

Validation accuracy (958 crimes):

| Model | Accuracy |
|---|---|
| Gaussian Naive Bayes (continuous features only) | 0.833 |
| Logistic Regression | 0.943 |
| MLP (100, 50) | 0.936 |
| **SVM (RBF kernel)** | **0.947** |

The discriminative models clearly beat Naive Bayes, and the SVM came out on top. Location turned out to be the most informative feature, because most killers operate in their own area of the city.

### A closer look: accuracy hides the small classes

Killer 3 committed about half of all crimes, so a model can score high accuracy by doing well on killer 3 alone. Per-class recall of the SVM tells a more honest story:

| Killer | Crimes in VAL | Recall |
|---|---|---|
| 1 | 17 | 0.88 |
| 2 | 62 | 1.00 |
| 3 | 491 | 0.98 |
| 4 | 45 | 0.76 |
| 5 | 48 | 0.63 |
| 6 | 133 | 1.00 |
| 7 | 143 | 0.99 |
| 8 | 19 | 0.47 |

- **Macro F1: 0.86** and **balanced accuracy: 0.84**, against 0.95 accuracy.
- Killer 5 is often mistaken for killer 3, and killer 8 for killers 3 and 5. Their crimes overlap geographically with killer 3's.
- Killer 4 is confused with killer 7, the neighbouring group on the map.

![SVM confusion matrix](figures/confusion_matrix.png)

### Unsupervised: do the killers form natural clusters?

![PCA coloured by true killer and by k-means cluster](figures/pca_kmeans_comparison.png)

k-means finds the geographically isolated killers well (for example killer 6, top right), but it splits the large killer-3 group into several clusters and merges smaller neighbouring groups. So the structure is only partly recoverable without labels. The script measures the match with the Adjusted Rand Index (ARI) and normalised mutual information (NMI).

More figures are in [`figures/`](figures/): class distribution, feature histograms, killer 3's covariance matrix and the PCA scree plot.

## What I would try next

- `class_weight="balanced"` in the SVM and Logistic Regression, to push recall up for killers 5 and 8.
- Tuning `C` and `gamma` with stratified cross-validation instead of a single validation split.
- A location-only baseline, to measure how much the other features actually add.

## How to run

```bash
pip install -r requirements.txt
python who_is_the_killer.py data/your_file.csv
```

The script saves all figures, a `results.csv` with accuracy, balanced accuracy and macro F1 for every model, and the test predictions (`submission.csv`) to `figures/`. You can choose another folder with `--out`.

**Data:** the dataset was provided by the course and is not included in this repository. See [`data/README.md`](data/README.md) for the expected format.

## Repository structure

```
who_is_the_killer.py   full pipeline: EDA, models, PCA/k-means, predictions
requirements.txt       Python dependencies
figures/               generated plots
data/                  put the CSV here (not tracked by git)
```

## Author

Odysseas Tamvakos, Department of Informatics, University of Piraeus

## License

The code is released under the [MIT License](LICENSE). The dataset belongs to the course and is not included.
