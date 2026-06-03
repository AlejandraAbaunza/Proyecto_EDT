import pandas as pd

from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    cohen_kappa_score
)

from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

# --------------------------------------------------
# 1. Cargar dataset
# --------------------------------------------------
df = pd.read_csv("datasetV2.tsv", sep="\t")

# --------------------------------------------------
# 2. Reagrupar 25 clases -> 5 macroclases
# --------------------------------------------------
mapping = {
    1:1, 6:1, 7:1, 8:1, 9:1,
    2:2, 10:2, 11:2, 12:2, 13:2,
    3:3, 14:3, 15:3, 16:3, 17:3,
    4:4, 18:4, 19:4, 20:4, 21:4,
    5:5, 22:5, 23:5, 24:5, 25:5
}

df["class5"] = df["class_id"].map(mapping)

# --------------------------------------------------
# 3. Información del dataset
# --------------------------------------------------
print("\nClases originales:")
print(sorted(df["class_id"].unique()))

print("\nClases reagrupadas:")
print(sorted(df["class5"].unique()))

print("\nFrecuencia por clase:")
print(df["class5"].value_counts().sort_index())

print("\nROIs por clase:")
print(df.groupby("class5")["roi_id"].unique())

# --------------------------------------------------
# 4. Bandas
# --------------------------------------------------
bandas = [
    "B2",
    "B3",
    "B4",
    "B5",
    "B6",
    "B7",
    "B8",
    "B11"
]

# --------------------------------------------------
# 5. Separación por ROI
# --------------------------------------------------
test_rois = []

for clase in sorted(df["class5"].unique()):

    rois = sorted(
        df[df["class5"] == clase]["roi_id"].unique()
    )

    # último ROI para prueba
    test_rois.append(rois[-1])

print("\nROIs usados para TEST:")
print(test_rois)

train_df = df[~df["roi_id"].isin(test_rois)]
test_df = df[df["roi_id"].isin(test_rois)]

print("\nMuestras entrenamiento:", len(train_df))
print("Muestras prueba:", len(test_df))

# --------------------------------------------------
# 6. Features y etiquetas
# --------------------------------------------------
X_train = train_df[bandas]
y_train = train_df["class5"]

X_test = test_df[bandas]
y_test = test_df["class5"]

# --------------------------------------------------
# 7. Escalamiento
# --------------------------------------------------
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# --------------------------------------------------
# 8. Modelos
# --------------------------------------------------
models = {

    "Decision Tree":
        DecisionTreeClassifier(
            random_state=42,
            max_depth=5,
            min_samples_leaf=5
        ),

    "SVM":
        SVC(
            kernel="rbf",
            C=10,
            gamma="scale"
        ),

    "ANN (MLP)":
        MLPClassifier(
            hidden_layer_sizes=(100,),
            max_iter=3000,
            random_state=42
        ),

    "KNN":
        KNeighborsClassifier(
            n_neighbors=5
        ),

    "Naive Bayes":
        GaussianNB()
}

# --------------------------------------------------
# 9. Evaluación
# --------------------------------------------------
results = []

for name, model in models.items():

    print("\n" + "="*60)
    print("MODELO:", name)

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    oa = accuracy_score(y_test, y_pred)

    kappa = cohen_kappa_score(
        y_test,
        y_pred
    )

    precision = precision_score(
        y_test,
        y_pred,
        average="weighted",
        zero_division=0
    )

    recall = recall_score(
        y_test,
        y_pred,
        average="weighted"
    )

    f1 = f1_score(
        y_test,
        y_pred,
        average="weighted"
    )

    print("Accuracy :", round(oa,4))
    print("Kappa    :", round(kappa,4))
    print("Precision:", round(precision,4))
    print("Recall   :", round(recall,4))
    print("F1       :", round(f1,4))

    labels = sorted(df["class5"].unique())

    cm = confusion_matrix(
        y_test,
        y_pred,
        labels=labels
    )

    print("\nMatriz de confusión:")
    print(cm)

    results.append([
        name,
        oa,
        kappa,
        precision,
        recall,
        f1
    ])

# --------------------------------------------------
# 10. Tabla final
# --------------------------------------------------
results_df = pd.DataFrame(
    results,
    columns=[
        "Model",
        "OA",
        "Kappa",
        "Precision",
        "Recall",
        "F1"
    ]
)

results_df = results_df.sort_values(
    by="F1",
    ascending=False
)

print("\n")
print("="*60)
print("RESULTADOS FINALES")
print("="*60)

print(results_df)

# --------------------------------------------------
# 11. Firmas espectrales
# --------------------------------------------------
print("\nPromedio espectral por clase:")

print(
    df.groupby("class5")[bandas].mean()
)

#print(df.groupby("class5")["roi_id"].unique())

# --------------------------------------------------
# 12. Guardar resultados
# --------------------------------------------------
#results_df.to_csv(
#    "resultados_modelos.csv",
#    index=False
#)