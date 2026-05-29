import pandas as pd

from sklearn.model_selection import (
    train_test_split,
    cross_val_score,
    StratifiedKFold
)

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

# -------------------------
# 1. Cargar dataset
# -------------------------
df = pd.read_csv("dataset.tsv", sep="\t")

# -------------------------
# 2. Features y target
# -------------------------
X = df[
    [
        "B2",
        "B3",
        "B4",
        "B5",
        "B6",
        "B7",
        "B8", 
        "B11"
    ]
]

y = df["class_id"]

# -------------------------
# 3. Escalamiento
# -------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# -------------------------
# 4. Split
# -------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# -------------------------
# 5. Modelos
# -------------------------
models = {
    "Decision Tree":
        DecisionTreeClassifier(random_state=42),

    "SVM":
        SVC(kernel="rbf", C=1),

    "ANN (MLP)":
        MLPClassifier(
            hidden_layer_sizes=(100,),
            max_iter=1000,
            random_state=42
        ),

    "KNN":
        KNeighborsClassifier(n_neighbors=5),

    "Naive Bayes":
        GaussianNB()
}

# -------------------------
# 6. Evaluación
# -------------------------
results = []

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

for name, model in models.items():

    print("\n" + "=" * 60)
    print(name)

    # Entrenamiento
    model.fit(X_train, y_train)

    # Predicción
    y_pred = model.predict(X_test)

    # Métricas
    oa = accuracy_score(y_test, y_pred)
    kappa = cohen_kappa_score(y_test, y_pred)

    precision = precision_score(
        y_test,
        y_pred,
        average="weighted"
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

    # Cross Validation
    cv_scores = cross_val_score(
        model,
        X_scaled,
        y,
        cv=cv,
        scoring="f1_weighted"
    )

    cv_mean = cv_scores.mean()
    cv_std = cv_scores.std()

    print("Accuracy:", round(oa, 4))
    print("Kappa:", round(kappa, 4))
    print("Precision:", round(precision, 4))
    print("Recall:", round(recall, 4))
    print("F1:", round(f1, 4))

    #print("CV F1 Mean:", round(cv_mean, 4))
    #print("CV F1 Std:", round(cv_std, 4))

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    results.append([
        name,
        oa,
        kappa,
        precision,
        recall,
        f1#,
        #cv_mean,
        #cv_std
    ])

# -------------------------
# 7. Tabla final
# -------------------------
results_df = pd.DataFrame(
    results,
    columns=[
        "Model",
        "OA",
        "Kappa",
        "Precision",
        "Recall",
        "F1"#,
        #"CV_F1_Mean",
        #"CV_F1_Std"
    ]
)

results_df = results_df.sort_values(
    #by="CV_F1_Mean",
    by="F1",
    ascending=False
)

print("\n")
print(results_df)

""""
results_df.to_csv(
    "model_comparison.csv",
    index=False
)
"""
#print(df.groupby("class_id")[["B2","B3","B4","B5","B6","B7","B8","B11"]].mean())