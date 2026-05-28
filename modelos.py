import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    f1_score,
    classification_report,
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

print("Columnas:", df.columns)

X = df.drop(columns=["Class"])
y = df["Class"]

# -------------------------
# 2. Split train/test
# -------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y,
    test_size=0.3,
    random_state=42
)

# -------------------------
# 3. Definir modelos
# -------------------------
models = {
    "Decision Tree": DecisionTreeClassifier(random_state=42),

    "SVM": SVC(kernel="rbf"),

    "ANN (MLP)": MLPClassifier(
        hidden_layer_sizes=(100,),
        max_iter=500,
        random_state=42
    ),

    "KNN": KNeighborsClassifier(n_neighbors=5),

    "Naive Bayes": GaussianNB()
}

# -------------------------
# 4. Evaluación
# -------------------------
results = []

for name, model in models.items():

    print("\n============================")
    print("Modelo:", name)

    # Train
    model.fit(X_train, y_train)

    # Predict
    y_pred = model.predict(X_test)

    # Metrics
    cm = confusion_matrix(y_test, y_pred)
    oa = accuracy_score(y_test, y_pred)
    kappa = cohen_kappa_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="weighted")

    print("Confusion Matrix:\n", cm)
    print("Overall Accuracy:", oa)
    print("Kappa:", kappa)
    print("F1-score:", f1)

    results.append([name, oa, kappa, f1])

# -------------------------
# 5. Tabla comparativa final
# -------------------------
results_df = pd.DataFrame(
    results,
    columns=["Model", "OA", "Kappa", "F1-score"]
)

print("\n===== COMPARISON TABLE =====")
print(results_df)