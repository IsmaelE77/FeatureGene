from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.linear_model import SGDClassifier
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif, chi2, mutual_info_classif
from joblib import parallel_backend
from sklearn.preprocessing import LabelEncoder
import random
import io
import time
import os
app = Flask(__name__)


# ---------- GA Core ----------
def create_random_chromosome(feature_count, true_ratio=0.3):
    n_true = int(feature_count * true_ratio)
    n_false = feature_count - n_true
    chromosome = [1] * n_true + [0] * n_false
    random.shuffle(chromosome)
    return chromosome



def initialize_population(pop_size, feature_count, true_ratio=0.05):
    return [create_random_chromosome(feature_count, true_ratio) for _ in range(pop_size)]



def evaluate_fitness(chromosome, X_train, X_test, y_train, y_test):
    """
    Evaluate fitness of a chromosome using logistic regression.
    """
    try:
        selected_features = [i for i, gene in enumerate(chromosome) if gene == 1]
        if not selected_features:
            return 0.0

        X_train_selected = X_train[:, selected_features]
        X_test_selected = X_test[:, selected_features]

        if np.isnan(X_train_selected).any() or np.isnan(X_test_selected).any():
            return 0.0

        model = SGDClassifier(
            loss="hinge",
            max_iter=1000,
            tol=1e-3,
            random_state=42
        )

        with parallel_backend("threading", n_jobs=2):
            model.fit(X_train_selected, y_train)

        accuracy = model.score(X_test_selected, y_test)
        return accuracy

    except Exception as error:
        print("An exception occurred:", error)
        return 0.0



def crossover(p1, p2, rate):
    if random.random() > rate:
        return p1[:], p2[:]
    point = random.randint(1, len(p1) - 1)
    return p1[:point] + p2[point:], p2[:point] + p1[point:]


def mutate(chromosome, rate):
    return [1 - g if random.random() < rate else g for g in chromosome]



def load_and_preprocess_csv(csv_content, target_column_idx, id_column_idx=None):
    """
    Loads and preprocesses CSV data for GA feature selection.

    Args:
        csv_content: String containing CSV data.
        target_column_idx: Index of the target column.
        id_column_idx: Optional index of ID column to exclude from features.

    Returns:
        tuple: ((X_train, X_test, y_train, y_test, feature_headers, target_header, df_transformed), meta)
        or (None, error_message) if an error occurs.
    """
    try:
        # Parse CSV data.
        df = pd.read_csv(io.StringIO(csv_content))

        # Remove any empty rows and columns.
        df = df.dropna(how='all')
        df = df.dropna(axis=1, how='all')

        # Validate column indices.
        if target_column_idx >= len(df.columns) or target_column_idx < 0:
            return None, f"Invalid target column index: {target_column_idx}. Dataset has {len(df.columns)} columns (0-{len(df.columns)-1})"

        if id_column_idx is not None and (id_column_idx >= len(df.columns) or id_column_idx < 0):
            return None, f"Invalid ID column index: {id_column_idx}. Dataset has {len(df.columns)} columns (0-{len(df.columns)-1})"

        # Determine which columns are features (exclude ID and target).
        columns_to_exclude = [target_column_idx]
        if id_column_idx is not None:
            columns_to_exclude.append(id_column_idx)

        feature_column_indices = [i for i in range(len(df.columns)) if i not in columns_to_exclude]
        feature_columns = df.columns[feature_column_indices]
        target_column = df.columns[target_column_idx]

        # Extract features and target as DataFrames/Series
        features_df = df[feature_columns].copy()
        target_series = df[target_column].copy()

        # Prepare metadata containers
        categorical_expansions = []
        target_encoding = {"encoded": False, "classes": []}

        # Label encode target if non-numeric or object
        if target_series.dtype == object or not np.issubdtype(target_series.dtype, np.number):
            target_label_encoder = LabelEncoder()
            encoded_target = target_label_encoder.fit_transform(target_series.astype(str))
            target_encoding = {
                "encoded": True,
                "classes": list(map(str, target_label_encoder.classes_)),
            }
            target_series = pd.Series(encoded_target, index=target_series.index, name=target_column)
        else:
            target_series = target_series.astype(float)

        # Detect categorical columns in features
        categorical_feature_columns = [col for col in features_df.columns if features_df[col].dtype == object]
        numeric_feature_columns = [col for col in features_df.columns if col not in categorical_feature_columns]

        # One-Hot encode categorical feature columns (no drop_first for transparency)
        if categorical_feature_columns:
            # Build metadata mapping before encoding
            for feature_name in categorical_feature_columns:
                categories = sorted(map(str, pd.Series(features_df[feature_name].astype(str)).unique()))
                # Column names that will be created by pandas.get_dummies
                created = [f"{feature_name}_{cat}" for cat in categories]
                categorical_expansions.append({
                    "column": feature_name,
                    "categories": categories,
                    "createdColumns": created,
                })
            features_df = pd.get_dummies(features_df, columns=categorical_feature_columns, prefix=categorical_feature_columns, dtype=float)

        # Ensure numeric dtype for all features
        for col in features_df.columns:
            if not np.issubdtype(features_df[col].dtype, np.number):
                features_df[col] = pd.to_numeric(features_df[col], errors='coerce')

        # Handle missing values by dropping rows with NaN across X or y
        combined = pd.concat([features_df, target_series], axis=1)
        combined = combined.dropna(axis=0, how='any')

        if combined.shape[0] < 2:
            return None, "Not enough valid data rows after cleaning"

        # Split back
        features_df = combined.drop(columns=[target_column])
        target_series = combined[target_column]

        features_matrix = features_df.values
        target_array = target_series.values

        # Split data into train/test sets with stratify when possible
        unique_classes, class_counts = np.unique(target_array, return_counts=True)
        if np.all(class_counts >= 2):
            X_train, X_test, y_train, y_test = train_test_split(
                features_matrix, target_array, test_size=0.3, random_state=42, stratify=target_array
            )
        else:
            X_train, X_test, y_train, y_test = train_test_split(
                features_matrix, target_array, test_size=0.3, random_state=42
            )

        feature_headers = list(features_df.columns)
        target_header = target_column

        # Build meta and transformed dataset CSV
        meta = {
            "originalFeatureCount": int(len(feature_columns)),
            "transformedFeatureCount": int(len(feature_headers)),
            "categoricalExpansions": categorical_expansions,
            "targetEncoding": target_encoding,
            "originalColumns": list(df.columns),
        }

        # Build transformed df (features + target) for download convenience
        df_transformed = features_df.copy()
        df_transformed[target_column] = target_series

        # Return core tuple plus meta
        core = (X_train, X_test, y_train, y_test, feature_headers, target_header, df_transformed)
        return (core, meta), None

    except Exception as e:
        return None, f"Error processing CSV: {str(e)}"


def run_ga_feature_selection(
    X_train,
    X_test,
    y_train,
    y_test,
    feature_headers,
    pop_size,
    crossover_rate,
    mutation_rate,
    max_gen,
    convergence_threshold=None,
):
    """
    Execute GA feature selection and return core results.

    Returns dict with:
      - best_fitness, best_chromosome, history, converged, selected_features
    """
    feature_count = len(feature_headers)

    population = initialize_population(pop_size, feature_count)
    best_fitness = 0.0
    best_chromosome = []
    history = []
    prev_best = 0.0
    converged = False

    for gen in range(max_gen):
        fitnesses = [evaluate_fitness(ch, X_train, X_test, y_train, y_test) for ch in population]
        gen_best = max(fitnesses)
        best_idx = fitnesses.index(gen_best)

        if gen_best > best_fitness:
            best_fitness = gen_best
            best_chromosome = population[best_idx][:]

        history.append(best_fitness)

        if gen > 0 and prev_best > 0 and convergence_threshold is not None:
            improvement = abs(best_fitness - prev_best)
            if improvement < convergence_threshold:
                converged = True
                break

        prev_best = best_fitness

        new_pop = []
        while len(new_pop) < pop_size:
            p1, p2 = random.sample(population, 2)
            c1, c2 = crossover(p1, p2, crossover_rate)
            new_pop.append(mutate(c1, mutation_rate))
            if len(new_pop) < pop_size:
                new_pop.append(mutate(c2, mutation_rate))
        population = new_pop

    selected_features = [f for f, g in zip(feature_headers, best_chromosome) if g == 1]
    return {
        "best_fitness": float(best_fitness),
        "best_chromosome": best_chromosome,
        "history": history,
        "converged": bool(converged),
        "selected_features": selected_features,
    }

def run_variance_threshold_selection(
    X_train,
    X_test,
    y_train,
    y_test,
    feature_headers,
    threshold: float,
):
    """
    Execute VarianceThreshold selection and evaluate a classifier on selected features.

    Returns dict with:
      - threshold, accuracy, selected_features, removed_features
    """
    selector = VarianceThreshold(threshold=threshold)
    selector.fit(X_train)

    mask = selector.get_support(indices=True)
    chromosome = [1 if i in mask else 0 for i in range(len(feature_headers))]

    accuracy = evaluate_fitness(chromosome, X_train, X_test, y_train, y_test)

    selected_features = [feature_headers[i] for i in mask]
    removed_features = [f for i, f in enumerate(feature_headers) if i not in set(mask)]

    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy),
        "selected_features": selected_features,
        "removed_features": removed_features,
    }

def run_select_kbest_selection(
    X_train,
    X_test,
    y_train,
    y_test,
    feature_headers,
    k: int,
    score_func_name: str,
):
    """
    Execute SelectKBest selection and evaluate a classifier on selected features.

    Returns dict with:
      - kUsed, scoreFuncUsed, accuracy, selected_features, removed_features
    """
    score_funcs = {
        "f_classif": f_classif,
        "chi2": chi2,
        "mutual_info": mutual_info_classif,
    }
    score_func = score_funcs.get(score_func_name, f_classif)

    # Ensure k is within valid range
    n_features = len(feature_headers)
    if k is None or k <= 0:
        k = max(1, n_features // 2)
    k = min(k, n_features)

    selector = SelectKBest(score_func=score_func, k=k)
    selector.fit(X_train, y_train)

    mask = selector.get_support(indices=True)
    chromosome = [1 if i in mask else 0 for i in range(n_features)]

    accuracy = evaluate_fitness(chromosome, X_train, X_test, y_train, y_test)

    selected_features = [feature_headers[i] for i in mask]
    removed_features = [f for i, f in enumerate(feature_headers) if i not in set(mask)]

    return {
        "kUsed": int(k),
        "scoreFuncUsed": score_func_name if score_func_name in score_funcs else "f_classif",
        "accuracy": float(accuracy),
        "selected_features": selected_features,
        "removed_features": removed_features,
    }

# ---------- Flask routes ----------
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/guide")
def guide():
    return render_template("guide.html")


@app.route("/list_datasets", methods=["GET"])
def list_datasets():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    ds_dir = os.path.join(base_dir, "dataset")
    try:
        if not os.path.isdir(ds_dir):
            return jsonify({"datasets": []})
        files = [f for f in os.listdir(ds_dir) if f.lower().endswith(".csv")]
        return jsonify({"datasets": files})
    except Exception as e:
        return jsonify({"error": f"Failed to list datasets: {str(e)}"})

@app.route("/get_dataset", methods=["POST"])
def get_dataset():
    data = request.json or {}
    name = data.get("name")
    if not name:
        return jsonify({"error": "Dataset name is required"})

    if os.path.sep in name or os.path.altsep and os.path.altsep in name:
        return jsonify({"error": "Invalid dataset name"})
    if not name.lower().endswith(".csv"):
        return jsonify({"error": "Only CSV datasets are supported"})

    base_dir = os.path.dirname(os.path.abspath(__file__))
    ds_dir = os.path.join(base_dir, "dataset")
    file_path = os.path.join(ds_dir, name)

    if not os.path.isfile(file_path):
        return jsonify({"error": f"Dataset not found: {name}"})

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            csv_content = f.read()
        return jsonify({"name": name, "csvData": csv_content})
    except Exception as e:
        return jsonify({"error": f"Failed to read dataset: {str(e)}"})

@app.route("/run_ga", methods=["POST"])
def run_ga():
    """
    Main endpoint to run the Genetic Algorithm for feature selection.

    This function is already implemented for you. Study it to understand
    how all the GA components work together.
    """
    data = request.json
    pop_size = int(data.get("popSize", 30))
    crossover_rate = float(data.get("crossRate", 0.7))
    mutation_rate = float(data.get("mutRate", 0.1))
    max_gen = int(data.get("maxGen", 20))
    convergence_threshold = data.get("convergenceThreshold")
    if convergence_threshold is not None:
        convergence_threshold = float(data.get("convergenceThreshold"))
    csv_content = data.get("csvData", "")

    # Get user-specified column indices
    id_column_idx = data.get("idColumn")  # Can be None if no ID column
    target_column_idx = data.get("targetColumn")  # Required

    if target_column_idx is None:
        return jsonify({"error": "Target column must be specified"})

    # --- Load CSV ---
    processed, error = load_and_preprocess_csv(csv_content, target_column_idx, id_column_idx)
    if error:
        return jsonify({"error": error})

    (result, meta) = processed
    X_train, X_test, y_train, y_test, feature_headers, target_header, df = result
    feature_count = len(feature_headers)
    
    start_exec = time.perf_counter()
    ga = run_ga_feature_selection(
        X_train,
        X_test,
        y_train,
        y_test,
        feature_headers,
        pop_size,
        crossover_rate,
        mutation_rate,
        max_gen,
        convergence_threshold,
    )
    exec_time = time.perf_counter() - start_exec

    # Prepare response
    response = {
        "bestFitness": round(ga["best_fitness"], 4),
        "bestChromosome": ga["best_chromosome"],
        "selectedFeatures": ga["selected_features"],
        "history": ga["history"],
        "featuresCount": feature_count,
        "rows": len(df),
        "target": target_header,
        "generations": len(ga["history"]),
        "converged": ga["converged"],
        "execTimeSeconds": round(exec_time, 4),
        "featureEngineering": meta,
        "transformedCsv": df.to_csv(index=False),
    }

    # Add ID column name if specified
    if id_column_idx is not None:
        try:
            response["idColumn"] = (meta.get("originalColumns") or [])[int(id_column_idx)]
        except Exception:
            pass

    return jsonify(response)

@app.route("/run_variance_threshold", methods=["POST"])
def run_variance_threshold():
    data = request.json
    threshold = float(data.get("threshold", 0.0))
    csv_content = data.get("csvData", "")
    id_column_idx = data.get("idColumn")
    target_column_idx = data.get("targetColumn")

    if target_column_idx is None:
        return jsonify({"error": "Target column must be specified"})

    processed, error = load_and_preprocess_csv(csv_content, target_column_idx, id_column_idx)
    if error:
        return jsonify({"error": error})

    (result, meta) = processed
    X_train, X_test, y_train, y_test, feature_headers, target_header, df = result

    start_exec = time.perf_counter()
    vt = run_variance_threshold_selection(
        X_train,
        X_test,
        y_train,
        y_test,
        feature_headers,
        threshold,
    )
    exec_time = time.perf_counter() - start_exec

    response = {
        "thresholdUsed": vt["threshold"],
        "accuracy": round(vt["accuracy"], 4),
        "selectedFeatures": vt["selected_features"],
        "removedFeatures": vt["removed_features"],
        "numFeaturesSelected": len(vt["selected_features"]),
        "numFeaturesTotal": len(feature_headers),
        "rows": len(df),
        "target": target_header,
        "execTimeSeconds": round(exec_time, 4),
        "featureEngineering": meta,
        "transformedCsv": df.to_csv(index=False),
    }
    if id_column_idx is not None:
        try:
            response["idColumn"] = (meta.get("originalColumns") or [])[int(id_column_idx)]
        except Exception:
            pass

    return jsonify(response)

@app.route("/run_comparison", methods=["POST"])
def run_comparison():
    data = request.json
    pop_size = int(data.get("popSize", 30))
    crossover_rate = float(data.get("crossRate", 0.7))
    mutation_rate = float(data.get("mutRate", 0.1))
    max_gen = int(data.get("maxGen", 20))
    convergence_threshold = data.get("convergenceThreshold")
    if convergence_threshold is not None:
        convergence_threshold = float(convergence_threshold)
    vt_threshold = float(data.get("threshold", 0.0))
    # SelectKBest params
    kb_k_raw = data.get("k")
    try:
        kb_k = int(kb_k_raw) if kb_k_raw is not None and kb_k_raw != "" else None
    except Exception:
        kb_k = None
    kb_score_func = data.get("scoreFunc", "f_classif")
    csv_content = data.get("csvData", "")
    id_column_idx = data.get("idColumn")
    target_column_idx = data.get("targetColumn")

    if target_column_idx is None:
        return jsonify({"error": "Target column must be specified"})

    processed, error = load_and_preprocess_csv(csv_content, target_column_idx, id_column_idx)
    if error:
        return jsonify({"error": error})

    (result, meta) = processed
    X_train, X_test, y_train, y_test, feature_headers, target_header, df = result
    feature_count = len(feature_headers)

    ga_start = time.perf_counter()
    ga = run_ga_feature_selection(
        X_train,
        X_test,
        y_train,
        y_test,
        feature_headers,
        pop_size,
        crossover_rate,
        mutation_rate,
        max_gen,
        convergence_threshold,
    )
    ga_exec = time.perf_counter() - ga_start

    vt_start = time.perf_counter()
    vt = run_variance_threshold_selection(
        X_train,
        X_test,
        y_train,
        y_test,
        feature_headers,
        vt_threshold,
    )
    vt_exec = time.perf_counter() - vt_start

    kb_start = time.perf_counter()
    kb = run_select_kbest_selection(
        X_train,
        X_test,
        y_train,
        y_test,
        feature_headers,
        kb_k,
        kb_score_func,
    )
    kb_exec = time.perf_counter() - kb_start

    response = {
        "dataset": {
            "target": target_header,
            "numFeaturesTotal": feature_count,
            "rows": len(df),
            "featureEngineering": meta,
        },
        "ga": {
            "bestFitness": round(ga["best_fitness"], 4),
            "history": ga["history"],
            "selectedFeatures": ga["selected_features"],
            "numFeaturesSelected": len(ga["selected_features"]),
            "generations": len(ga["history"]),
            "converged": ga["converged"],
            "accuracy": round(ga["best_fitness"], 4),
            "execTimeSeconds": round(ga_exec, 4),
            "transformedCsv": df.to_csv(index=False),
        },
        "varianceThreshold": {
            "thresholdUsed": vt["threshold"],
            "accuracy": round(vt["accuracy"], 4),
            "selectedFeatures": vt["selected_features"],
            "removedFeatures": vt["removed_features"],
            "numFeaturesSelected": len(vt["selected_features"]),
            "execTimeSeconds": round(vt_exec, 4),
        },
        "selectKBest": {
            "kUsed": kb["kUsed"],
            "scoreFuncUsed": kb["scoreFuncUsed"],
            "accuracy": round(kb["accuracy"], 4),
            "selectedFeatures": kb["selected_features"],
            "removedFeatures": kb["removed_features"],
            "numFeaturesSelected": len(kb["selected_features"]),
            "execTimeSeconds": round(kb_exec, 4),
        },
    }
    if id_column_idx is not None:
        try:
            response["dataset"]["idColumn"] = (meta.get("originalColumns") or [])[int(id_column_idx)]
        except Exception:
            pass

    return jsonify(response)


@app.route("/run_select_kbest", methods=["POST"])
def run_select_kbest():
    data = request.json
    k = data.get("k")
    try:
        k = int(k) if k is not None and k != "" else None
    except Exception:
        k = None
    score_func = data.get("scoreFunc", "f_classif")
    csv_content = data.get("csvData", "")
    id_column_idx = data.get("idColumn")
    target_column_idx = data.get("targetColumn")

    if target_column_idx is None:
        return jsonify({"error": "Target column must be specified"})

    processed, error = load_and_preprocess_csv(csv_content, target_column_idx, id_column_idx)
    if error:
        return jsonify({"error": error})

    (result, meta) = processed
    X_train, X_test, y_train, y_test, feature_headers, target_header, df = result

    start_exec = time.perf_counter()
    kb = run_select_kbest_selection(
        X_train,
        X_test,
        y_train,
        y_test,
        feature_headers,
        k,
        score_func,
    )
    exec_time = time.perf_counter() - start_exec

    response = {
        "kUsed": kb["kUsed"],
        "scoreFuncUsed": kb["scoreFuncUsed"],
        "accuracy": round(kb["accuracy"], 4),
        "selectedFeatures": kb["selected_features"],
        "removedFeatures": kb["removed_features"],
        "numFeaturesSelected": len(kb["selected_features"]),
        "numFeaturesTotal": len(feature_headers),
        "rows": len(df),
        "target": target_header,
        "execTimeSeconds": round(exec_time, 4),
        "featureEngineering": meta,
        "transformedCsv": df.to_csv(index=False),
    }
    if id_column_idx is not None:
        try:
            response["idColumn"] = (meta.get("originalColumns") or [])[int(id_column_idx)]
        except Exception:
            pass

    return jsonify(response)


if __name__ == "__main__":
    app.run(debug=True)
