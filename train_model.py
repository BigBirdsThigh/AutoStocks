import os
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import accuracy_score, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
from itertools import product
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold, train_test_split, GridSearchCV
from dotenv import load_dotenv

load_dotenv()

# ================== Configuration ==================
DATA_DIR = os.getenv("DATA_DIR")

if not DATA_DIR:
    raise ValueError("DATA_DIR environment variable is not set.")

PREDICTION_DAYS = 126 # predict price 6 months ahead of current date (e.g. 126 rows down from current row)
PARAM_GRID = {
    'max_depth': [3, 5, 7, 9],  # Depth values to try
    'learning_rate': [0.01, 0.05, 0.1, 0.2],  # Learning rates to try
    'n_estimators': [100, 200, 500],  # Iterationsto try
    'subsample': [0.6, 0.8, 1.0],  # How much of the data that the tree sees
    'colsample_bytree': [0.6, 0.8, 1.0]  # Feature sampling
}

N_JOBS = 5
CHUNK_SIZE = 15 # processing in N chunks to avoid bulk resource consumption and avoid freezing my device
HDF5_FILE = "precomputed_features.h5" # file to store precomputed features

# ================== Calculate the Indicators ==================

def compute_rsi(series, period=14):
    delta = series.diff(1)
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def compute_macd(series, slow=26, fast=12, signal=9):
    fast_ema = series.ewm(span=fast, adjust=False).mean()
    slow_ema = series.ewm(span=slow, adjust=False).mean()
    macd = fast_ema - slow_ema
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    return macd - signal_line

def compute_atr(df, period=14):
    high_low = df['high'] - df['low']
    high_close = abs(df['high'] - df['adj_close'].shift(1))
    low_close = abs(df['low'] - df['adj_close'].shift(1))
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return true_range.rolling(period).mean()


# ================== Helpers ==================

def split_train_validation(X_train, y_train, val_size=0.1):
    """Split the training data into a smaller training set and a validation set."""
    return train_test_split(X_train, y_train, test_size=val_size, stratify=y_train, shuffle=True)


def get_param_combinations(param_grid):
    """Get all parameter combinations from the grid."""
    keys, values = zip(*param_grid.items())
    return [dict(zip(keys, combination)) for combination in product(*values)]


# ================== Perform the Grid Search ==================
def chunked_grid_search(X_train, y_train, param_combinations, chunk_size, cv=3, scoring='accuracy', verbose=0):
    """Perform a grid search in chunks."""
    results = []
    # Split the parameter combinations into chunks
    chunks = [param_combinations[i:i + chunk_size] for i in range(0, len(param_combinations), chunk_size)]
    with tqdm(total=len(chunks), desc="Grid Search Chunks") as pbar:
        for chunk_idx, chunk in enumerate(chunks):
            # Convert the chunk into a parameter grid
            chunk_param_grid = {key: list(set([comb[key] for comb in chunk])) for key in chunk[0]}
            model = xgb.XGBClassifier(objective='multi:softmax', num_class=3, n_jobs=N_JOBS, tree_method="hist")
            grid_search = GridSearchCV(estimator=model, param_grid=chunk_param_grid, cv=cv, scoring=scoring, verbose=verbose, n_jobs=N_JOBS)
            grid_search.fit(X_train, y_train)
            # Store the best score and params from this chunk
            results.append({
                "best_score": grid_search.best_score_,
                "best_params": grid_search.best_params_
            })
            # Save intermediate results
            with open("intermediate_results.txt", "a") as f:
                f.write(f"Chunk {chunk_idx}, Best Score: {grid_search.best_score_}, Best Params: {grid_search.best_params_}\n")
            # Update the progress bar
            pbar.update(1)
    # Return the best result overall
    return max(results, key=lambda x: x['best_score'])

def initialise_chunked_search(X_train, y_train, X_test, y_test, param_grid, chunk_size, important_features=None):
    """
    Train a model using chunked grid search.
    If important_features is given, it uses only those features.
    """
    X_train_used = X_train[important_features] if important_features is not None else X_train
    X_test_used = X_test[important_features] if important_features is not None else X_test

    # Get all parameter combinations
    param_combinations = get_param_combinations(param_grid)

    # Perform chunked grid search
    best_chunk_result = chunked_grid_search(X_train_used, y_train, param_combinations, chunk_size)

    # Initialize the model with early_stopping_rounds
    best_model = xgb.XGBClassifier(
        objective='multi:softmax',
        num_class=3,
        n_jobs=N_JOBS,
        tree_method="hist",
        early_stopping_rounds=10,
        **best_chunk_result["best_params"]
    )

    # Split the training data for early stopping, we use this data to check if the model has improved at all for n consecutive runs
    # if not then we say the model will not improve much further and we end the run early
    X_train_split, X_val_split, y_train_split, y_val_split = split_train_validation(X_train_used, y_train)

    # Fit the model
    best_model.fit(
        X_train_split,
        y_train_split,
        eval_set=[(X_val_split, y_val_split)],
        verbose=0
    )

    # Predict and evaluate
    y_pred = best_model.predict(X_test_used)
    accuracy = accuracy_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    print(f"\nBest Parameters: {best_chunk_result['best_params']}")
    print(f"Accuracy: {accuracy}")
    print(f"Mean Absolute Error: {mae}")
    return accuracy, mae, best_chunk_result["best_params"]

# ================== Main Execution ==================
print("\n[INFO] Loading stock data from:", DATA_DIR)

# Precompute features and save to HDF5
if os.path.exists(HDF5_FILE):
    print("Loading precomputed features...")
    df_combined = pd.read_hdf(HDF5_FILE, key="features")
else:
    print("Processing files...")
    all_data = []
    stock_files = [f for f in os.listdir(DATA_DIR) if f.endswith('.csv')]
    # stock_files = stock_files[:4000] # remove this line 
    for stock_file in tqdm(stock_files, desc="Processing files"):
        file_path = os.path.join(DATA_DIR, stock_file)
        df = pd.read_csv(file_path)
        df['Date'] = pd.to_datetime(df['Date'])
        df.sort_values('Date', inplace=True)

        df.rename(columns={'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close', 
                           'Adj Close': 'adj_close', 'Volume': 'volume'}, inplace=True)

        # Compute indicators
        df['rsi'] = compute_rsi(df['adj_close'])
        df['macd'] = compute_macd(df['adj_close'])
        df['sma200'] = df['adj_close'].rolling(window=200).mean()
        df['atr'] = compute_atr(df)

        # Compute future return and target variable
        df['future_return'] = (df['adj_close'].shift(-PREDICTION_DAYS) - df['adj_close']) / df['adj_close']
        df['target'] = np.where(df['future_return'] > 0.15, 2, 
                                np.where(df['future_return'] < -0.10, 0, 1))
        df.dropna(inplace=True)
        all_data.append(df)
    df_combined = pd.concat(all_data, ignore_index=True)
    print("Saving precomputed features...")
    df_combined.to_hdf(HDF5_FILE, key="features", mode="w")

# Prepare feature matrix and labels
required_features = ['open', 'high', 'low', 'adj_close', 'volume', 'rsi', 'macd', 'sma200', 'atr']
X = df_combined[required_features].astype(np.float32)  # Convert to float32 for lower memory usage
y = df_combined['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, shuffle=True)

print("\n=== Starting Runs ===")

# NOTE: I have tried to run these in parallel but the resource usage causes my machine to freeze when running more than one at a time

results = []
# Run 1: All Features, No Normalization
print("Run 1: All Features, No Normalization")
results.append(initialise_chunked_search(X_train, y_train, X_test, y_test, PARAM_GRID, CHUNK_SIZE))

# Run 2: Feature Selection
print("Run 2: Feature Selection")
def run_feature_selection(X_train, y_train, param_grid, chunk_size, feature_threshold=0.01, n_splits=5):
    """
        Performs a grid search, trains the best model using cross-validation, 
        and returns important features.
    """
    print("\n[INFO] Performing grid search to find the best parameters for feature selection...")
    best_chunk_result = chunked_grid_search(X_train, y_train, get_param_combinations(param_grid), chunk_size)
    best_params = best_chunk_result["best_params"]

    print(f"\n[INFO] Best parameters found: {best_params}")
    
    # cross-validation helps see how well the model works on different parts of the data
    # by splitting the data into k folds and using each fold as a validation set we can
    # check if the model consistently performs well
    # if certain features consistently show importance accross all folds
    # they are more likely to be genuinely useful predictors and not be noisy

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    all_importances = np.zeros(X_train.shape[1])

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
        print(f"\n[INFO] Running fold {fold_idx + 1}/{n_splits}")

        # train_idx: indices for train set
        # val_idx: indices for validation set
        X_tr, X_val = X_train[train_idx], X_train[val_idx]
        y_tr, y_val = y_train[train_idx], y_train[val_idx]

        # train the model using the current folds training data
        model = xgb.XGBClassifier(**best_params, objective='multi:softmax', num_class=3, n_jobs=N_JOBS, tree_method="hist")
        model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], early_stopping_rounds=10, verbose=0)

        # after training, the model will have feature importance based on this fold        
        all_importances += model.feature_importances_ # add this to the cumulative array

    # once all folds are process, average the importances, providing a more reliable measure of
    # feature relevance
    mean_importances = all_importances / n_splits

    # loop through calculated averages and compare them against threshold
    # if a feature's importance is > threshold, it is added to the important features
    important_features = [required_features[i] for i in range(len(mean_importances)) if mean_importances[i] > feature_threshold]
    
    print("[INFO] Average feature importances across folds:")
    for feature, importance in zip(required_features, mean_importances):
        print(f"{feature}: {importance:.4f}")

    print(f"\n[INFO] Selected important features (threshold={feature_threshold}): {important_features}")
    return important_features

important_features = run_feature_selection(X_train, y_train, X_test, y_test, PARAM_GRID, CHUNK_SIZE)

# Use the important features in the next step
# NOTE: it might be a good idea to use dimensionality reduction here too after feature reduction (or before?), will toy with this after the 1st run of the code
results.append(initialise_chunked_search(X_train, y_train, X_test, y_test, PARAM_GRID, CHUNK_SIZE, important_features))

# Run 3: normalise All Features
print("Run 3: normalise All Features")
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
results.append(initialise_chunked_search(X_train_scaled, y_train, X_test_scaled, y_test, PARAM_GRID, CHUNK_SIZE))

# Run 4: Remove Low-Importance Features + normalise
print("Run 4: Remove Low-Importance Features + normalise")
X_train_filtered = X_train[important_features]
X_test_filtered = X_test[important_features]
X_train_filtered_scaled = scaler.fit_transform(X_train_filtered)
X_test_filtered_scaled = scaler.transform(X_test_filtered)
results.append(initialise_chunked_search(X_train_filtered_scaled, y_train, X_test_filtered_scaled, y_test, PARAM_GRID, CHUNK_SIZE))

# Determine the best run
best_run_idx = np.argmax([res[0] for res in results]) # Best accuracy
print(f"\nBest run is Run {best_run_idx + 1} with Accuracy: {results[best_run_idx][0]}")

# save the best model
best_params = results[best_run_idx][2]
final_model = xgb.XGBClassifier(**best_params, objective='multi:softmax', num_class=3, n_jobs=N_JOBS, tree_method="hist")
final_model.fit(X_train, y_train)
final_model.save_model("best_model.json")
