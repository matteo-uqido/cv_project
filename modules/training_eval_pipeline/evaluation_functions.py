import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy.stats import pearsonr

def evaluate_model(model, test_generator):
    predicted_counts = []
    true_counts = []

    for i in range(len(test_generator)):
        batch_images, batch_labels = test_generator[i]
        predicted_counts.extend(get_model_predictions(model, batch_images))
        true_counts.extend(np.round([np.sum(label) for label in batch_labels]))
    
    return np.array(true_counts), np.array(predicted_counts)

def calculate_metrics(true_counts, predicted_counts):
    mae = mean_absolute_error(true_counts, predicted_counts)
    mse = mean_squared_error(true_counts, predicted_counts)
    pearson_corr = pearsonr(true_counts, predicted_counts)[0]
    rmse_value = np.sqrt(mse)
    
    return {
        "MAE": mae,
        "MSE": mse,
        "Pearson Correlation": pearson_corr,
        "RMSE": rmse_value
    }

def print_metrics(metrics):
    print(f"MAE: {metrics['MAE']:.2f}")
    print(f"MSE: {metrics['MSE']:.2f}")
    print(f"Pearson Correlation: {metrics['Pearson Correlation']:.2f}")
    print(f"RMSE: {metrics['RMSE']:.2f}")

def get_model_predictions(model, X):

    predictions = model.predict(X, batch_size=32, verbose=0)

    # If output is a single count per image
    if predictions.ndim == 1 or predictions.shape[1] == 1:
        return np.round(predictions.flatten())

    # If output is a density map
    counts = []
    for i in range(len(predictions)):
        count = np.round(np.sum(predictions[i]))
        counts.append(count)

    return counts

