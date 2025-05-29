import numpy as np
from scipy.ndimage import gaussian_filter
from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy.stats import pearsonr

def evaluate_model(model, test_generator):
    predicted_counts = []
    true_counts = []

    for i in range(test_generator.__len__()):
        batch_images, batch_labels = test_generator[i]
        predicted_counts.append(get_model_predictions(model, batch_images, smooth=False))
        true_counts.append(batch_labels)
    
    return np.concatenate(true_counts), np.concatenate(predicted_counts)

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

def get_model_predictions(model, X, smooth=False, sigma=2):

    predictions = model.predict(X, batch_size=32, verbose=0)

    # If output is a single count per image
    if predictions.ndim == 1 or predictions.shape[1] == 1:
        return predictions.flatten()

    # If output is a density map
    counts = []
    for i in range(len(predictions)):
        density_map = predictions[i]
        if smooth:
            density_map = gaussian_filter(density_map, sigma=sigma, mode='constant')
        count = np.sum(density_map)
        counts.append(count)

    return np.array(counts)

