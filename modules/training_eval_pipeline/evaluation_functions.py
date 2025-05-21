import numpy as np
from scipy.ndimage import gaussian_filter
from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy.stats import pearsonr

def evaluate_model(model, test_generator):
    predicted_counts = []
    true_counts = []

    for i in range(test_generator.__len__()):
        batch_images, batch_labels = test_generator[i]
        predicted_counts.append(full_eval(batch_images, model, smooth=False))
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

def full_eval(x, model, smooth=False):
    batch_size = x.shape[0]
    counts_batch = np.zeros(batch_size)

    for b in range(batch_size):
        predicted_count = model.predict(x[b:b+1])[0]

        if smooth:
            density_map = model.predict(x[b:b+1])[0]
            density_map = gaussian_filter(density_map, 2, mode='constant')
            predicted_count = np.sum(density_map)

        counts_batch[b] = predicted_count

    return counts_batch


