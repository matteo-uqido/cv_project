import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
import random
from paths import GT_PATH, FRAMES_PATH, DATA_PATH
from modules.data_utils.load_and_preprocess_data import load_mall_dataset

def plot_random_samples_with_annotations(random_seed=42):
    # Set seed for reproducibility
    random.seed(random_seed)
    np.random.seed(random_seed)

    # Load ground truth annotations
    gt_data = loadmat(GT_PATH)
    head_annotations_per_frame = gt_data['frame'][0]

    # Load dataset dataframe and image paths
    dataframe = load_mall_dataset(DATA_PATH)
    image_paths = np.array([f"{FRAMES_PATH}/{name}" for name in dataframe['image_name']])

    # Sample 3 random unique indices
    sample_indices = random.sample(range(len(image_paths)), 3)

    # Plot the selected random samples
    plt.figure(figsize=(15, 5))
    for i, idx in enumerate(sample_indices):
        image = plt.imread(image_paths[idx])
        head_coords = head_annotations_per_frame[idx][0][0][0]
        head_count = len(head_coords)

        plt.subplot(1, 3, i + 1)
        plt.imshow(image)
        plt.scatter(head_coords[:, 0], head_coords[:, 1], marker='x', color='red')
        plt.title(f"Frame {idx} - Count: {head_count}")
        plt.axis('off')

    plt.suptitle("Random Mall Dataset Samples with Head Annotations")
    plt.tight_layout()
    plt.show()


def plot_training_history(history):
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(loss) + 1)

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, loss, 'b-', label='Training Loss')
    plt.plot(epochs, val_loss, 'r-', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.yscale('log')
    plt.show()


def plot_headcount_frequency_histogram():
    # Load ground truth data
    mall_gt = loadmat(GT_PATH)
    mall_head_positions_gt = mall_gt['frame'][0]
    print(f'Total number of images: {len(mall_head_positions_gt)}')
    
    # Helper function to plot histogram with median line
    def plot_histogram_with_median(data):
        median = np.median(data)
        plt.hist(data, bins=20)
        plt.axvline(median, color='r', linestyle='--')
        plt.text(1.05 * median, plt.ylim()[1] * 0.9, 'Median', color='r')
        plt.xlabel('Number of people in the frame')
        plt.ylabel('Frequency')
        plt.title('Histogram of People Count per Frame')
        plt.show()
    
    # Extract number of heads per frame
    num_heads_list = []
    for gt in mall_head_positions_gt:
        n_heads = len(gt[0][0][0])
        num_heads_list.append(n_heads)
    
    # Plot histogram
    plot_histogram_with_median(num_heads_list)

def reverse_preprocessing(x):
    image = x.copy() 

    image[:, :, 0] = image[:, :, 0] * 0.229 + 0.485  # R
    image[:, :, 1] = image[:, :, 1] * 0.224 + 0.456  # G
    image[:, :, 2] = image[:, :, 2] * 0.225 + 0.406  # B

    image = image * 255.0
    image = np.clip(image, 0, 255).astype(np.uint8)

    return image

def plot_density_map_comparisons(test_generator, model, num_samples=3, random_seed=42):
    random.seed(random_seed)
    np.random.seed(random_seed)

    # Get one random batch
    X_batch, y_batch = next(iter(test_generator))

    # Predict the density maps
    y_pred = model.predict(X_batch)

    # Choose random indices from the batch
    indices = np.random.choice(range(X_batch.shape[0]), size=num_samples, replace=False)

    for i, idx in enumerate(indices):
        original_img = reverse_preprocessing(X_batch[idx][..., :3])  # Take RGB channels
        ground_truth = y_batch[idx].squeeze()
        prediction = y_pred[idx].squeeze()

        # Calculate counts
        gt_count = np.round(np.sum(ground_truth))
        pred_count = np.round(np.sum(prediction))

        # Plotting
        plt.figure(figsize=(12, 4))
        plt.suptitle(f"Sample {i+1}", fontsize=16)

        plt.subplot(1, 3, 1)
        plt.imshow(original_img)
        plt.title("Original Image")
        plt.axis('off')

        plt.subplot(1, 3, 2)
        plt.imshow(ground_truth, cmap='jet')
        plt.title(f"Ground Truth\nCount: {int(gt_count)}")
        plt.colorbar()
        plt.axis('off')

        plt.subplot(1, 3, 3)
        plt.imshow(prediction, cmap='jet')
        plt.title(f"Predicted\nCount: {int(pred_count)}")
        plt.colorbar()
        plt.axis('off')

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()

def plot_preprocessing_output_comparison(trained_model, test_generator, num_samples=3, random_seed=42):
    random.seed(random_seed)
    np.random.seed(random_seed)

    # Access the preprocessing module from the trained model
    preprocessing_module = trained_model.preprocessing_module

    # Get one random batch
    X_batch, y_batch = next(iter(test_generator))

    # Choose random indices from the batch
    indices = np.random.choice(range(X_batch.shape[0]), size=num_samples, replace=False)

    for i, idx in enumerate(indices):
        input_sample = X_batch[idx:idx+1]  # Select a single sample (batch size = 1)

        # Get the output of the preprocessing module
        preprocessed_output = preprocessing_module(input_sample, training=False).numpy().squeeze()

        # Extract RGB channels and combined output
        original_img = reverse_preprocessing(X_batch[idx][..., :3])  # Take RGB channels
        density_map = y_batch[idx].squeeze()  # Take density map from y_batch
        combined_output = preprocessed_output  # All 3 channels (mixed RGB + density map)

        # Plotting
        plt.figure(figsize=(15, 5))
        plt.suptitle(f"Sample {i+1}", fontsize=16)

        plt.subplot(1, 3, 1)
        plt.imshow(original_img)
        plt.title("Original Image")
        plt.axis('off')

        plt.subplot(1, 3, 2)
        plt.imshow(density_map, cmap='jet')
        plt.title("Density Map")
        plt.colorbar()
        plt.axis('off')

        plt.subplot(1, 3, 3)
        plt.imshow(np.clip(combined_output, 0, 1))  # Clip values for visualization
        plt.title("Preprocessed Output")
        plt.axis('off')

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()