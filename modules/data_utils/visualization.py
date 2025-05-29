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

def plot_density_comparisons(ground_truths,predictions, num_samples=3):

    assert len(predictions) == len(ground_truths), "Predictions and ground truths must have the same length."
    
    total_samples = len(predictions)
    indices = np.random.choice(total_samples, size=num_samples, replace=False)
    
    fig, axes = plt.subplots(num_samples, 2, figsize=(8, 4 * num_samples))
    
    for i, idx in enumerate(indices):
        # Predicted
        ax_pred = axes[i, 0] if num_samples > 1 else axes[0]
        ax_pred.imshow(predictions[idx], cmap='viridis')
        ax_pred.set_title(f"Predicted #{idx}")
        ax_pred.axis('off')
        
        # Ground truth
        ax_gt = axes[i, 1] if num_samples > 1 else axes[1]
        ax_gt.imshow(ground_truths[idx], cmap='viridis')
        ax_gt.set_title(f"Ground Truth #{idx}")
        ax_gt.axis('off')
    
    plt.tight_layout()
    plt.show()