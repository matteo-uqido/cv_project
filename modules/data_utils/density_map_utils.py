import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.spatial import KDTree
#takes as input a list of lists of head positions
def create_ground_truth(head_positions, height=480, width=640):
    ground_truth = np.zeros((height, width), dtype=np.float32)
    for pos in head_positions:
        x, y = pos
        xi = int(round(x))
        yi = int(round(y))
        if 0 <= xi < width and 0 <= yi < height:
            ground_truth[yi, xi] = 1
    return ground_truth


# Generates a density map using Gaussian filter transformation
def gaussian_filter_density(ground_truth, beta=0.1):
    density = np.zeros(ground_truth.shape, dtype=np.float32)
    ground_truth_count = np.count_nonzero(ground_truth)

    if ground_truth_count == 0:
        return density

    index_of_nonzero_elements = np.nonzero(ground_truth)
    points = np.array(list(zip(index_of_nonzero_elements[1].ravel(), index_of_nonzero_elements[0].ravel())))

    tree = KDTree(points.copy(), leafsize=2048)
    distances, _ = tree.query(points, k=4)

    for i, point in enumerate(points):
        point_2d = np.zeros(ground_truth.shape, dtype=np.float32)
        point_2d[point[1], point[0]] = 1.

        if ground_truth_count > 1:
            sigma = (distances[i][1] + distances[i][2] + distances[i][3]) * beta
        else:
            sigma = np.average(np.array(ground_truth.shape)) / 4.

        density += gaussian_filter(point_2d, sigma, mode='constant')
    return density

def scale_coordinates(head_positions, scale_x=0.5, scale_y=0.5):
    return [[x * scale_x, y * scale_y] for x, y in head_positions]

def create_density_map(head_positions,beta=0.1):
    scaled_positions = scale_coordinates(head_positions)
    ground_truth = create_ground_truth(scaled_positions, height=240, width=320)
    return gaussian_filter_density(ground_truth,beta=beta)

def create_density_map_optimized(head_positions,beta=0.1):
    scaled_positions = scale_coordinates(head_positions)
    ground_truth = create_ground_truth(scaled_positions, height=240, width=320)
    return gaussian_filter_density_optimized(ground_truth,beta=beta)


def gaussian_kernel_2d(size, sigma):
    """Create a 2D Gaussian kernel array."""
    ax = np.arange(-size // 2 + 1, size // 2 + 1)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    return kernel / np.sum(kernel)

def gaussian_filter_density_optimized(ground_truth, beta=0.1):
    density = np.zeros(ground_truth.shape, dtype=np.float32)
    ground_truth_count = np.count_nonzero(ground_truth)

    if ground_truth_count == 0:
        return density

    # Get nonzero points (x, y) coordinates
    index_of_nonzero_elements = np.nonzero(ground_truth)
    points = np.array(list(zip(index_of_nonzero_elements[1], index_of_nonzero_elements[0])))

    tree = KDTree(points.copy(), leafsize=2048)
    distances, _ = tree.query(points, k=4)

    h, w = ground_truth.shape

    for i, point in enumerate(points):
        if ground_truth_count > 1:
            sigma = (distances[i][1] + distances[i][2] + distances[i][3]) * beta
        else:
            sigma = np.average(np.array(ground_truth.shape)) / 4.

        # Define kernel size (always odd)
        kernel_size = int(6 * sigma) + 1
        kernel = gaussian_kernel_2d(kernel_size, sigma)  # You should have this function defined elsewhere

        x, y = point
        half_k = kernel_size // 2

        # Initial bounds on density map
        x1 = x - half_k
        y1 = y - half_k
        x2 = x + half_k + 1
        y2 = y + half_k + 1

        # Initial bounds on kernel
        kx1 = 0
        ky1 = 0
        kx2 = kernel_size
        ky2 = kernel_size

        # Adjust bounds if out of density map (image) bounds
        if x1 < 0:
            kx1 = -x1
            x1 = 0
        if y1 < 0:
            ky1 = -y1
            y1 = 0
        if x2 > w:
            kx2 = kernel_size - (x2 - w)
            x2 = w
        if y2 > h:
            ky2 = kernel_size - (y2 - h)
            y2 = h

        kernel_crop = kernel[ky1:ky2, kx1:kx2]

        # Safety check: both shapes must match exactly
        assert kernel_crop.shape == density[y1:y2, x1:x2].shape, \
            f"Shape mismatch: kernel_crop {kernel_crop.shape}, density patch {density[y1:y2, x1:x2].shape}"

        density[y1:y2, x1:x2] += kernel_crop

    return density

