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
