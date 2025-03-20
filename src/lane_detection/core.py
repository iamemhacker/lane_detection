import cv2
import numpy as np
import os
import json

from sklearn.cluster import KMeans
# from sklearn.cluster import DBSCAN
from typing import Generator, Tuple


def detect(video_path: str) \
        -> Generator[Tuple[np.array, np.array], None, None]:
    """
    Lane lines generator.
    Yeils a tuple for each frame in the given video path (frame, lines).
    Frames are array in shape [H, W, C] (RGB format).
    Lines are arrays in format [x1, y1, x2, y2].
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(current_dir, "config.json")) as config_file:
        config = json.loads(config_file.read())
        edge_config = config["edges"]
        lines_config = config["lines"]
        clustering_config = config["clustering"]
    [lo, hi] = [edge_config["low_th"], edge_config["hi_th"]]
    [threshold, min_line_length, max_line_gap] = [lines_config[k]
                                                  for k
                                                  in ("threshold",
                                                      "min_line_length",
                                                      "max_line_gap")]
    video = cv2.VideoCapture(video_path)
    while True:
        ret, frame = video.read()
        if not ret:
            break
        image_width = frame.shape[1]
        gray_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray_img, lo, hi)
        lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi/180,
                                threshold=threshold,
                                minLineLength=min_line_length,
                                maxLineGap=max_line_gap)
        if lines.size == 0:
            yield np.zeros(4, dtype=np.int8)
            continue
        grouped = _merge_lines(lines, image_width, **clustering_config)
        yield frame, grouped


def _merge_lines(lines: np.array, img_width: int, slope_cutoff: int, **kargs) \
        -> np.array:
    """
    Merges the lines, so that line with the same slope that are verically
    close, will be merged to a single horizontal line.
    """
    # Filter out the noisy lines based on slope.
    lines = lines.squeeze(axis=1)
    slopes = np.abs((lines[:, 1] - lines[:, 3]) /
                    (lines[:, 0] - lines[:, 2] + 1E-6))
    [cut_point] = np.percentile(slopes, [slope_cutoff * 100])
    horizontal_lines = lines[slopes < cut_point]
    labels = _cluster_lines(horizontal_lines)

    # Group lines by their labels.
    groups = [horizontal_lines[np.where(labels == label)]
              for label
              in np.unique(labels)]
    lines = []
    for group in groups:
        mini = np.argmin(group[:, 0])
        maxi = np.argmax(group[:, 2])
        [x1, y1] = group[mini, :2]
        [x2, y2] = group[maxi, 2:]
        lines.append(np.array([x1, y1, x2, y2]))
    return np.vstack(lines)


def _average_angle(lines: np.array) -> np.float32:
    (X1, Y1, X2, Y2) = (lines[:, c] for c in range(4))
    angles = np.arctan2(Y2 - Y1, X2 - X1)  # Angle of the line
    return np.mean(angles)


def _rotate_points(points, angle):
    """ Rotates points by the given angle (radians) around the origin """
    rotation_matrix = np.array([
        [np.cos(angle), -np.sin(angle)],
        [np.sin(angle), np.cos(angle)]
    ])
    return np.dot(points, rotation_matrix.T)


def _rotate_lines(lines, angle):
    """ Rotates all lines by the given angle """
    rotated_lines = []
    for x1, y1, x2, y2 in lines:
        p1 = _rotate_points(np.array([[x1, y1]]), angle)[0]
        p2 = _rotate_points(np.array([[x2, y2]]), angle)[0]
        rotated_lines.append([p1[0], p1[1], p2[0], p2[1]])
    return np.array(rotated_lines)


def _cluster_lines(lines: np.array, n_clusters: int = 4) -> np.array:
    avg_angle = _average_angle(lines)
    # Align lanes horizontally
    lines_rotated = _rotate_lines(lines, -avg_angle)
    # Average Y of each line
    y_coords = np.mean(lines_rotated[:, [1, 3]], axis=1)
    dy = (lines_rotated[:, 3] - lines_rotated[:, 1])
    dx = (lines_rotated[:, 2] - lines_rotated[:, 0])
    slopes = dy / (dx + 1e-6)
    features = np.column_stack((y_coords, slopes))
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(features)
    # dbscan = DBSCAN(eps=10, min_samples=2)
    # labels = dbscan.fit_predict(features)
    return labels
