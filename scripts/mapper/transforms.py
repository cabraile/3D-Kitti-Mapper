import numpy as np

def apply_transform(T : np.ndarray, points : np.ndarray) -> np.ndarray :
    """points: N-by-channels"""
    points_homog = np.hstack( [points, np.ones((len(points),1)) ] )
    points_homog = (T @ points_homog.T).T
    return points_homog[:,:-1]