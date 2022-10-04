import os
import sys
from typing import List
import numpy as np

from scipy.spatial.transform import Rotation

from shapely.geometry import box, Point

MODULES_PATH = os.path.realpath( 
    os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
    )
)
sys.path.append(MODULES_PATH)
from transforms import apply_transform


def filter_dynamic_objects(
    points_array : np.ndarray, 
    rgb_array : np.ndarray, 
    K : np.ndarray, 
    T_from_lidar_to_camera : np.ndarray,
    bounding_boxes_xyxy : List[np.ndarray]
) -> np.ndarray:
    """Filter points that belong to dynamic objects in the LiDAR frame.
    
    Arguments
    -----
    points_array : Scan points' (N-by-M) array, where N represents the number 
        of points and M (M>=3) the number of channels.
    rgb_array : The image's channels-last array to which the scans will be 
        projected.
    K : The camera calibration matrix.
    T_from_lidar_to_camera : The 4-by-4 extrinsic transformation matrix (SE3) 
        that projects the points from the lidar frame to the camera frame
    mask_filter : The boolean mask indicating the parts of the image where the
        projected points will be accepted (where true).
    bounding_boxes_xyxy: The list of bounding boxes used for filtering out 
        points that are contained inside them after projected.
        The bounding boxes are represented by the array
        (x1,y1,x2,y2), where (x1,y1) represents the bounding box in the 
        top-left corner and (x2,y2) represents the bounding box in the 
        bottom-right corner.

    Returns
    ------
    The xyzrgb K-by-6 array, where xyzrgb is the array of the K-remaining 
    points that are not contained in the boxes. The points that fall outside of 
    the image boundaries have rgb = 0. Also, the RGB value is in the 0-1 range.
    """# Project the lidar scan to image coordinates
    projected_homog = apply_transform( T_from_lidar_to_camera , points_array[:,:3] )
    projected_homog = (K @ projected_homog.T).T
    
    # Drop points that are behind the camera
    mask = projected_homog[:,2] >= 0 
    projected_homog = projected_homog[mask] 
    points_array = points_array[mask]
    col_row_coordinates = (projected_homog[:,:2]/projected_homog[:,2].reshape(-1,1)).astype(int)

    # Get oob points' mask
    height, width = rgb_array.shape[:2]
    rows = col_row_coordinates[:,1]
    cols = col_row_coordinates[:,0]
    mask = (cols > 0) & (cols < width) & (rows > 0) & (rows < height)

    # Filter oob points
    rows = rows[mask]
    cols = cols[mask]
    points_array = points_array[mask]

    # Check which points are not inside the bounding boxes of the detections
    points_shp = [ Point(cols[i],rows[i]) for i in range(len(points_array)) ]
    is_contained_inside_any_box = np.full((len(points_shp)),False)

    # TODO: vectorize
    for bbox in bounding_boxes_xyxy:
        bbox_shp = box(*bbox)
        for idx, point in enumerate(points_shp):
            is_contained_inside_any_box[idx] = is_contained_inside_any_box[idx] | point.within(bbox_shp) | bbox_shp.contains(point)
    
    # Filter points that were contained in a box
    points_array = points_array[~is_contained_inside_any_box]
    rows = rows[~is_contained_inside_any_box]  
    cols = cols[~is_contained_inside_any_box]

    # Colorize RGB XYZ    
    points_rgb = rgb_array[rows, cols] / 255.
    xyz_rgb = np.hstack([points_array[:,:3], points_rgb])

    return xyz_rgb