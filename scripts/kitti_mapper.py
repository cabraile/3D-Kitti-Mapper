import os
import sys
import time
import argparse
import yaml
import tqdm

import numpy as np

import utm
from scipy.spatial.transform import Rotation

import pykitti
from pykitti.utils import OxtsPacket


import open3d as o3d
import torch

ROOT_DIR = os.path.realpath( 
    os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        os.path.pardir
    )
)

MODULES_PATH =os.path.realpath( 
    os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "mapper"
    )
)
sys.path.append(MODULES_PATH)
from lidar_tools import filter_dynamic_objects

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, help="Path to the dataset root directory.")
    parser.add_argument("--date",    required=True, help="Date of the dataset recording (YYYY_MM_DD).")
    parser.add_argument("--drive",   required=True, help="Drive number (XXXX).")
    parser.add_argument("--output",  default= "output.ply", help="Path to the output PLY formated pointcloud.")
    return parser.parse_args()

def get_groundtruth_state_as_array( seq_data : OxtsPacket) -> np.ndarray:
    """Given a Kitti GPS data, convert it to a state vector."""
    easting, northing, _, _ = utm.from_latlon(seq_data.lat, seq_data.lon)
    elevation = seq_data.alt
    roll    = seq_data.roll
    pitch   = seq_data.pitch
    yaw     = seq_data.yaw
    velocity_north  = seq_data.vn
    velocity_east   = seq_data.ve
    velocity_up     = seq_data.vu
    velocity_roll   = seq_data.wx
    velocity_pitch  = seq_data.wy
    velocity_yaw    = seq_data.wz
    data_array = np.array([
        [easting], [northing], [elevation],
        [roll], [pitch], [yaw],
        [velocity_east], [velocity_north], [velocity_up],
        [velocity_roll], [velocity_pitch], [velocity_yaw]
    ])
    return data_array

def from_state_to_transform(translation : np.ndarray, rpy : np.ndarray) -> None:
    T = np.eye(4)
    T[:3,:3] = Rotation.from_euler("xyz", angles=rpy, degrees=False).as_matrix()
    T[:3, 3] = np.copy(translation)
    return T

def main() -> int:
    args = parse_args()
    settings_path = os.path.join(ROOT_DIR,"cfg","settings.yaml")
    with open(settings_path,"r") as settings_file:
        settings_dict = yaml.load(settings_file, yaml.CFullLoader)

    # Kitti data
    data = pykitti.raw(os.path.abspath(args.dataset), args.date, args.drive)
    K_left  = data.calib.K_cam3
    K_right = data.calib.K_cam2
    T_from_lidar_to_camera_left = data.calib.T_cam3_velo
    T_from_lidar_to_camera_right= data.calib.T_cam2_velo
    T_from_baselink_to_lidar         = data.calib.T_velo_imu
    T_from_lidar_to_baselink         = np.linalg.inv(T_from_baselink_to_lidar) # TODO: use closed-form formula
    
    T_from_left_camera_to_baselink   = T_from_lidar_to_baselink @ np.linalg.inv(T_from_lidar_to_camera_left)
    T_from_right_camera_to_baselink  = T_from_lidar_to_baselink @ np.linalg.inv(T_from_lidar_to_camera_right)

    if settings_dict["filter_dynamic"]:
        model = torch.hub.load('ultralytics/yolov5', 'yolov5m', pretrained=True)

    # Main loop
    print()
    print("Started loop")
    print("======================")

    global_pcd = o3d.geometry.PointCloud()
    n_iters = len(data) if settings_dict["load_n_frames"] == 0 else min(settings_dict["load_n_frames"],len(data))
    
    for seq_idx in tqdm.tqdm(range(0,n_iters, settings_dict["sequence_step"])):
        # Retrieve sequence data
        timestamp = (data.timestamps[seq_idx] - data.timestamps[0]).total_seconds()

        # Filter lidar points
        points_lidar = data.get_velo(seq_idx)
        points_lidar = points_lidar[ np.linalg.norm(points_lidar[:,:3],axis=1) >= 1.0 ] # Filter points of the vehicle
        points_lidar = points_lidar[ points_lidar[:,2] >= -3.0 ] # Filter weird points that are down the road.

        # Removes LiDAR points that contain dynamic objects
        if settings_dict["colorize_lidar"]:
            rgb_right, rgb_left = data.get_rgb(seq_idx) # right = cam2, left = cam3
            rgb_left = np.array(rgb_left)

            # Filter out dynamic objects
            if settings_dict["filter_dynamic"]:
                model_detection = model(rgb_left)
                detections = model_detection.pandas().xyxy[0]
                detections = detections[detections["confidence"] > 0.5]
                detections = detections[detections["name"].isin(["car", "person", "bicycle", "motorcycle", "handbag", "train", "truck", "bus"]) ]

                detections_bboxes = [ np.array(row[["xmin", "ymin","xmax","ymax"]]) for _, row in detections.iterrows() ]
            else:
                detections_bboxes = []

            # Colorize and filter (TODO: decouple)
            points_lidar = filter_dynamic_objects(
                points_array=points_lidar[:,:3],
                rgb_array=rgb_left,
                K=K_left,
                T_from_lidar_to_camera=T_from_lidar_to_camera_left,
                bounding_boxes_xyxy=detections_bboxes
            )

        # Convert from the LiDAR frame to the world frame
        gps_and_imu_data = data.oxts[seq_idx].packet
        groundtruth_state_array = get_groundtruth_state_as_array(gps_and_imu_data)
        T_from_baselink_to_world = from_state_to_transform(groundtruth_state_array[:3].flatten(), groundtruth_state_array[3:6].flatten())

        T_from_lidar_to_world = T_from_baselink_to_world @ T_from_lidar_to_baselink
        points_lidar_homog_T = np.hstack( [ points_lidar[:,:3], np.ones( (len(points_lidar),1) ) ] ) .T
        points_world = (T_from_lidar_to_world @ points_lidar_homog_T).T[:,:3]

        # Store points in the point cloud
        local_pcd = o3d.geometry.PointCloud()
        local_pcd.points = o3d.utility.Vector3dVector(points_world)
        if settings_dict["filter_dynamic"]:
            local_pcd.colors = o3d.utility.Vector3dVector(points_lidar[:,3:])
        global_pcd += local_pcd
        
        # Downsample point cloud
        if seq_idx % settings_dict["downsample_period"] == 0:
            global_pcd = o3d.geometry.voxel_down_sample(global_pcd, settings_dict["voxelgrid_resolution"])

    # Downsample one last time
    global_pcd = o3d.geometry.voxel_down_sample(global_pcd, settings_dict["voxelgrid_resolution"])

    # Write output file
    output_path = os.path.abspath(args.output)
    o3d.io.write_point_cloud(output_path, global_pcd)
    return 0

if __name__=="__main__":
    sys.exit(main())