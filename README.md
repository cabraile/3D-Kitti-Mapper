This repository is still a work in progress.

# About
In this repository, I implemented tools for building the 3D point cloud map for the (raw, sync) Kitti dataset, this way you do not need to reinvent the wheel everytime you wanted to work with 3D maps.

# Setup

## Option 1: Anaconda

**Environment file still not concluded.**

Execute on the terminal:
```bash
conda env create -f environment.yml
```

## Option 2: Pip
Assuming you have your NVIDIA drivers installed and the appropriate cudatoolkit versions on your machine, execute on the terminal:
```bash
pip install -r requirements.txt
```

# Download the dataset
The raw (synced + rectified) dataset can be downloaded from the [project's webpage](https://www.cvlibs.net/datasets/kitti/raw_data.php). Download both the data and the calibration data of interest.

# Running

**First step:** Change the settings in `settings.yaml` (believe me, the arguments are described properly there).

**Second step:** Run 
```bash
python3 scripts/kitti_mapper.py \
    --date ${DATE} \
    --drive ${DRIVE} \
    --output ${OUTPUT} \
    --dataset ${DATASET}
```

About the arguments:
* `--date`: The date of the recorded dataset.
* `--drive`: The drive dataset identifier.
* `--output`: Path to the output `PLY` formated pointcloud.
* `--dataset`: The path where the Kitti dataset is stored. It should be the directory organized in the following structure:
  * `${DATASET}/`
    * `2011_09_30/`
    * `2011_10_03/`
    * ...
    * `${DATE}/`
      * `${DATE}_drive_{DRIVE}_sync/`
        * `image_02/`
        * `image_03/`
        * `oxts/`
        * `velodyne_points/`
      * ...
      * `calib_cam_to_cam.txt`
      * `calib_imu_to_velo.txt`
      * `calib_velo_to_cam.txt`
