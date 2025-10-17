#!/bin/bash


# subdir='0325_rds_defaced'

# Activate the ros_env conda environment and run the first Python script
conda activate ros_env
# python3 1_Lidar_from_rosbags.py "$subdir" &
python3 1_Lidar_from_rosbags.py &
pid1=$!
wait $pid1

# Run the second Python script in the same environment
# python3 2_Pose_from_rosbags.py "$subdir" &
python3 2_Pose_from_rosbags.py &
pid2=$!
wait $pid2

# Activate the Jetson conda environment and run the third Python script
conda activate crowd_env
# python3 3_Detections_from_lidar.py "$subdir" &
python3 3_Detections_from_lidar.py &
pid3=$!
wait $pid3

# Run the fourth Python script in the same environment
# python3 4_Tracks_from_detections.py "$subdir" &
python3 4_Tracks_from_detections.py &
pid4=$!
wait $pid4

# Optionally, print a message when all tasks are completed
echo "All tasks have been completed."
