#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 4 || $# -gt 5 ]]; then
  echo "Usage: $0 DATASET CONFIG_YAML FOLDER MODEL_3D [MODEL_2D]" >&2
  echo "DATASET: CrowdBot, JRDB, JRDB_TEST, or SCAND" >&2
  exit 2
fi

dataset="$1"
config_path="$2"
folder="$3"
model_3d="$4"
model_2d="${5:-}"

if [[ "$dataset" == "CrowdBot" ]]; then
  dataset="Crowdbot"
fi

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/.." && pwd)"
cd "$repo_root"

common=(--dataset "$dataset" --config "$config_path" --folder "$folder")
lidar_extra=()
pose_extra=()

case "$dataset" in
  Crowdbot)
    lidar_extra=(
      --topic-3d /front_lidar/vp_global
      --topic-3d /rear_lidar/vp_global
      --topic-2d /front_lidar/scan_modified
      --topic-2d /rear_lidar/scan_modified
      --topic-2d /scan_multi
    )
    ;;
  JRDB)
    : "${JRDB_TRAIN_TIMESTAMPS_ROOT:?Set JRDB_TRAIN_TIMESTAMPS_ROOT to the JRDB train timestamps directory}"
    lidar_extra=(
      --topic-3d /upper_velodyne/velodyne_points
      --topic-3d /lower_velodyne/velodyne_points
      --topic-2d /segway/scan_multi
      --jrdb-train-timestamps-root "$JRDB_TRAIN_TIMESTAMPS_ROOT"
    )
    ;;
  JRDB_TEST)
    : "${JRDB_TEST_ROOT:?Set JRDB_TEST_ROOT to the JRDB test dataset root}"
    : "${JRDB_TEST_ODOM_ROOT:?Set JRDB_TEST_ODOM_ROOT to the SteamLO test odometry directory}"
    lidar_extra=(--jrdb-test-root "$JRDB_TEST_ROOT" --jrdb-test-odom-root "$JRDB_TEST_ODOM_ROOT")
    pose_extra=(--jrdb-test-odom-root "$JRDB_TEST_ODOM_ROOT")
    ;;
  SCAND)
    ;;
  *)
    echo "Unsupported full-pipeline dataset: $dataset" >&2
    exit 2
    ;;
esac

conda run -n ros_env python "$script_dir/1_Lidar_from_rosbags.py" "${common[@]}" "${lidar_extra[@]}"
conda run -n ros_env python "$script_dir/2_Pose_from_rosbags.py" "${common[@]}" "${pose_extra[@]}"

detection_extra=()
tracking_extra=()
if [[ -n "$model_2d" ]]; then
  if [[ "$dataset" == "JRDB_TEST" ]]; then
    echo "JRDB_TEST has no 2D lidar; do not pass MODEL_2D." >&2
    exit 2
  fi
  detection_extra=(--detect-2d --model-2d "$model_2d")
  tracking_extra=(--track-2d)
fi

conda run -n crowd_env python "$script_dir/3_Detections_from_lidar.py" \
  "${common[@]}" --model "$model_3d" "${detection_extra[@]}"
conda run -n crowd_env python "$script_dir/4_Tracks_from_detections.py" \
  "${common[@]}" "${tracking_extra[@]}"

echo "All four processing stages completed."
