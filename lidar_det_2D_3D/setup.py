from setuptools import setup, find_packages

setup(
    name="lidar_det_2D_3D",
    version="0.1",
    packages=find_packages(
        include=["lidar_det", "lidar_det.*", "lidar_det.*.*"]
    ),
    license="LICENSE.txt",
    description="Object detection from LiDAR point cloud both with DR-SPAAM (2D) and Person-MinkUnet (3D)",
)
