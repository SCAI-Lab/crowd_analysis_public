from setuptools import setup, find_packages

setup(
    name='ab3dmot',
    version='0.1',
    packages=find_packages(include=['AB3DMOT_libs', 'data', 'scripts', 'Xinshuo_PyToolbox', 'Xinshuo_PyToolbox.*']),
    install_requires=[],
    description="Object tracking from LiDAR point cloud both in 2D and 3D",
)
