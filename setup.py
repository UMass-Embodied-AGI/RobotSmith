from setuptools import setup, find_packages

setup(
    name="funcany",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "trimesh",
        "open3d",
        "cma",
        "rtree",
        "PyQt5",
        "pyglet<2",
    ],
)
# sudo apt install libgl1-mesa-dri
