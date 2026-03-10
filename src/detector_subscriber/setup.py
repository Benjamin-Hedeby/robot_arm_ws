from setuptools import setup
import os
from glob import glob

package_name = 'detector_subscriber'

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='dev_computer',
    maintainer_email='chrmoric@gmail.com',
    description='Visualizes results of "detector" package',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            f'spatial_subscriber = {package_name}.spatial_tracker_subscriber_node:main',
            f'aruco_visualizer = {package_name}.aruco_subscriber_node:main',
        ],
    },
)
