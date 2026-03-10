from setuptools import setup
import os
from glob import glob

package_name = 'spatial_detector'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        # necessary for ROS2 package discovery
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Christian Dragkilde',
    maintainer_email='chrmoric@gmail.com',
    description='Split spatial detection: Pi publisher + desktop visualizer',
    license='Apache-2.0',
    entry_points={
        'console_scripts': [
            'spatial_overlay = spatial_detector.spatial_overlay:main',
        ],
    },
)

