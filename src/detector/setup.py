from setuptools import setup
import os
from glob import glob

package_name = 'detector'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Christian Dragkilde Moric',
    maintainer_email='chrmoric@gmail.com',
    description='Detect ArUco markers and YOLO spatial detections from the DepthAI camera feed.',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'aruco_detector_node = detector.aruco_detector_node:main',
            'spatial_tracker_node = detector.spatial_tracker_publisher:main',
        ],
    },
)