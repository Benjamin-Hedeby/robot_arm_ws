from setuptools import find_packages, setup

package_name = 'orange_detections'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='dev_computer',
    maintainer_email='benjaminhedeby@live.dk',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'orange_detections = orange_detections.orange_detections:main',
            'detections_republish_orange = orange_detections.detections_republish_orange:main',
            'orange_detections_test = orange_detections.orange_detections_test:main',
        ],
    },
)
