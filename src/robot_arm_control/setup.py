from setuptools import find_packages, setup

package_name = 'robot_arm_control'

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
            'live_ik_streamer = robot_arm_control.live_ik_streamer:main',
            'detections_republish = robot_arm_control.detections_republish:main',
            'task_controller = robot_arm_control.task_controllerV2:main',
            'IK_square_test = robot_arm_control.IK_square_test:main',
            'rviz_translator = robot_arm_control.rviz_translator:main',
        ],
    },
)
