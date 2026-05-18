from setuptools import find_packages, setup

package_name = 'controller_joint_test'

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
    maintainer='lucas-brendborg',
    maintainer_email='lucas@brendborg.dk',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'controller_joint_test = controller_joint_test.controller_joint_test:main',
            'live_ik_streamer = controller_joint_test.live_ik_streamer:main',
            'detections_republish = controller_joint_test.detections_republish:main',
            'task_controller = controller_joint_test.task_controller:main',
            'IK_square_test = controller_joint_test.IK_square_test:main',
            'gripper_controller = controller_joint_test.gripper_controller:main',
            'rviz_translator = controller_joint_test.rviz_translator:main',
            'orange_detections = controller_joint_test.orange_detections:main',
<<<<<<< HEAD
            'detections_republish_orange = controller_joint_test.detections_republish_orange:main',
            'orange_detections_test = controller_joint_test.orange_detections_test:main',
=======
            'detections_republish_orange = controller_joint_test.detections_republish_orange:main'
>>>>>>> d26ad43022d8c0e6c284222edd5c1652c90681e3
        ],
    },
)