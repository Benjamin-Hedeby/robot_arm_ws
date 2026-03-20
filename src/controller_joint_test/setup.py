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
            'gripper_controller = controller_joint_test.gripper_controller:main'
        ],
    },
)