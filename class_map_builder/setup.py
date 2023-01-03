from setuptools import setup

package_name = 'class_map_builder'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='ri',
    maintainer_email='dlwhdrlf0619@naver.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
        f'map_pub = {package_name}.map_publisher:main',
        f'class_mapping = {package_name}.class_mapping:main',
        f'tf_odom = {package_name}.tf_correction:main',
        ],
    },
)
