from setuptools import setup

package_name = 'create_map'

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
    maintainer_email='ri@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
        f'map_pub = {package_name}.map_publisher:main',
        f'mapping = {package_name}.mapping:main',
        f'new_odom = {package_name}.odom_correction:main',
        f'tf_filter = {package_name}.tf_filter:main',
        f'tf_odom = {package_name}.tf_correction:main',
        ],
    },
)
