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
        f'build_map = {package_name}.build_map:main',
        f'create_map = {package_name}.semantic_mapping:main',
        f'map_pub = {package_name}.map_publisher:main',
        f'segmap_pub = {package_name}.segmap_pub:mail',
        ],
    },
)
