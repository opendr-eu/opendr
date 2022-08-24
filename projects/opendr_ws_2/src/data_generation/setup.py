from setuptools import setup

package_name = 'data_generation'

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
    maintainer='OpenDR Project Coordinator',
    maintainer_email='tefas@csd.auth.gr',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'synthetic_facial_generation = data_generation.synthetic_facial_generation:main'
        ],
    },
)
