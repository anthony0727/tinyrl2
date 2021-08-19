from setuptools import setup, find_packages

setup(
    name='tinyrl2',
    version='1.0.0',
    author='Won Seok Jung',
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        'torch',
        # 'numpy',
        # 'pyyaml',
        # 'gin-config',
        # 'click',
        # 'tqdm',
        # 'mpi4py',
        # 'absl-py',
        # 'tensorboard',
        # 'marlenv @ git+https://github.com/kc-ml2/marlenv.git',
        # 'matplotlib', 'moviepy', 'pillow<=7.0.0',
    ],
    # dependency_links=[
    # ],
    # extras_require={
    #     'visual': ['matplotlib', 'moviepy', 'pillow<=7.0.0', ]
    # },
    # entry_points={
    #     'console_scripts': ['rl2=rl2.cli:main']
    # }
)
