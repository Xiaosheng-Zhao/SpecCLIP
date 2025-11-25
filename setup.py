from setuptools import setup, find_packages

setup(
    name="specclip",
    version="1.0.0",
    packages=find_packages(include=['specclip', 'specclip.*']),
    install_requires=[
        'torch',
        'lightning>=1.9.0',
        'numpy',
        'h5py',
        'python-dotenv',
    ],
    python_requires='>=3.8',
)
