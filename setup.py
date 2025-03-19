from setuptools import setup, find_packages

setup(
    name="vep_protein",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "pandas",
        "numpy",
        "requests",
        "tqdm",
    ],
)
