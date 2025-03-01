from setuptools import find_packages, setup

packages = find_packages(include=["segmentation"])

setup(
    name="wacv-seg-proto",
    version="0.1.0",
    author="Hugo Porta",
    description="Extension of ProtoSeg for multi-scale prototyping in XAI",
    packages=packages,
)