"""
Setup configuration for the GMC-Link tracking library.
"""
from setuptools import setup, find_packages

setup(
    name="GMC_Link",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "torch",
        "torchvision",
        "numpy",
        "opencv-python",
        "sentence-transformers",
        "tqdm",
        "scipy",
    ],
    description="A plug-and-play module for Referring Multi-Object Tracking",
    author="Hsin-Chen Pai",
)
