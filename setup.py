# setup.py - NeuroFusion Controller Package Setup
from setuptools import setup, find_packages

setup(
    name="neurofusion-controller",
    version="1.0.0",
    description="Hybrid Brain-AI Control System for Real-Time Human-Machine Collaboration",
    author="Goktug Can Simay",
    author_email="simaygoktug@gmail.com",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.21.0",
        "torch>=1.12.0",
        "scipy>=1.8.0",
        "matplotlib>=3.5.0",
        "seaborn>=0.11.0",
        "pyserial>=3.5",
        "scikit-learn>=1.1.0",
        "pandas>=1.4.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
        ],
        "hardware": [
            "pyserial>=3.5",
            "socket",
        ],
        "visualization": [
            "matplotlib>=3.5.0",
            "seaborn>=0.11.0",
            "plotly>=5.0.0",
        ]
    },
    python_requires=">=3.8",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
)