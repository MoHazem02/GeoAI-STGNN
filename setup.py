"""Setup script for the Spatio-Temporal GNN & Active Learning Pipeline."""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="satellite-change-detection",
    version="0.1.0",
    author="AI/GIS Research Engineer",
    author_email="research@example.com",
    description="A Spatio-Temporal GNN & Active Learning Pipeline for Satellite Environmental Change Detection",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/your-username/satellite-change-detection",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: GIS",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "satcd=main:main",
        ],
    },
)
