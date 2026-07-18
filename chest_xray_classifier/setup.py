"""
Setup configuration for Chest X-Ray Classifier package.

Install with: pip install -e .
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README
readme_file = Path(__file__).parent / "README.md"
long_description = readme_file.read_text(encoding="utf-8") if readme_file.exists() else ""

# Read requirements
requirements_file = Path(__file__).parent / "requirements.txt"
install_requires = []
if requirements_file.exists():
    with open(requirements_file) as f:
        install_requires = [line.strip() for line in f if line.strip() and not line.startswith("#")]

setup(
    name="chest-xray-classifier",
    version="1.0.0",
    description="Deep Learning-based Chest X-Ray Classification System",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="AI Medical Imaging Team",
    author_email="contact@example.com",
    url="https://github.com/example/chest-xray-classifier",
    license="MIT",
    
    packages=find_packages(),
    python_requires=">=3.11",
    install_requires=install_requires,
    
    entry_points={
        "console_scripts": [
            "chest-xray-cli=main:main",
        ],
    },
    
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Healthcare Industry",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Medical Science Apps",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    
    keywords="medical-imaging chest-xray classification deep-learning tensorflow",
    
    project_urls={
        "Documentation": "https://example.com/docs",
        "Issue Tracker": "https://github.com/example/chest-xray-classifier/issues",
        "Source Code": "https://github.com/example/chest-xray-classifier",
    },
)
