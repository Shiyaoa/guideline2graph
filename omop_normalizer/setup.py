"""
OMOP Chinese Term Normalizer

中文医学术语标准化工具包，用于将中文临床指南文本映射到OMOP CDM标准词汇表
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="omop-normalizer",
    version="1.0.0",
    author="Medical Informatics Team",
    author_email="",
    description="中文医学术语标准化工具，映射到OMOP CDM词汇表",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Healthcare Industry",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
    ],
    python_requires=">=3.8",
    install_requires=[
        "pandas>=1.3.0",
        "requests>=2.25.0",
        "openai>=1.0.0",
    ],
    extras_require={
        "fuzzy": ["rapidfuzz>=2.0.0"],
        "anthropic": ["anthropic>=0.18.0"],
        "excel": ["openpyxl>=3.0.0"],
        "all": [
            "rapidfuzz>=2.0.0",
            "anthropic>=0.18.0",
            "openpyxl>=3.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "omop-normalize=omop_normalizer.cli:main",
        ],
    },
)