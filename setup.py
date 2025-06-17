from setuptools import find_packages, setup

with open("requirements.txt", "r", encoding="utf-8") as f:
    requirements = f.read().splitlines()

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="data-alchemy",
    author="wuhanghao",
    author_email="wuhanghao@gmail.com",
    url="https://github.com/VivaLaDijkstra/data-alchemy",
    description="""Data Alchemy is a tool for data building, cleaning, analysis, and visualization.
Debug and explore your data with a simple command-line interface.""",
    long_description=long_description,
    long_description_content_type="text/markdown",
    version="0.0.1",
    packages=find_packages(),
    install_requires=requirements,
    python_requires=">=3.10",
    extras_require={
        "dev": ["pytest", "black"],
        "docs": ["Sphinx"],
    },
    entry_points={
        "console_scripts": [
            "data-alchemy=data_alchemy.main:main",  # command=module:function
        ],
    },
    license="MIT",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
