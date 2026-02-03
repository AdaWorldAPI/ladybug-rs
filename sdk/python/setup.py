#!/usr/bin/env python3
from setuptools import setup

setup(
    name="ladybug-vsa",
    version="0.3.0",
    description="Python SDK for LadybugDB cognitive database",
    long_description=open("README.md").read() if __import__("os").path.exists("README.md") else "",
    long_description_content_type="text/markdown",
    author="Ada Consciousness Project",
    url="https://github.com/AdaWorldAPI/ladybug-rs",
    py_modules=["ladybugdb"],
    python_requires=">=3.8",
    install_requires=[],  # Zero dependencies - uses stdlib urllib
    extras_require={
        "requests": ["requests>=2.25.0"],
        "pandas": ["pandas>=1.0.0"],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Database",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    keywords="vector database cognitive vsa hamming fingerprint nars",
)
