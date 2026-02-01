from setuptools import setup

setup(
    name="ladybugdb",
    version="0.2.0",
    description="Python SDK for LadybugDB cognitive database",
    author="Jan Hübener",
    py_modules=["ladybugdb"],
    install_requires=["requests>=2.28"],
    python_requires=">=3.8",
)
