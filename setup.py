import setuptools
import os

with open(os.path.join(os.path.dirname(__file__), 'README.md'), "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r") as f:
    required_packages = f.read().splitlines()

setuptools.setup(
    name="pfm",  # Replace with your library's name (should be unique on PyPI)
    version="0.1.0",  # Replace with your library's version
    author="A.J. Vetturini",  # Replace with your name
    author_email="avetturi@andrew.cmu.edu",  # Replace with your email
    description="Phase field modelling with JAX",  # Replace with a concise description
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ajvetturini/phase-field-models",  # Replace with your library's repository URL (if any)
    packages=setuptools.find_packages(),  # Automatically finds your package directories
    install_requires=required_packages,  # Installs dependencies from requirements.txt
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",  # Replace with your license
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.12',  # Specify the minimum Python version your library supports
)