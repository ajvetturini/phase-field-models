import setuptools
import os

with open(os.path.join(os.path.dirname(__file__), 'README.md'), "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r") as f:
    required_packages = f.read().splitlines()

setuptools.setup(
    name="pfm", 
    version="0.1.0",  
    author="A.J. Vetturini",  
    author_email="avetturi@andrew.cmu.edu", 
    description="Phase field modelling with JAX",  
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ajvetturini/phase-field-models", 
    packages=setuptools.find_packages(),  
    install_requires=required_packages, 
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",  # Replace with your license
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.12',  # Specify the minimum Python version your library supports
)
