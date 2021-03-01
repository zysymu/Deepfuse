import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="Deepfuse",
    version="1.0.0",
    author="Marcos Tidball",
    author_email="marcostidball@ufrgs.br",
    description="A python package for automatically detecting and classifying Low Surface Brightness Galaxies!",
    long_description=long_description,
    url="https://github.com/zysymu/Deepfuse",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
