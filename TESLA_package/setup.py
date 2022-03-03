import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="TESLAforST", 
    version="1.2.1",
    author="Jian Hu",
    author_email="jianhu@pennmedicine.upenn.edu",
    description="TESLA: Deciphering tumor ecosystems at super-resolution from spatial transcriptomics",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/jianhuupenn/TESLA",
    packages=setuptools.find_packages(),
    install_requires=["torch","pandas","numpy","scipy","scanpy","anndata","sklearn", "numba"],
    #install_requires=[],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
