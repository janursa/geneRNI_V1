import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

with open("requirements.txt", "r") as f:
    requirements = f.read().splitlines()


setuptools.setup(
    name="geneRNI",
    version="1.0.2",
    author="Jalil Nourisa, Antoine Passemiers",
    author_email="jalil.nourisa@gmail.com",
    description="A Python package for gene regulatory network inference",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/janursa/geneRNI",
    package_dir = {'geneRNI':'geneRNI'},
    packages=['geneRNI', 'geneRNI.models', 'geneRNI.grn'],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.8',
    install_requires=requirements,
    include_package_data=True,
)

