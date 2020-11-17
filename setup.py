import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="weakformghnn",
    version="1.0.0",
    author="Kevin L. Course, Trefor W. Evans, Prasanth B. Nair",
    author_email="kevin.course@mail.utoronto.ca",
    description="PyTorch implementation of weak form generalized Hamiltonian learning.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/coursekevin/weakformghnn",
    packages=setuptools.find_packages(),
    install_requires=['torch'],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
)
