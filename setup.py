from setuptools import setup, find_packages
setup(
    name="fasta",
    version="0.1",
    description="fast adaptive proximal gradient descent algorithm",
    packages=find_packages(),

    install_requires=[
        'numpy >= 1.8',
        'scipy >=0.17',
    ],


    # metadata for upload to PyPI
    author = "Proloy Das",
    author_email = "proloy@umd.com",
    license = "apache",
    project_urls = {
        "Source Code": "https://github.com/proloyd/fastapy",
    }
)
