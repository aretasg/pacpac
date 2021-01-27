from setuptools import setup, find_packages

dependencies = []

setup(
    name="pacpac",
    version="0.1",
    url="https://github.com/aretasg/pacpac",
    license="BSD",
    author="Aretas Gaspariunas",
    author_email="aretasgasp@gmail.com",
    description="Python package to probe antibody VH sequences for a paratope/clonotype of interest and/or cluster into groups of similar paratopes/clonotypes",
    platforms="Linux",
    long_description=open('README.md').read(),
    packages=["pacpac"],
    install_requires=dependencies,
    python_requires=">=3.6.12",
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Operating System :: POSIX :: Linux",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
    ],
)
