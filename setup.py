from setuptools import setup, find_packages

dependencies = []

setup(
    name="pacpac",
    version="0.3.4",
    url="https://github.com/aretasg/pacpac",
    license="BSD",
    author="Aretas Gaspariunas",
    author_email="aretasgasp@gmail.com",
    description="Python package to probe and cluster antibody paratopes and clonotypes",
    platforms="posix",
    packages=find_packages(),
    package_data={"": ["weights.h5"]},
    install_requires=dependencies,
    python_requires=">=3.7, <3.8",
    entry_points={"console_scripts": "pacpac = pacpac.cli:cli"},
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Operating System :: POSIX",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
    ],
)
