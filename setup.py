from setuptools import setup

setup(
    name="kids_sbi_methods",
    version="1.0.0",
    description="Helper package to interface PyDELFI with the Kids SBI simulation pipeline",
    url="https://github.com/Kiyam/kids_sbi_methods",
    author="Kiyam Lin",
    author_email="linkiyam@gmail.com",
    license="BSD 2-clause",
    packages=["kids_sbi_methods"],
    install_requires=[
        "matplotlib",
        "numpy",
        "scipy",
        "tensorflow==1.15",
        "pathlib",
        "configparser",
        "pyDOE",
    ],
    classifiers=[
        "Development Status :: 1 - Planning",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: BSD License",
        "Operating System :: POSIX :: Linux",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.9",
    ],
)
