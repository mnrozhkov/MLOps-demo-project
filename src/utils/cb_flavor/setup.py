import setuptools

setuptools.setup(
    name="cb_flavor",
    version="0.0.1",
    description="Catboost flavor for MLflow",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)