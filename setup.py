from setuptools import setup, find_packages

setup(
    name='digital consciousness model',
    version='0.1.0',
    description='Bayesian model for DCM',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'pymc',
        'arviz',
        'pandas',
        'requests',
        'matplotlib',
        'seaborn',
        'scikit-learn',
        'pytensor',
        'squigglepy',
    ],
    python_requires='>=3.10, <=3.12.10',
)
