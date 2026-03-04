from setuptools import setup, find_packages

setup(
    name='phylotraitGNN',
    description='A package for running GNNs on phylogenetic trees',
    license='Attribution-NonCommercial-ShareAlike 4.0 International',
    packages=find_packages(),

    install_requires=[
        "pandas",
        "numpy",
        'scikit-learn',
        'torch',
        'torch-geometric',
        'biopython',
        'networkx',
        'bayesian-optimization==3.2.0'
    ],
    # *strongly* suggested for sharing
    version='1.0',
    long_description=open('README.md').read(),
)
