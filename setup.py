from setuptools import setup, find_packages

setup(
    name='ensemble_pruning',
    version='0.1.0',
    description='Ensemble pruning meta-estimator for scikit-learn',
    author='Christian Messina',
    author_email='christian.messina.val@gmail.com',
    license='BSD-3-Clause',
    packages=find_packages(),
    install_requires=[
        'numpy>=1.15.0',
        'scikit-learn>=0.20.0'
    ],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: BSD License',
        'Programming Language :: Python :: 3',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
)