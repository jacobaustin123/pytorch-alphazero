from setuptools import setup, find_packages

setup(
    name="alphazero", 
    author='Jacob Austin',
    author_email='jacob.austin@columbia.edu',
    packages=find_packages(),
    description='A PyTorch implementation of the 2017 Nature AlphaZero paper from Google DeepMind',
    install_requires=[
       "pytest",
       "torch",
       "numpy",
   ],
)
