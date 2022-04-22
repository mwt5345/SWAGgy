from setuptools import setup

with open("README.md",'r') as f:
    long_description = f.read()

setup(
   name='SWAGgy',
   version='0.1',
   description='Implementation of SWAG for axion DM search',
   license="MIT",
   long_description=long_description,
   author='Michael W. Toomey',
   author_email='michael_toomey@brown.edu',
   packages=['swaggy'],
)
