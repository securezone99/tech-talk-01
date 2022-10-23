from setuptools import setup

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    name='classifier',
    packages=['classifier'],
    install_requires=requirements
)