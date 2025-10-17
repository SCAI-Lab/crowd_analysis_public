from setuptools import setup
import os, sys

sys.path.append(os.path.dirname(__file__))

setup(
    name='crowdbot_data',
    version=1.0,
    description='Minimal package to retrieve data in Crowdbot data convention',
    author='Dominik Wojcikiewicz',
    author_email='dwojcikie@ethz.ch',
    packages=[
        'crowdbot_data',
    ],
)
