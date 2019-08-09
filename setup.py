
    
from __future__ import print_function

from os import environ, geteuid
from os.path import realpath, expanduser
from setuptools import setup
from setuptools.command.develop import develop
from subprocess import check_call
from sys import executable as python_binary_path, platform, version_info
import tarfile

# Import urlretrieve independent of Python2 or Python3.
if version_info.major == 2:
  from urllib import urlretrieve
else:
  from urllib.request import urlretrieve


# This defines custom operations that will be run at the end of installation.
class PostDevelopCommands(develop):
  def run(self):
    develop.run(self)


# Installation information for use by pip.
setup(
  name='pyglrm_d3m',
  version='v0.1.1',
  description='Run low-rank imputer, high-rank imputer, and huber-pca in Python.',
  author='Chengrun Yang, Matthew Zalesak, Anya Chopra & Madeleine Udell',
  author_email='cy438@cornell.edu',
  url='https://gitlab.datadrivendiscovery.org/cyang2/pyglrm_d3m',
  include_package_data=True,
  install_requires=["numpy", "d3m"],
  setup_requires=[],
  package_data={'' : ['README.md']},
  license='MIT',
  packages=['pyglrm_d3m'], 
  classifiers=[
    'Development Status :: 4 - Beta',
    'Intended Audience :: Developers',
    'License :: OSI Approved :: MIT License',
    'Operating System :: OS Independent',
    'Programming Language :: Python',
    'Topic :: Low Rank Models',
    'Topic :: Software Development :: Libraries :: Python Modules'],
  cmdclass={'develop': PostDevelopCommands},
  entry_points = {
    'd3m.primitives':
      [
       'feature_extraction.huber_pca.Cornell = pyglrm_d3m.huber_pca:HuberPCA',
       'data_preprocessing.low_rank_imputer.Cornell = pyglrm_d3m.low_rank_imputer:LowRankImputer',
       'collaborative_filtering.high_rank_imputer.Cornell = pyglrm_d3m.high_rank_imputer:HighRankImputer'],
      }
  )


