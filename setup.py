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


# Strings for installation information.
msg_configurejulia = 'Ensuring Julia has required packages installed...'
msg_juliaerror = 'Failed to run Julia.  Please ensure that Julia is properly installed on this\nsystem and then rerun the installer.'
msg_needjulia = 'Julia not found on this system.'
msg_needjulia_auto = 'We will try to automatically install and configure Julia for use with this\npackage.'
msg_install_failed = 'This package requies Julia and the installer does not know how to install Julia\non this system. Please install Julia manually and then rerun this installation.'
msg_postinstall = 'Setup has installed Julia for use by all users.'


# Installs Julia on the system, returning an error upon failure.
def installjulia():
  try:  # Set error names by Python version.
    RunTimeError
  except NameError:
    RunTimeError = RuntimeError
    
  print(msg_needjulia_auto)
  if platform.startswith('linux'):
    check_call([python_binary_path, 'setup-linux.py']) 
    print(msg_postinstall)
  else:
    raise RunTimeError(msg_install_failed)


# Install packages PyCall and LowRankModels in the Julia library.
def configurejulia():
  print(msg_configurejulia)
  
  try:  # Set error names by Python version.
    FileNotFoundError
  except NameError:
    FileNotFoundError = OSError
  try:
    RunTimeError
  except NameError:
    RunTimeError = RuntimeError
  
  # Now we figure out where Julia packages will be stored.
  if geteuid() == 0:
    package_path = "/opt/julia_global_package_store"
  else:
    package_path = expanduser("~/.julia")
  
  # Test if Julia is installed.
  try:
    check_call(['julia', '--version'])
  except FileNotFoundError:
    print(msg_needjulia) # If it fails to find Julia, install Julia.
    installjulia()
  
  # Julia package installation.
  try:
    check_call(['julia', 'setup.jl', python_binary_path, realpath('setup.sh'),
        package_path])
    return
  except FileNotFoundError:
    raise RunTimeError(msg_juliaerror)
    

# This defines custom operations that will be run at the end of installation.
class PostDevelopCommands(develop):
  def run(self):
    configurejulia()
    develop.run(self)


# Installation information for use by pip.
setup(
  name='pyglrm_d3m',
  version='v0.1.1',
  description='Run LowRankModels.jl in Python.',
  author='Chengrun Yang, Matthew Zalesak, Anya Chopra & Madeleine Udell',
  author_email='cy438@cornell.edu',
  url='https://gitlab.datadrivendiscovery.org/cyang2/pyglrm_d3m',
  include_package_data=True,
  install_requires=["julia==0.4.1", "numpy", "d3m"],
  setup_requires=["julia==0.4.1"],
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
