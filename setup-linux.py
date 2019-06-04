# setup-linux.py

from os import environ, mkdir, remove
from os.path import exists, expanduser, join
from platform import machine as machine_architecture
from subprocess import check_call
from sys import version_info
import tarfile

# Import urlretrive independent of Python2 vs Python3.
if version_info.major == 2:
  from urllib import urlretrieve
else:
  from urllib.request import urlretrieve

# These are the locations of the official Julia binaries for linux.
address_julia_linux64 = 'https://julialang-s3.julialang.org/bin/linux/x64/0.6/julia-0.6.0-linux-x86_64.tar.gz'
address_julia_linux32 = 'https://julialang-s3.julialang.org/bin/linux/x86/0.6/julia-0.6.0-linux-i686.tar.gz'


# Adds a number to filename in case a file by the same name already exists.
def namefile(filename, suffix):
  if not exists(filename + suffix):
    return filename + suffix
  count = 1
  while exists(filename + str(count) + suffix):
    count = count + 1
  return filename + str(count) + suffix


# Create the folder for Julia and setup file names.
foldername = '/opt/julia_files/'
filename = 'juila-0.6'
filesuffix = '.tar.gz'
if not exists(foldername):
  mkdir(foldername)
filename = namefile(join(foldername, filename), filesuffix)


# Download and extract Julia from the official website.
try:
  if machine_architecture().endswith('64'):
    urlretrieve(address_julia_linux64, filename)
  else:
    urlretrieve(address_julia_linux32, filename)

  tar = tarfile.open(filename)
  tar.extractall(foldername, tar.getmembers())
finally:
  remove(filename)


# Finally, complete installation by placing a symbolic link in /usr/local/bin.
target = join(foldername, 'julia-903644385b/bin/julia')
check_call(['ln', '-s', target, '/usr/local/bin/julia'])
