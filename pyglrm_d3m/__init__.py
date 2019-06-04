import atexit
import os
import os.path
import tempfile
import shutil

# In the D3M program we assume that the package is installed as root and run by
# another user.  To avoid making assumptions about this user, including whether
# it will have a home directory for us to use, we will create a temporary
# working directory for Julia based on the one created during installation.

if not os.path.isdir(os.path.expanduser('~/.julia/v0.6/PyCall')) or \
    not os.path.isdir(os.path.expanduser('~/.julia/v0.6/LowRankModels')):
  ''' Create a temporary directory to hold the files needed by PyCall and
  LowRankModels.  We must be sure to set the environment variable JULIA_PKGDIR
  here so that when the Python module "julia" is loaded Julia can find PyCall.
  '''
  temp_julia_libs = tempfile.mkdtemp()
  os.environ["JULIA_PKGDIR"] = temp_julia_libs
  shutil.copytree("/opt/julia_global_package_store/v0.6",
      os.path.join(temp_julia_libs, 'v0.6'))

  # Register function to clean up the temporary files when Python exits.
  def pyglrm_d3m_special_cleanup():
    shutil.rmtree(temp_julia_libs)
  atexit.register(pyglrm_d3m_special_cleanup)


__author__ = 'Cornell'
__version__ = 'v0.1.1'

#__all__ = ('HuberPCA','LowRankImpute')


## Import the actual Pyglrm features.
#from .huber_pca import *
#from pyglrm_d3m.low_rank_imputer import *

