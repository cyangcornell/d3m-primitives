# Pyglrm_d3m

Pyglrm_d3m is a python package for modeling and fitting generalized low rank models (GLRMs), based on the Julia package LowRankModels.jl. As of 01/24/2018 it is compatible with D3M's primitive interface, and has an example of Huber PCA.


## Requirements
- OS: Any linux or MacOS
- Python (2.7 or 3.x)
- Working installation of git
- D3M core package (v2019.1.21, d64ba5f479fe68da02bbaa7268b8495dc90a46b0)
- D3M common-primitives (a983848127e2ee85eb89c114e92bf18de3989970)

This package has been tested on Ubuntu linux and may not work on non-linux OSes.

This package also relies on Julia.  On linux systems this package can install the most recent version of Julia automatically.  The version of Julia distributed by most linux distributions is may not be recent enough to run this package.  We recommend you use the official binary from [the Julia website](https://julialang.org/downloads/).

**Note** If you use the version of Julia installed by this package, you may need to run

```
export PATH=$PATH:$HOME/.julia/julia-903644385b/bin
```

in order to access Julia.


## Installation

### Windows

Windows based installations are not supported yet.

### MacOS Installation

3.  Install the most recent version of Julia (0.6) by following downloading the appropriate installer from [the Julia website](https://julialang.org/downloads/) and following the direction for your operating system on the [instructions page](https://julialang.org/downloads/platform.html).
3.  Check that Julia runs on the command line by running the command ```julia``` on the command line.
3.  Using your choice of ```pip```, ```pip2```, or ```pip3``` depending on the version of Python you intend on using, run the command
    ```
    pip install git+https://gitlab.datadrivendiscovery.org/cyang2/pyglrm_d3m --user
    ```
    
    The installation will get the package via git - you may need to enter you password for gitlab.

### Linux

3.  Note that the default distribution of Julia included in most package managers is not sufficiently up to date to run this package.  We instead using the version of Julia from the Julia website.  The installer for this package can install Julia for you.
3.  Using your choice of ```pip```, ```pip2```, or ```pip3``` depending on the version of Python you intend on using, run the command
    ```
    pip install git+https://gitlab.datadrivendiscovery.org/cyang2/pyglrm_d3m --user
    ```
    
    The installation will get the package via git - you may need to enter you password for gitlab.
3.  If you let pip install Julia, you may need to run the command
    ```
    export PATH=$PATH:$HOME/.julia/julia-903644385b/bin
    ```

## Common Troubleshooting

3.  Segmentation faults

    The underlying software that runs the package compiles itself for one version of Python at a time.  For example, if you install the package using Python 2.7 and then use Python 3.6 you will get a segmentation fault.
    
    If switching between versions of Python is your problem, there is a simple solution.  Each time you switch version of Python first run
    ```
    whereis python
    whereis python3
    ```
    or
    ```
    which python
    which python3
    ```
    to find the absolute path to the version of Python you plan to use.  Then run the following commands in Julia
    
    ```
    ENV["PYTHON"] = "/path/to/python/binary"
    Pkg.build("PyCall")
    exit()
    ```
    
    This should resolve the issue.
    
    Note: specifically, when using Anaconda3 as Python distribution, the absolute path set to `ENV["PYTHON"]` should be the path to Python 3 as well, although both Python and Python 3 are Python 3 distributions.

3.  On linux, after installation "Julia" cannot be found.

    You likely need to run the command
    ```
    export PATH=$PATH:$HOME/.julia/julia-903644385b/bin
    ```
    
## Version History
- v0.1.1 (06-08-2018): Add imputation functionality.
- v0.1.0 (01-24-2018): Initial version with Huber PCA functionality.
