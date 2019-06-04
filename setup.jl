# setup.jl

# First, suppress warnings since they really bother people.  I hope this
# solution works long term.
logging(DevNull; kind=:warn)

# To make the package easy to install globally we create a `template library'
# for Julia.  This will include the packages PyCall, LowRankModels, and all
# of their dependencies.  When individual users load pyglrm_d3m we will copy
# this prepared library into a temporary directory.  Setting the environment
# variable JULIA_PKGDIR tells Julia to use a library directory other
# than ~/.julia.
store_name = ARGS[3]
ENV["JULIA_PKGDIR"] = store_name
if !isdir(store_name)
  mkdir(store_name)
end
Pkg.init()
Pkg.update()

function ensure(package::String)  # Install package if not installed.
  if typeof(Pkg.installed(package)) == Void
    Pkg.add(package)
  end
end


# Ensure that Julia is configured with the necessary packages.
ENV["PYTHON"] = ARGS[1] # Setup using "current" version of Python.
ensure("LowRankModels")
ensure("NullableArrays")
ensure("FactCheck")
ensure("PyCall")

# check out the lates GLRM
Pkg.checkout("LowRankModels")


# This sets the version of the packages that are used.  In the long term it
# would be better to use built in Julia commands.  However, here we pin the
# versions to specific git commit numbers and thus perform this manually. If
# we want to pin the versions to a release number then we can perform this
# from within Julia.

shell_file = ARGS[2]                        # <- We are given shell script name
directory = store_name * "/v0.6"     # <- Add Julia version number
run(`bash $shell_file $directory`)          # <- Script selects commit numbers.


# Finally, we build the packages now since PyCall needs to be built with
# ENV["PYTHON"] set and LowRankModels takes so long that it may cause users to
# think the system is not working upon first use.

Pkg.build("LowRankModels") ; using LowRankModels
Pkg.build("PyCall") # <- DO NOT CALL using PyCall from Julia, only from Python.
