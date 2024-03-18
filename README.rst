##########################
hello-world python package
##########################

A `hello-world` equivalent for a python package distribution. Does not provide any functionality.
It only contains the directory structure and the minimum files (as a template) necessary to start 
a complete python package distribution.

----

Repository Structure
====================

**package** (dir)
  The source code. This is the code that will be installed after the `setup.py`

**docs** (dir)
  Documentation directory. Contains both the configuration info to produce the docs and the 
  actual docs after they are gnerated. Currently only html documentation is supported.

**dist** (dir)
  Where the python distribution package will be placed. wheel file and source code.

**test** (dir)
  Well... test directory. 
 
**release.sh** (bash script)
  Checks version and generates a python distribution for the package.

**requirements.txt** (text file)
  Any python package that is required and not in the default python library.

**VERSION** (text file)
  Just holds the version of the package.

**tox.ini** (ini file)
  To automate tasks that require different virtual environments. `dev`, `flake` and `docs` already configured.

**setup.cfg** (python config file)
  Holds metadata for the package. Is read from the setup.py.

**setup.py** (python module)
  Generates the package distribution.

**.gitignore** (text hidden file)
  Not to be ignored.

----

Main Idea
=========
  Use it as stepping stone for each new python package. It already has docs, tox and flake configured and it runs.

----


release.sh
==========
  Generates a distribution, either development or stable for the package.

  .. code-block:: bash

    $ release.sh  -h 

----

Usage (*after git clone*)
=========================
  No need to add extra files. Just edit the existing ones. The existing files/directories are the minimum required to have
  a complete python package. Checkout the directory, change metadata as package-name, authors, version, dependencies, the
  README itself, the docs documentation ... Do not forget to delete the **.git** directory since this will be a new project.


Usage (*package*)
===================
  In order to have a true sample, a package with some kind of functionality has to exist. **package** contains a package 
  that reads a `yaml` file and lets user navigate in the dictionary structure via a CLI.

  .. code-block:: bash

  $ package  # start the cli
   
