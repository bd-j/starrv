Installation on Odyssey
======

Setup environments
==

* git: follow instructions [here](https://rc.fas.harvard.edu/resources/documentation/software/git-and-github-on-odyssey/) and [here](https://help.github.com/articles/generating-an-ssh-key/) to get ssh based git working

* Download dependencies. In your code directory:

  ```
  git clone git@github.com:dfm/emcee.git;
  git clone git@github.com:bd-j/sedpy.git;
  git clone git@github.com:bd-j/prospector.git;
  git clone git@github.com:bd-j/starrv.git
  ```

* Get an Anaconda environment (here called ``pro``):

  ```
  module purge
  module load python
  conda create -n pro --clone="$PYTHON_HOME"
  ```

* Install dependencies in Anaconda environment:
  ```
  source activate pro
  cd sedpy; python setup.py install
  cd ../emcee; python setup.py install
  cd ../prospector; python setup.py install
  ```

You might want to add some of these to your bashrc
```
module load python
source activate pro #optional
```

Get data and run ``starrv``
==
You need two HDF5 files.  One is at
https://www.dropbox.com/s/tns7x41o9dzznbb/ckc_R10K.h5?dl=0

and the other is at
https://www.dropbox.com/s/4giguokgs41kmp6/culled_libv2_w_mdwarfs_w_unc_w_allc3k.h5?dl=0

or you can copy them from `/n/home02/bdjohnson/regal/starrv/data/`

Move them to
``starrv/data/``
and then we are going to stripe the data directory,
since we are going to have many many cpus hitting the c3k file.
In fact we should probably make copies of that file and then stripe it.
```
cd starrv
lfs setstripe -c 10 data/
```


and then run for a set of stars
```
python fit_broad_lambda.py 0 10 20 1024
```
The numbers are starid_start, starid_end, ncore, niter.  the stars that are run are the integers [starid_start, starid_end).  There are 12 segments per star, so choose the number of stars appropriately for the number of cores.
