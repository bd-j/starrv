Installation on Odyssey
======

Setup environments
==

* git: follow instructions [here](https://rc.fas.harvard.edu/resources/documentation/software/git-and-github-on-odyssey/) and [here](https://help.github.com/articles/generating-an-ssh-key/) to get ssh based git working

* Download dependencies. In your code directory:

  ```bash
  git clone git@github.com:dfm/emcee.git;
  git clone git@github.com:bd-j/sedpy.git;
  git clone git@github.com:bd-j/prospector.git;
  git clone git@github.com:bd-j/starrv.git
  ```

* Get an Anaconda environment (here called ``pro``):

  ```bash
  module purge
  module load python
  conda create -n pro --clone="$PYTHON_HOME"
  ```

* Install dependencies in Anaconda environment:
  ```bash
  source activate pro
  cd sedpy; python setup.py install
  cd ../emcee; python setup.py install
  cd ../prospector; python setup.py install
  ```

You might want to add some of these to your bashrc
```bash
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
since we are going to have many many cpus hitting the c3k file every tenth of second or so.
In fact we should probably make copies of that file as well.
```bash
cd starrv
lfs setstripe -c 10 data/
for ((i=1; i<=10; i++));
  do cp "data/ckc_R10K.h5" "data/ckc_R10K_${i}.h5";
done
```
The last command takes a lot of time (and disk space) so it's probably best to just use the files in my data directory.

and then run for a set of stars
```bash
python fit_broad_lambda.py 0 10 20 \
       --niter=1024 --verbose=False  \
       --libname=data/ckc_R10K_1.h5
```
The numbers are starid_start, starid_end, ncore.  the stars that are run are the integers [starid_start, starid_end).  There are 12 segments per star, so choose the number of stars appropriately for the number of cores.  The run time per segment is ~1 hour * niter/512 * nwalkers/64 on my laptop - multiply by 5 for AMD cores.
