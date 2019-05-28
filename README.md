# Learning to Route in Similarity Graphs
Code for ICML2019 [paper](http://proceedings.mlr.press/v97/baranchuk19a.html)

# What does it do?
It learns a mapping for vertices in an HNSW graph so as to improve nearest neighbor search and avoid local optima.
![img](https://github.com/dbaranchuk/learning-to-route-for-ann/raw/master/images/toy_task.png)

# What do I need to run it?
* A machine with some CPU (preferably 8+) and a GPU
  * Running with no GPU or less than 4 CPU cores may cause premature senility;
* Some popular Linux x64 distribution
  * Tested on Ubuntu16.04, should work fine on any popular linux64 and even MacOS;
  * Windows and x32 systems may require heavy wizardry to run;
  * When in doubt, use Docker, preferably GPU-enabled (i.e. nvidia-docker)

# How do I run it?
1. Clone or download this repo. `cd` yourself to it's root directory.
2. Grab or build a working python enviromnent. [Anaconda](https://www.anaconda.com/) works fine.
3. Install packages from `requirements.txt`
 * Notably, the code __really__ requires joblib 0.9.4 and pytorch 1.0.0.
 * You will also need jupyter or some other way to work with .ipynb files
4. Run jupyter notebook and open a notebook in `./notebooks/`
 * Before you run the first cell, change `%env CUDA_VISIBLE_DEVICES=#` to an index that you plan to use.
 * First it downloads data from dropbox. You will need up to __50-70Gb__ (Precomputed optimal paths take most of the space. Patch with computing optimal routing on the fly coming soon).
 * Second, defines an experiment setup. We provide one example per dataset:
    * `SIFT100K_dcs256_gcn_size64_routing352_verification16.ipynb` - SIFT100K dataset, 256dcs budget, 128d vectors compressed to 64d, 352 routing dcs and 16 dcs for verification 
    * `DEEP100K_dcs128_gcn_size96_routing120_verification8.ipynb` - DEEP100K dataset, 128dcs budget, 96d vectors, no compression, 120 routing dcs and 8 dcs for verification
    * `GLOVE100K_dcs256_gcn_size75_routing596_verification32.ipynb` - GLOVE100K dataset, 256dcs budget, 300d vectors compressed to 75d, 596 routing dcs and 32 dcs for verification
    * An experiment setup
 * Another time-consuming stage is preparing path_cache. 
   * In[7] in both notebooks. 
   * If the process was interrupted or you suspect something is broken, `!rm -rf {cache_path}` and start over.


# Ways to improve training performance
* Grab a bigger GPU and/or more CPU cores
* Multi-GPU training using torch DataParallel module
