### Deep Image Homography Estimation

#### Abstract

This project implements the 2016 paper [_Deep Image Homography Estimation_](https://arxiv.org/pdf/1606.03798.pdf) by DeTone, Malisiewicz, and Rabinovich. 
We create an image homography training set by randomly warping the dataset presented in the 2015 paper [_Microsoft COCO: Common Objects in Context_](https://arxiv.org/pdf/1405.0312.pdf) 
by Lin et al. We then architect and train a deep convolutional neural network to learn how to compute a 3x3 homography matrix given an image pair. 

#### Installation
This procedure was tested on Ubuntu 16.04 and Mac OS X 10.11.6 (El Capitan). GPU-accelerated training is supported on Ubuntu only.

Prerequisites: Install Python package dependencies using [my instructions.](https://github.com/alexhagiopol/deep_learning_packages) Then, activate the environment:

    source activate deep-learning

Optional, but recommended on Ubuntu: Install support for NVIDIA GPU acceleration with CUDA v8.0 and cuDNN v5.1:

    wget https://www.dropbox.com/s/08ufs95pw94gu37/cuda-repo-ubuntu1604-8-0-local-ga2_8.0.61-1_amd64.deb
    sudo dpkg -i cuda-repo-ubuntu1604-8-0-local-ga2_8.0.61-1_amd64.deb
    sudo apt-get update
    sudo apt-get install cuda
    wget https://www.dropbox.com/s/9uah11bwtsx5fwl/cudnn-8.0-linux-x64-v5.1.tgz
    tar -xvzf cudnn-8.0-linux-x64-v5.1.tgz
    cd cuda/lib64
    export LD_LIBRARY_PATH=`pwd`:$LD_LIBRARY_PATH
    cd ..
    export CUDA_HOME=`pwd`
    sudo apt-get install libcupti-dev
    pip install --ignore-installed --upgrade https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow_gpu-1.1.0-cp35-cp35m-linux_x86_64.whl

Clone the deep_homography_estimation repo:

    git clone https://github.com/alexhagiopol/deep_homography_estimation
    cd deep_homography_estimation

Download the [MSCOCO Dataset](http://mscoco.org/):
    
    mkdir MSCOCO
    cd MSCOCO    
    wget http://msvocds.blob.core.windows.net/coco2014/train2014.zip
    wget http://msvocds.blob.core.windows.net/coco2014/val2014.zip
    wget http://msvocds.blob.core.windows.net/coco2014/test2014.zip
    wget http://msvocds.blob.core.windows.net/coco2015/test2015.zip
    unzip *.zip
    tar -xvzf traffic-signs-data.tar.gz
    rm -rf traffic-signs-data.tar.gz
