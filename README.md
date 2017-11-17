# traffic-sign-recognition
本科毕设，基于Faster-RCNN做的道路交通标志识别（检测与识别）
数据集：德国道路标志数据集GTSDB和TDSRB(http://benchmark.ini.rub.de)

![OUTPUT](output.gif)
For installation, I modified the original Faster-RCNN [README.md](https://github.com/rbgirshick/py-faster-rcnn/blob/master/README.md) file to adapt changes for run this module. Please check below for license and citation information.

### Contents
1. [Requirements: software](#requirements-software)
2. [Requirements: hardware](#requirements-hardware)
3. [Basic installation](#installation-sufficient-for-the-demo)
4. [Beyond the demo: training and testing](#beyond-the-demo-installation-for-training-and-testing-models)
5. [Usage](#usage)

### Requirements: software

1. Requirements for `Caffe` and `pycaffe` (see: [Caffe installation instructions](http://caffe.berkeleyvision.org/installation.html))

    **Note:** Caffe *must* be built with support for Python layers!

    ```make
    # In your Makefile.config, make sure to have this line uncommented
    WITH_PYTHON_LAYER := 1
    # Unrelatedly, it's also recommended that you use CUDNN
    USE_CUDNN := 1
    ```

    You can see the sample [Makefile.config](caffe-fast-rcnn/Makefile.config) avialable with this repository. It uses conda with GPU support. You need to modify this file to suit your hardware configuration.
2. Python packages you might not have: `cython`, `python-opencv`, `easydict`
3. [Optional] MATLAB is required for **official** PASCAL VOC evaluation only. The code now includes unofficial Python evaluation code.

### Requirements: hardware

1. For training smaller networks (ZF, VGG_CNN_M_1024) a good GPU (e.g., Titan, K20, K40, ...) with at least 3G of memory suffices
2. For training Fast R-CNN with VGG16, you'll need a K40 (~11G of memory)
3. For training the end-to-end version of Faster R-CNN with VGG16, 3G of GPU memory is sufficient (using CUDNN)

### Installation (sufficient for the demo)

1. Clone the Faster R-CNN repository
    ```Shell
    git clone https://github.com/shizhenjun/traffic-sign-recognition.git
    ```

2. Build the Cython modules
    ```Shell
    cd $FRCN_ROOT/detection and classification/lib
    make
    ```

3. Build Caffe and pycaffe
    ```Shell
    cd $FRCN_ROOT/caffe-fast-rcnn
    # Now follow the Caffe installation instructions here:
    #   http://caffe.berkeleyvision.org/installation.html

    # If you're experienced with Caffe and have all of the requirements installed
    # and your Makefile.config in place, then simply do:
    make -j8 && make pycaffe
    ```

### Beyond the demo: installation for training and testing models

Before starting, you need to download the traffic sign datasets from [German Traffic Signs Datasets](http://benchmark.ini.rub.de/?section=gtsdb&subsection=dataset). In this implementation, the training and test datasets that were used for the competition ( [training data set](http://benchmark.ini.rub.de/Dataset_GTSDB/TrainIJCNN2013.zip) (1.1 GB), [test data set](http://benchmark.ini.rub.de/Dataset_GTSDB/TestIJCNN2013.zip) (500 MB) ) is used.

Here, the main goal is to enable Faster R-CNN to detect and classify traffic sign. So, model performance evaluation in test dataset was not carried out. The downloaded test dataset was only used for visual testing. After the dataset is downloaded, prepare the following directory structure. The training zip file contains the following files
- folders
- images (00000.ppm, 00001.ppm...., 00599.ppm)
- gt.txt

Copy all the images into Images directory as shown below. Rename gt.txt as train.txt and keep both gt.txt and train.txt as shown below. 

##### Format Your Dataset
At first, the dataset must be well organzied with the required format.
```
GTSDB
|-- Annotations
    |-- gt.txt (Annotation files)
|-- Images
    |-- *.ppm (Image files)
|-- ImageSets
    |-- train.txt
```

##### Download pre-trained ImageNet models

Pre-trained ImageNet models can be downloaded for the three networks described in the paper: ZF and VGG16.

```Shell
cd $FRCN_ROOT/detection and classification
./data/scripts/fetch_imagenet_models.sh
```
VGG16 comes from the [Caffe Model Zoo](https://github.com/BVLC/caffe/wiki/Model-Zoo), but is provided here for your convenience.
ZF was trained at MSRA.

### Usage
This implementation is tested only for approximate joint training.

To train and test a TSR Faster R-CNN detector using the **approximate joint training** method, use `experiments/scripts/faster_rcnn_end2end.sh`.
Output is written underneath `$FRCN_ROOT/output`.

```Shell
cd $FRCN_ROOT/detection and classification
./experiments/scripts/faster_rcnn_end2end.sh [GPU_ID] [NET] [--set ...] [DATASET]
# GPU_ID is the GPU you want to train on
# NET in {ZF, VGG_CNN_M_1024, VGG16} is the network arch to use
# --set ... allows you to specify fast_rcnn.config options, e.g.
# --set EXP_DIR seed_rng1701 RNG_SEED 1701
# DATASET to be used for training
```
Example script to train ZF model:
```Shell
cd $FRCN_ROOT/detection and classification
./experiments/scripts/faster_rcnn_end2end.sh 0 ZF gtsdb
```
Trained Fast R-CNN networks are saved under:
```
output/<experiment directory>/<dataset name>/
# Example: output/faster_rcnn_end2end/gtsdb_train
```
Test outputs are saved under:
```
output/<experiment directory>/<dataset name>/<network snapshot name>/
```