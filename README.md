# Scaling The Scattering Transform
This repository contains the experiments found in the paper: https://arxiv.org/pdf/1703.08961.pdf

### Requirements
In order to run our experiments you will need at minimum the following python packages: pytorch,opencv,pyscatwave package.
The simplest way to install pytorch and opencv is through anaconda. We recommend python 2.7 + anaconda.
The pyscatwave package can be found here https://github.com/edouardoyallon/pyscatwave

### Imagenet
We provide a pre-trained model similar to the one described in the paper. 

To run the trained model of scattering+resnet on imagenet ILSVRC:

1) Make sure you have downloaded at least the validation set of ILSVRC2012 and have it organized by class categories
*Note*: due to problems with pytorch dataset constructors make sure your imagenet directory has no hidden files, or extra directories besides the 1000 ILSVRC categories.. otherwise all the images will be mislabeled
2) Download the model file from  ED PUT LINK HERE
3) Add this to the imagenet/ directory
4) Run the script main_test.py to evaluate on the ILSVRC validation set specifying --imagenetpath to point to your imagenet directory

To train our scattering+ resnet (or variants) model from scratch:
1) make sure you have downloaded the ILSVRC training set and validation set and have it organized by class categories (see note above about extra directories)
2) Run the main.py script making sure to modify the parameters as needed for your setup. Our basic model was trained on 2x 1080 GPUs. 


To apply the SLE as a feature extractor:
1) Download from ED  PUT LINK
2) ED or SERGEY MAKE TOY EXAMPLE

### STL-10
Simply run python main_STL.py script in the STL directory

### CIFAR-10
To run the small sample experiments
