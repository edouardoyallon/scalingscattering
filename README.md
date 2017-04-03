## Scaling The Scattering Transform : Deep Hybrid Networks
This repository contains the experiments found in the paper: https://arxiv.org/abs/1703.08961
<img src="https://s-media-cache-ak0.pinimg.com/564x/d4/22/b5/d422b5ca88c7a0d1035475c216e09c02.jpg" width="300">
### Requirements
In order to run our experiments you will need at minimum the following python packages: pytorch,opencv,pyscatwave package.
The simplest way to install pytorch and opencv is through anaconda. We recommend python 2.7 + anaconda.
The pyscatwave package can be found here https://github.com/edouardoyallon/pyscatwave

### Imagenet
We provide a pre-trained model similar to the one described in the paper. 

To run the trained model of scattering+resnet on imagenet ILSVRC validation set:

1) Make sure you have downloaded at least the validation set of ILSVRC2012 and have it organized by class categories
*Note*: due to problems with pytorch dataset constructors make sure your imagenet directory has no hidden files, or extra directories besides the 1000 ILSVRC categories.. otherwise all the images will be mislabeled
2) Download the model file from  http://www.di.ens.fr/~oyallon/scatter_resnet_10_model.pt7
3) Add this to the imagenet/ directory
4) Run the script main_test.py to evaluate on the ILSVRC validation set specifying --imagenetpath to point to your imagenet directory

Training scripts for imagenet and SLE feature extractor will be added soon

### STL-10
Simply run python main_STL.py script in the STL directory

### CIFAR-10
To run the small sample experiments
Example:

```bash
python main_small_sample_class_normalized.py --model resnet12_8_scat --save "test"  --seed 1 --sampleSize 500 --mul 20
```
