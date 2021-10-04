# Thesis
Thesis research

For this work, we used as base the [Flownet3D in pytorch](https://github.com/hyangwinter/flownet3d_pytorch) and as brief guidence from [Fu et al.](https://pubmed.ncbi.nlm.nih.gov/33129147/) for tweeks and preparation of data.
In [Preprocessing.ipynb](https://github.com/lameski123/prethesis/blob/main/Preprocessing.ipynb) a pipeline of data creation and pre-processing.
In [model.py](https://github.com/lameski123/prethesis/blob/main/model.py) is the tweek of the original Flownet3D. 
In [data.py](https://github.com/lameski123/prethesis/blob/main/data.py) is the data loader. 
In [util.py](https://github.com/lameski123/prethesis/blob/main/util.py) the implementation of functions needed to make the model work properly (these functions are CUDA dependent for more information I would recomend to check [Flownet3D in pytorch](https://github.com/hyangwinter/flownet3d_pytorch)).
In [test_model.py](https://github.com/lameski123/prethesis/blob/main/distError.py) we produced test results on unseen data during training.

# Installation

*tested with python==3.8*

```
git clone https://github.com/lameski123/thesis
git checkout thesis
pip install -r requirements.txt
pip install chamferdist
cd pointnet2
python setup.py install
```

# Train
Training is done using the train.py file. you will need a WandB account to see the results. there are a couple of 
options for loss. see the z`options.py` file for that. 

`python train.py --dataset_path %DATASET_PATH% --loss rigidity`

# Test

Test is done using the test.py file. two arguments are mandatory, data path and the save model path. 

`python test.py --dataset_path %DATASET_PATH% ----model_path %MODEL_PATH%`

# Running on the cluster

we need the cuda exec. to compile the libraries. so we should pull a `devel` image not `runtime`

the code is tested on `pytorch/pytorch:1.9.0-cuda11.1-cudnn8-devel`.


