# prethesis
Pre-thesis research

For this work, we used as base the [Flownet3D in pytorch](https://github.com/hyangwinter/flownet3d_pytorch) and as brief guidence from [Fu et al.](https://pubmed.ncbi.nlm.nih.gov/33129147/) for tweeks and preparation of data. 
In [Preprocessing.ipynb](https://github.com/lameski123/prethesis/blob/main/Preprocessing.ipynb) a pipeline of data creation and pre-processing is provided using SITK and ITK as main libraries for data deformation.
In [model.py](https://github.com/lameski123/prethesis/blob/main/model.py) is the tweek of the original Flownet3D 
In [data.py](https://github.com/lameski123/prethesis/blob/main/data.py) is the data loader (currently the loader is changed specifically for testing purposes, but with small changes in the class constructor one can make it ready for training)
In [util.py](https://github.com/lameski123/prethesis/blob/main/util.py) the implementation of functions needed to make the model work properly (these functions are CUDA dependent for more information I would recomend to check [Flownet3D in pytorch](https://github.com/hyangwinter/flownet3d_pytorch))
In [test_model.py](https://github.com/lameski123/prethesis/blob/main/test_model.py) we produced test results on unseen data during training.

Here is an example registration on our data:
![alt text](https://drive.google.com/file/d/1wcsb_EG7Eodvju4BsVTB4nMf0tdg4X2G/view?usp=sharing)
yellow-source; blue-registration; red-target


