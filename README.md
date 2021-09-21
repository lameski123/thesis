# Thesis
Thesis research

For this work, we used as base the [Flownet3D in pytorch](https://github.com/hyangwinter/flownet3d_pytorch) and as brief guidence from [Fu et al.](https://pubmed.ncbi.nlm.nih.gov/33129147/) for tweeks and preparation of data.\n
In [Preprocessing.ipynb](https://github.com/lameski123/prethesis/blob/main/Preprocessing.ipynb) a pipeline of data creation and pre-processing.
In [model.py](https://github.com/lameski123/prethesis/blob/main/model.py) is the tweek of the original Flownet3D. \n
In [data.py](https://github.com/lameski123/prethesis/blob/main/data.py) is the data loader. \n
In [util.py](https://github.com/lameski123/prethesis/blob/main/util.py) the implementation of functions needed to make the model work properly (these functions are CUDA dependent for more information I would recomend to check [Flownet3D in pytorch](https://github.com/hyangwinter/flownet3d_pytorch)).\n
In [test_model.py](https://github.com/lameski123/prethesis/blob/main/distError.py) we produced test results on unseen data during training.\n


