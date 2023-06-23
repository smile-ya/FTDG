# Demo

## Dependencies

Part of the needed dependencies are listed as below for the experiment
```
cuda 8.0
cudnn 5.1

Python 2.7.14
tensorflow-gpu (1.2.1)
Keras (2.1.3)
h5py (2.7.1)
Pillow (5.0.0)
opencv-python
```

To install on Linux(ubuntu)
```
When installing tensorflow 1.2.1, you can download the specified version on PYPI or the website of tensorflow.
pip install tensorflow-gpu
pip install keras
pip install Pillow
pip install h5py
pip install opencv-python
```

## To run

generate adversarial examples for Driving
```
cd Driving
python gen_diff.py [4] 0.5 5 5 Dave_dropout
#meanings of arguments
#python gen_diff.py 
[4] -> the neuron selection strategies
0.5 -> the activation threshold of a neuron
5 -> the number of neurons selected to cover
5 -> the number of times for mutation on each seed
Dave_dropout -> the DL model under test
```

generate adversarial examples for MNIST
```
cd MNIST
python gen_diff.py [4] 0.5 5 5 Model1
#meanings of arguments are the same as above
```

generate adversarial examples for SVHN
```
cd SVHN
python gen_diff.py [4] 0.5 5 5 ModelA
#meanings of arguments are the same as above
```