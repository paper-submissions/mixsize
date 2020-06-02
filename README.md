# Mixsize: Training Convnets With Mixed Image Sizes for Improved Accuracy, Speed and Scale Resiliency

This is a complete training example for Deep Convolutional Networks on various datasets (ImageNet, Cifar10, Cifar100, MNIST).

It is based off [imagenet example in pytorch](https://github.com/pytorch/examples/tree/master/imagenet) with helpful additions such as:
  - Training on several datasets other than imagenet
  - Complete logging of trained experiment
  - Graph visualization of the training/validation loss and accuracy
  - Definition of preprocessing and optimization regime for each model
  - Distributed training
 
 To clone:
 ```
 git clone --recursive https://github.com/paper-submissions/mixsize
 ```
 
This code can be used to replicate results from "Mix & Match: training convnets with mixed image sizes for improved accuracy, speed and scale resiliency"
    
For example, training the resnet44 with mixed sizes example in paper:
```
python main.py --model resnet --dataset cifar10 --save cifar10_mixsize_d -b 64 --model-config "{'regime': 'sampled_D+'}" --epochs 200
```
Then, calibrate for specific size and evaluate using
```
python evaluate.py ./results/cifar10_mixsize_d/checkpoint.pth.tar --dataset cifar10 -b 64 --input-size 32 --calibrate-bn
```

## Pretrained models
Pretrained models (ResNet50, ImageNet) are available [here](https://www.dropbox.com/sh/058gqn562vfspa3/AACBukNaWV0_ElwmqBHdsolGa?dl=0)
    
## Dependencies

- [pytorch](<http://www.pytorch.org>)
- [torchvision](<https://github.com/pytorch/vision>) to load the datasets, perform image transforms
- [pandas](<http://pandas.pydata.org/>) for logging to csv
- [bokeh](<http://bokeh.pydata.org>) for training visualization


## Data
- Configure your dataset path with ``datasets-dir`` argument
- To get the ILSVRC data, you should register on their site for access: <http://www.image-net.org/>


## Model configuration

Network model is defined by writing a <modelname>.py file in <code>models</code> folder, and selecting it using the <code>model</code> flag. Model function must be registered in <code>models/\_\_init\_\_.py</code>
The model function must return a trainable network. It can also specify additional training options such optimization regime (either a dictionary or a function), and input transform modifications.

e.g for a model definition:

```python
class Model(nn.Module):

    def __init__(self, num_classes=1000):
        super(Model, self).__init__()
        self.model = nn.Sequential(...)

        self.regime = [
            {'epoch': 0, 'optimizer': 'SGD', 'lr': 1e-2,
                'weight_decay': 5e-4, 'momentum': 0.9},
            {'epoch': 15, 'lr': 1e-3, 'weight_decay': 0}
        ]

        self.data_regime = [
            {'epoch': 0, 'input_size': 128, 'batch_size': 256},
            {'epoch': 15, 'input_size': 224, 'batch_size': 64}
        ]
    def forward(self, inputs):
        return self.model(inputs)
        
 def model(**kwargs):
        return Model()
```
