# Deep Learning Final Project
> Alexander Powers
## An Exploration of Multi-Task Learning
This project explores different network architectures that leverage weight sharing to improve performance on multiple tasks.
    
### The Problem (CIFAR100)
The CIFAR100 dataset consists of RGB images, fine labels(100 classes), and coarse labels(20 classes). Each fine label class is a proper subset of a coarse label class (i.e. one fine label can't have two coarse labels and vice versa).
### Architectures to be trained
#### 1) Independent networks (the control architecture)
```text
input_image --> conv_layers --> fc_layers --> fine_label       
input_image --> conv_layers --> fc_layers --> coarse_label
```       
#### 2) Hard parameter sharing in convolutional layers
```text
                           /--> fc_layers --> fine_label
input_image --> conv_layers      
                           \--> fc_layers --> coarse_label 
```
#### 3) Using coarse label output as weights
```text
input_image   ---->   conv_layers   ---->   fc_layers_1
                                                      \                
                                                    concat -> fc_layers_2  -> fine_label
                                                      /
input_image -> conv_layers -> fc_layers -> coarse_label
``` 
#### 4) Using fine label output as weights
```text
input_image   ---->   conv_layers   ---->   fc_layers_1
                                                      \                
                                                    concat -> fc_layers_2  -> coarse_label
                                                      /
input_image -> conv_layers -> fc_layers -> fine_label
``` 
#### 5) Combination of 2 & 3
```text
                         / -------------> fc_layers_1
                        /                           \
input_image -> conv_layers                        concat -> fc_layers_2 -> fine_label
                        \                           /
                         \-> fc_layers -> coarse_label 
```
#### 6) Combination of 2 & 4
```text
                         / -------------> fc_layers_1
                        /                           \
input_image -> conv_layers                        concat -> fc_layers_2 -> coarse_label
                        \                           /
                         \-> fc_layers -> fine_label 
```
