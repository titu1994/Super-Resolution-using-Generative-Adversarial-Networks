# Super Resolution using Generative Adversarial Networks

This is an implementation of the SRGAN model proposed in the paper [Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial
Network](https://arxiv.org/pdf/1609.04802v2.pdf) in Keras. Note that this project is a work in progress.

A simplified view of the model can be seen as below: <br>
<img src="https://github.com/titu1994/Super-Resolution-using-Generative-Adversarial-Networks/blob/master/architecture/SRGAN-simple-architecture.jpg?raw=true" width=40% height=50%>

# Implementation Details

The SRGAN model is built in stages within models.py. Initially, only the SR-ResNet model is created, to which the VGG network is appended 
to create the pre-training model. The VGG weights are freezed as we will not update these weights.

In the pre-train mode:

1. The discriminator model is not attached to the entire network. Therefore it is only the SR + VGG model that will be pretrained.
2. During pretraining, the VGG perceptual losses will be used to train (using the ContentVGGRegularizer) and TotalVariation loss (using 
TVRegularizer). No other loss (MSE, Binary crosss entropy, Discriminator) will be applied.
3. Content Regularizer loss will be applied to the VGG Convolution 2-2 layer

In the full train mode:

1. The discriminator model is attached to the entire network. Therefore it creates the SR + GAN + VGG model (SRGAN)
2. Discriminator loss is also added to the VGGContentLoss and TVLoss.
3. Content regularizer loss is applied to the VGG Convolution 5-3 layer. (VGG 16 is used instead of 19 for now)

# Usage
Currently, models.py contains most of the code to train and create the models. To use different modes, uncomment the parts of the code that you need.

Note the difference between the *_network objects and *_model objects. 
- The *_network objects reflect to the helper classes which create and manage the Keras models, load and save weights and 
set whether the model can be trained or not.
- The *_models objects represent the underlying Keras model. 

To pretrain the network:
```
srgan_network = SRGANNetwork(img_width=32, img_height=32, batch_size=1)
srgan_network.pre_train_network(iamges_path, nb_epochs=1, nb_images=50000)
```

To just create the pretrain model:
```
srgan_network = SRGANNetwork(img_width=32, img_height=32, batch_size=1)
srgan_model = srgan_network.build_srgan_pretrain_network()

# Plot the model
from keras.utils.visualize_util import plot
plot(srgan_model, to_file='SRGAN.png', show_shapes=True)
```

To train the full network (Does NOT work properly right now, Discriminator is not correctly trained):
```
srgan_network = SRGANNetwork(img_width=32, img_height=32, batch_size=1)
srgan_network.train_full_network(coco_path, nb_images=80000, nb_epochs=10)
```

# Drawbacks:
- Since keras has internal checks for batch size, we have to bypass an internal keras check called check_array_length(),
which checks the input and output batch sizes. As we provide the original images to Input 2 and Input 3, batch size doubles. 
This causes an assertion error in internal keras code. For now, we rewrite the fit logic of keras in keras_training_ops.py and use 
the bypass fit functions.
- There is no way to train the discriminator model when it is attached to the full model for now. This is because the batch size is doubling 
by the end of the network. Due to this, we cannot apply binary cross entropy loss for the discriminator network (full model input is of 
batch size x, discriminator output batch size os 2x). 
- For some reason, the Deconvolution networks are not learning the upscaling function properly. This causes grids to form throughout the 
upscaled image. This is possibly due to the large (4x) upscaling procedure.

# Plans
The codebase is currently very chaotic, since I am focusing on correct implementation before making the project better. Therefore,
expect the code to drastically change over commits. 

Some things I am currently trying out:
- Training the discriminator model seperately, getting the updates and passing them onto the full model. This involves transfering weights,
updates and using more internal code of keras.
- Replacing the 2 deconv layers for a Sub-Pixel Convolution layer. Note that since this is different from the original implementation,
I will use a different set of build and fit methods. 
- Convert this model into an auto encoder style upsampling model. Such models learn very fast and as the discussion below shows, the outputs 
are almost exactly the same. Note that I will be using a different set of functions to create such a model, as it deviates from the original model.

# Discussion
There is an ongoing discussion at https://github.com/fchollet/keras/issues/3940 where I detail some of the outputs and attempts to correct 
the errors.

