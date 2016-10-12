from keras.engine.topology import Layer
from keras import backend as K

class Normalize(Layer):
    '''
    Custom layer to subtract the outputs of previous layer by 120,
    to normalize the inputs to the VGG and GAN networks.
    '''

    def __init__(self, type="vgg", value=120, **kwargs):
        super(Normalize, self).__init__(**kwargs)
        self.type = type
        self.value = value

    def build(self, input_shape):
        pass

    def call(self, x, mask=None):
        if self.type == "gan":
            return x / self.value
        else:
            if K.backend() == "theano":
                import theano.tensor as T
                T.set_subtensor(x[:, 0, :, :], x[:, 0, :, :] - 103.939)
                T.set_subtensor(x[:, 1, :, :], x[:, 1, :, :] - 116.779)
                T.set_subtensor(x[:, 2, :, :], x[:, 2, :, :] - 123.680)
            else:
                # No exact substitute for set_subtensor in tensorflow
                # So we subtract an approximate value
                x = x - self.value
            return x


    def get_output_shape_for(self, input_shape):
        return input_shape

class SubpixelUpscale(Layer):

    def __init__(self, r=2):
        super(SubpixelUpscale, self).__init__()

        self.r = r

    def build(self, input_shape):
        pass

    def call(self, x, mask=None):
        pass

    def get_output_shape_for(self, input_shape):
        pass