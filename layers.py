from keras.engine.topology import Layer
from keras import backend as K
import itertools

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
                T.set_subtensor(x[:, 0, :, :], x[:, 0, :, :] - 103.939, inplace=True)
                T.set_subtensor(x[:, 1, :, :], x[:, 1, :, :] - 116.779, inplace=True)
                T.set_subtensor(x[:, 2, :, :], x[:, 2, :, :] - 123.680, inplace=True)
            else:
                # No exact substitute for set_subtensor in tensorflow
                # So we subtract an approximate value
                x = x - self.value
            return x


    def get_output_shape_for(self, input_shape):
        return input_shape


class Denormalize(Layer):
    '''
    Custom layer to subtract the outputs of previous layer by 120,
    to normalize the inputs to the VGG and GAN networks.
    '''

    def __init__(self, **kwargs):
        super(Denormalize, self).__init__(**kwargs)

    def build(self, input_shape):
        pass

    def call(self, x, mask=None):
        return (x + 1) * 127.5


    def get_output_shape_for(self, input_shape):
        return input_shape


''' Theano Backend function '''
# TODO: Complete implementation for Tenforflow Backend
def depth_to_scale(input, scale, channels, dim_ordering=K.image_dim_ordering(), name=None):
    ''' Uses phase shift algorithm [1] to convert channels/depth for spacial resolution '''
    import theano.tensor as T

    b, k, row, col = input.shape
    output_shape = (b, channels, row * scale, col * scale)

    out = T.zeros(output_shape)
    r = scale

    for y, x in itertools.product(range(scale), repeat=2):
        out = T.inc_subtensor(out[:, :, y::r, x::r], input[:, r * y + x :: r * r, :, :])

    return out


'''
Implementation is incomplete. Use lambda layer for now.
'''

# TODO: Complete SubpixelConvolution2D layer implementation
class SubpixelConvolution2D(Layer):

    def __init__(self, r, channels):
        super(SubpixelConvolution2D, self).__init__()

        self.r = r
        self.channels = channels

    def build(self, input_shape):
        pass

    def call(self, x, mask=None):
        y = depth_to_scale(x, self.r, self.channels)
        return y

    def get_output_shape_for(self, input_shape):
        if K.image_dim_ordering() == "th":
            b, k, r, c = input_shape
            return (b, self.channels, r * self.r, c * self.r)
        else:
            b, r, c, k = input_shape
            return (b, r * self.r, c * self.r, self.channels)