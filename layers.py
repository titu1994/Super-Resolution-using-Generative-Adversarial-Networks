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


''' Theano Backend function '''

def depth_to_scale(x, scale, output_shape, dim_ordering=K.image_dim_ordering(), name=None):
    ''' Uses phase shift algorithm [1] to convert channels/depth for spacial resolution '''

    import theano.tensor as T

    scale = int(scale)

    if dim_ordering == "tf":
        x = x.transpose((0, 3, 1, 2))
        out_row, out_col, out_channels = output_shape
    else:
        out_channels, out_row, out_col = output_shape

    b, k, r, c = x.shape
    out_b, out_k, out_r, out_c = b, k // (scale * scale), r * scale, c * scale

    out = K.reshape(x, (out_b, out_k, out_r, out_c))

    for channel in range(out_channels):
        channel += 1

        for i in range(out_row):
            for j in range(out_col):
                a = i // scale
                b = j // scale
                d = channel * scale * (j % scale) + channel * (i % scale)

                T.set_subtensor(out[:, channel - 1, i, j], x[:, d, a, b], inplace=True)

    if dim_ordering == 'tf':
        out = out.transpose((0, 2, 3, 1))

    return out


'''
Implementation is incomplete. Use lambda layer for now.
'''

# TODO: Complete SubpixelConvolution2D layer implementation
# class SubpixelConvolution2D(Layer):
#
#     def __init__(self, r):
#         super(SubpixelConvolution2D, self).__init__()
#         self.r = r
#
#     def build(self, input_shape):
#         pass
#
#     def call(self, x, mask=None):
#         x = depth_to_scale(x, self.r, )
#         return x
#
#     def get_output_shape_for(self, input_shape):
#         if K.image_dim_ordering() == "th":
#             b, k, r, c = input_shape
#             return (b, k / (self.r * self.r), r * self.r, c * self.r)
#         else:
#             b, r, c, k = input_shape
#             return (b, r * self.r, c * self.r, k / (self.r * self.r))