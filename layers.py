import tensorflow as tf
import keras

class ReLU6(keras.layers.Layer):
    def __init__(self):
        super(ReLU6,self).__init__()
        self.relu6 = keras.layers.ReLU(max_value=6.0)

    def build(self, input_shape):
        super(ReLU6,self).build(input_shape)

    def call(self, x):
        return self.relu6(x)

    def compute_output_shape(self, input_shape):
        return input_shape


class HSwish(keras.layers.Layer):
    def __init__(self):
        super(HSwish,self).__init__()
        self.relu6 = ReLU6()

    def build(self, input_shape):
        super(HSwish,self).build(input_shape)

    def call(self, x):
        hsigmoid = self.relu6(x + 3.0) / 6.0
        return x * hsigmoid

    def compute_output_shape(self, input_shape):
        return input_shape


class Identify(keras.layers.Layer):
    def __init__(self):
        super(Identify,self).__init__()

    def build(self, input_shape):
        super(Identify, self).build(input_shape)

    def call(self, x):
        return x

    def compute_output_shape(self, input_shape):
        return input_shape

class Bneck(keras.layers.Layer):
    def __init__(self, kernel_size, input_size, expand_size, output_size, activation,stride,use_se):
        super(Bneck,self).__init__()
        self.use_se = use_se
        self.stride = stride
        self.output_size = output_size

        self.conv1 = keras.layers.Conv2D(filters=expand_size, kernel_size=1,strides=1,use_bias=False)
        self.bn1 = keras.layers.BatchNormalization()
        self.nonlinear1 = activation

        padding = kernel_size // 2
        self.pad = keras.layers.ZeroPadding2D(padding=padding)
        self.conv2 = keras.layers.DepthwiseConv2D(kernel_size=kernel_size,strides=stride, use_bias=False)
        self.bn2 = keras.layers.BatchNormalization()
        self.nonlinear2 = activation

        self.conv3 = keras.layers.Conv2D(kernel_size=1,strides=1,filters=output_size,use_bias=False)
        self.bn3 = keras.layers.BatchNormalization()

        self.short_cut = Identify()
        if stride == 1 and input_size != output_size:
            self.short_cut = keras.Sequential([keras.layers.Conv2D(kernel_size=1,strides=1,filters=output_size),
                                               keras.layers.BatchNormalization()])

    def build(self, input_shape):
        super(Bneck,self).build(input_shape)

    def call(self, x):
        out = self.nonlinear1(self.bn1(self.conv1(x)))
        out = self.nonlinear2(self.bn2(self.conv2(self.pad(out))))
        if self.use_se:
            out = self.use_se(out)
        out = self.bn3(self.conv3(out))


        out = out + self.short_cut(x) if self.stride ==1 else out
        return out

    def compute_output_shape(self, input_shape):
        if self.stride == 1:
            return (input_shape[0],input_shape[1],input_shape[2],self.output_size)
        else:
            return (input_shape[0],input_shape[1]//2, input_shape[2]//2,self.output_size)


class SEModule(keras.layers.Layer):
    def __init__(self, input_size, reduction=4):
        super(SEModule,self).__init__()
        self.se = keras.Sequential([keras.layers.GlobalAveragePooling2D(),
                                    keras.layers.Dense(input_size // reduction),
                                    keras.layers.ReLU(),
                                    keras.layers.Dense(input_size),
                                    HSwish()])

    def build(self, input_shape):
        super(SEModule,self).build(input_shape)

    def call(self, x):
        out = self.se(x)
        out = keras.layers.Reshape((1,1,-1))(out)
        return x * out
    def compute_output_shape(self, input_shape):
        return input_shape


