from tensorflow.keras import layers, models, Input
from tensorflow.keras.models import Model
from tensorflow.keras.applications import VGG16
from tensorflow.keras.initializers import RandomNormal

def create_CSRNET_model():
    input_shape = (240, 320, 3)
    init = RandomNormal(stddev=0.01, seed=123)

    input_tensor = Input(shape=input_shape)

    # Load full VGG16 without top, using the input_tensor
    full_vgg = VGG16(include_top=False, weights='imagenet', input_tensor=input_tensor)

    # Take all layers except the last 6 layers of full VGG16
    truncated_layers = full_vgg.layers[:-6]
    x = input_tensor
    for layer in truncated_layers[1:]:  # skip the InputLayer at index 0
        x = layer(x)

    x = layers.Conv2D(512, (3, 3), activation='relu', dilation_rate=2,
                      kernel_initializer=init, padding='same')(x)
    x = layers.Conv2D(512, (3, 3), activation='relu', dilation_rate=2,
                      kernel_initializer=init, padding='same')(x)
    x = layers.Conv2D(512, (3, 3), activation='relu', dilation_rate=2,
                      kernel_initializer=init, padding='same')(x)
    x = layers.Conv2D(256, (3, 3), activation='relu', dilation_rate=2,
                      kernel_initializer=init, padding='same')(x)
    x = layers.Conv2D(128, (3, 3), activation='relu', dilation_rate=2,
                      kernel_initializer=init, padding='same')(x)
    x = layers.Conv2D(64, (3, 3), activation='relu', dilation_rate=2,
                      kernel_initializer=init, padding='same')(x)

    output = layers.Conv2D(1, (1, 1), dilation_rate=1,
                           kernel_initializer=init, padding='same')(x)

    model = models.Model(inputs=input_tensor, outputs=output)

    return model

def build_preprocessing_module():
    input_tensor = Input(shape=(240, 320, 4))
    x = layers.Conv2D(64, (3, 3), padding='same', activation='relu')(input_tensor)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(64, (3, 3), padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(3, (1, 1), padding='same', activation='linear')(x)  # Output 3 channels
    return Model(inputs=input_tensor, outputs=x, name="preprocessing_module")

def build_truncated_vgg():
    full_vgg = VGG16(include_top=False, weights='imagenet', input_shape=(240, 320, 3))
    truncated_output = full_vgg.layers[-7].output
    #truncated_output = full_vgg.get_layer('block3_pool').output
    vgg = Model(inputs=full_vgg.input, outputs=truncated_output, name='truncated_vgg16')
    return vgg

def build_backend_model(input_shape):
    init = RandomNormal(stddev=0.01, seed=123)
    inputs = Input(shape=input_shape)
    x = layers.Conv2D(512, (3, 3), activation='relu', dilation_rate=2, kernel_initializer=init, padding='same')(inputs)
    x = layers.Conv2D(512, (3, 3), activation='relu', dilation_rate=2, kernel_initializer=init, padding='same')(x)
    x = layers.Conv2D(512, (3, 3), activation='relu', dilation_rate=2, kernel_initializer=init, padding='same')(x)
    x = layers.Conv2D(256, (3, 3), activation='relu', dilation_rate=2, kernel_initializer=init, padding='same')(x)
    x = layers.Conv2D(128, (3, 3), activation='relu', dilation_rate=2, kernel_initializer=init, padding='same')(x)
    x = layers.Conv2D(64, (3, 3), activation='relu', dilation_rate=2, kernel_initializer=init, padding='same')(x)
    output = layers.Conv2D(1, (1, 1), dilation_rate=1, kernel_initializer=init, padding='same')(x)
    return Model(inputs=inputs, outputs=output, name='backend_model')


class CustomCSRNetModel(models.Model):
    def __init__(self, preprocessing_module, truncated_vgg, backend_model, **kwargs):
        super().__init__(**kwargs)
        self.preprocessing_module = preprocessing_module
        self.truncated_vgg = truncated_vgg
        self.backend_model = backend_model

    def call(self, inputs, training=False):
        input_channels = inputs.shape[-1]  # Use static shape info

        def with_preprocessing():
            x = self.preprocessing_module(inputs, training=training)
            x = self.truncated_vgg(x, training=training)
            return self.backend_model(x, training=training)

        def without_preprocessing():
            x = self.truncated_vgg(inputs, training=training)
            return self.backend_model(x, training=training)

        if input_channels == 4:
            return with_preprocessing()
        elif input_channels == 3:
            return without_preprocessing()
        else:
            raise ValueError(f"Expected input with 3 or 4 channels, but got shape: {inputs.shape}")


def create_modified_CSRNet_model():
    preproc = build_preprocessing_module()
    vgg = build_truncated_vgg()
    backend = build_backend_model(input_shape=(30, 40, 512))  # Output size of block3_pool for input 240x320x3

    return CustomCSRNetModel(preproc, vgg, backend)
