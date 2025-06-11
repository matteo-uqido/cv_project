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
    truncated_layers = full_vgg.layers[:-5]
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
