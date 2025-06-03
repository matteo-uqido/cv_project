try:
    from tensorflow.keras import layers, models, Input
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import Dense
    from tensorflow.keras.applications import VGG16
    from tensorflow.keras.applications import vgg16, vgg19, resnet50, xception
    from tensorflow.keras.initializers import RandomNormal
    print("TensorFlow imports succeeded.")
except ImportError as e:
    print("TensorFlow import failed:", e)


#this file creates the models we are going to employ for crowd counting 
def create_chosen_model(model_type: str):
    model_type = model_type.lower()
    
    if model_type == 'resnet50':
        return create_resnet50_model()
    elif model_type == 'vgg16':
        return create_vgg16_model()
    elif model_type == 'vgg19':
        return create_vgg19_model()
    elif model_type == 'xception':
        return create_xception_model()
    elif model_type == 'modified_csrnet':
        return create_modified_CSRNet_model()
    elif model_type == 'csrnet':
        return create_CSRNET_model()
    else:
        raise ValueError(f"Unsupported model_type: {model_type}")


def create_resnet50_model():
    base_model = resnet50.ResNet50(
        weights='imagenet',  
        include_top=False,  
        input_shape=(224, 224, 3),  
        pooling='avg', 
    )
    
    x = base_model.output  
    x = Dense(1024, activation='relu')(x)  
    predictions = Dense(1, activation='linear')(x)  
    model = Model(inputs=base_model.input, outputs=predictions)

    k = -7
    for layer in model.layers[:k]:
        layer.trainable = False
    print('Trainable:')
    for layer in model.layers[k:]:
        print(layer.name)
        layer.trainable = True

    return model
def create_vgg16_model():
    base_model = vgg16.VGG16(
        weights='imagenet',  
        include_top=False,  
        input_shape=(224, 224, 3),  
        pooling='avg', 
    )
    
    x = base_model.output  
    x = Dense(1024, activation='relu')(x)  
    predictions = Dense(1, activation='linear')(x)  
    model = Model(inputs=base_model.input, outputs=predictions)

    k = -7
    for layer in model.layers[:k]:
        layer.trainable = False
    print('Trainable:')
    for layer in model.layers[k:]:
        print(layer.name)
        layer.trainable = True

    return model
def create_vgg19_model():
    base_model = vgg19.VGG19(
        weights='imagenet',  
        include_top=False,  
        input_shape=(224, 224, 3),  
        pooling='avg', 
    )
    
    x = base_model.output  
    x = Dense(1024, activation='relu')(x)  
    predictions = Dense(1, activation='linear')(x)  
    model = Model(inputs=base_model.input, outputs=predictions)

    k = -7
    for layer in model.layers[:k]:
        layer.trainable = False
    print('Trainable:')
    for layer in model.layers[k:]:
        print(layer.name)
        layer.trainable = True

    return model
def create_xception_model():
    base_model = xception.Xception(
        weights='imagenet',  
        include_top=False,  
        input_shape=(224, 224, 3),  
        pooling='avg', 
    )
    for layer in base_model.layers:
        layer.trainable = False

    x = base_model.output  
    x = Dense(1024, activation='relu')(x)  
    predictions = Dense(1, activation='linear')(x)  
    model = Model(inputs=base_model.input, outputs=predictions)

    return model

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

def create_modified_CSRNet_model():
    data_processor_shape = (240, 320, 4)
    input_tensor = layers.Input(shape=data_processor_shape)
    x = preprocessing_module(input_tensor)  # Output: (240, 320, 3)

    # Load full VGG16
    full_vgg = VGG16(include_top=False, weights='imagenet', input_shape=(240, 320, 3))

    # Truncate VGG16 to keep only up to block3_pool (downsampling by 1/8)
    truncated_output = full_vgg.get_layer('block3_pool').output
    vgg = Model(inputs=full_vgg.input, outputs=truncated_output,name='truncated_vgg16')

    # Freeze VGG16 layers
    for layer in vgg.layers:
        layer.trainable = False

    x = vgg(x)  # Apply truncated VGG to preprocessed input

    # CSRNet backend
    init = RandomNormal(stddev=0.01, seed=123)
    x = layers.Conv2D(512, (3, 3), activation='relu', dilation_rate=2, kernel_initializer=init, padding='same')(x)
    x = layers.Conv2D(512, (3, 3), activation='relu', dilation_rate=2, kernel_initializer=init, padding='same')(x)
    x = layers.Conv2D(512, (3, 3), activation='relu', dilation_rate=2, kernel_initializer=init, padding='same')(x)
    x = layers.Conv2D(256, (3, 3), activation='relu', dilation_rate=2, kernel_initializer=init, padding='same')(x)
    x = layers.Conv2D(128, (3, 3), activation='relu', dilation_rate=2, kernel_initializer=init, padding='same')(x)
    x = layers.Conv2D(64, (3, 3), activation='relu', dilation_rate=2, kernel_initializer=init, padding='same')(x)
    output = layers.Conv2D(1, (1, 1), dilation_rate=1, kernel_initializer=init, padding='same')(x)

    model = models.Model(inputs=input_tensor, outputs=output)
    return model

def preprocessing_module(input_tensor):
    x = layers.Conv2D(64, (3, 3), padding='same', activation='relu')(input_tensor)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(64, (3, 3), padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(3, (1, 1), padding='same', activation='linear')(x)  # Output 3 channels
    return x

