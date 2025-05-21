try:
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import Dense
    from tensorflow.keras.applications import vgg16, vgg19, resnet50, xception
    print("TensorFlow imports succeeded.")
except ImportError as e:
    print("TensorFlow import failed:", e)


#this file creates the models we are going to employ for crowd counting 
def create_chosen_model(model_type: str):
    """
    Factory function to create a model based on model_type.
    Args:
        model_type (str): One of ['resnet50', 'vgg16', 'vgg19', 'xception']
    Returns:
        keras.Model: The selected model.
    Raises:
        ValueError: If an unsupported model_type is provided.
    """
    model_type = model_type.lower()
    
    if model_type == 'resnet50':
        return create_resnet50_model()
    elif model_type == 'vgg16':
        return create_vgg16_model()
    elif model_type == 'vgg19':
        return create_vgg19_model()
    elif model_type == 'xception':
        return create_xception_model()
    else:
        raise ValueError(f"Unsupported model_type: {model_type}. Choose from 'resnet50', 'vgg16', 'vgg19', 'xception'.")


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
    """
    This function creates a Xception model with the following specifications:
    - Pretrained on ImageNet
    - Excludes the top fully-connected layer
    - Input shape of (224, 224, 3)
    - Global average pooling layer added after the last convolutional block
    """
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