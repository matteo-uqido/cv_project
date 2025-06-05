from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense
from tensorflow.keras.applications import vgg16, vgg19, resnet50, xception


def create_resnet50_model():
    base_model = resnet50.ResNet50(
        weights='imagenet',  
        include_top=False,  
        input_shape=(240, 320, 3),  
        pooling='avg', 
    )
    
    x = base_model.output  
    x = Dense(1024, activation='relu')(x)  
    predictions = Dense(1, activation='linear')(x)  
    model = Model(inputs=base_model.input, outputs=predictions)

    k = -7
    for layer in model.layers[:k]:
        layer.trainable = False
    for layer in model.layers[k:]:
        layer.trainable = True

    return model

def create_vgg16_model():
    base_model = vgg16.VGG16(
        weights='imagenet',  
        include_top=False,  
        input_shape=(240, 320, 3),  
        pooling='avg', 
    )
    
    x = base_model.output  
    x = Dense(1024, activation='relu')(x)  
    predictions = Dense(1, activation='linear')(x)  
    model = Model(inputs=base_model.input, outputs=predictions)

    k = -7
    for layer in model.layers[:k]:
        layer.trainable = False
    for layer in model.layers[k:]:
        layer.trainable = True

    return model

def create_vgg19_model():
    base_model = vgg19.VGG19(
        weights='imagenet',  
        include_top=False,  
        input_shape=(240, 320, 3),  
        pooling='avg', 
    )
    
    x = base_model.output  
    x = Dense(1024, activation='relu')(x)  
    predictions = Dense(1, activation='linear')(x)  
    model = Model(inputs=base_model.input, outputs=predictions)

    k = -7
    for layer in model.layers[:k]:
        layer.trainable = False
    for layer in model.layers[k:]:
        layer.trainable = True

    return model

def create_xception_model():
    base_model = xception.Xception(
        weights='imagenet',  
        include_top=False,  
        input_shape=(240, 320, 3),  
        pooling='avg', 
    )
    for layer in base_model.layers:
        layer.trainable = False

    x = base_model.output  
    x = Dense(1024, activation='relu')(x)  
    predictions = Dense(1, activation='linear')(x)  
    model = Model(inputs=base_model.input, outputs=predictions)

    return model