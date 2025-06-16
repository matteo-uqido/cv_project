from modules.models.CSRNET_model import create_CSRNET_model
from modules.models.crowd_counting_models import create_resnet50_model, create_vgg16_model, create_vgg19_model, create_xception_model

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
    elif model_type == 'csrnet':
        return create_CSRNET_model()
    else:
        raise ValueError(f"Unsupported model_type: {model_type}")


