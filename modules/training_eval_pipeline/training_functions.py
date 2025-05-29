from modules.models.create_desired_model import create_chosen_model
from config.config_loader import load_model_config
from keras.optimizers import Adam
from keras.callbacks import ReduceLROnPlateau,EarlyStopping
from tensorflow.keras import backend as K

def euclidean_loss(y_true, y_pred):
    squared_diff = K.square(y_pred - y_true)
    loss = K.mean(squared_diff) / 2
    return loss

def compile_model(model_name:str):
    # Load model parameters from YAML
    model_params = load_model_config(model_name)['parameters']


    # Map loss name from YAML to actual function or string
    loss_functions = {
        "mse": "mean_squared_error",       
        "euclidean_loss": euclidean_loss   
    }
    loss_name = model_params["loss"]
    loss_to_use = loss_functions.get(loss_name, "mean_squared_error")

    # Create the model
    model = create_chosen_model(model_name)

    adam = Adam(learning_rate=model_params['adam_lr'])

    # Compile the model
    model.compile(optimizer=adam, loss=loss_to_use, metrics=model_params["metrics"])

    return model

def fit_model_using_best_parameters(model_name:str, model=None,train_generator=None, valid_generator=None):
    
    # Load model parameters from YAML
    model_params = load_model_config(model_name)['parameters']
    
    es = EarlyStopping(monitor=model_params['monitor'], 
                   mode='min',
                   patience=model_params['es_patience'])
    
    learning_rate_reduction = ReduceLROnPlateau(
        monitor=model_params['monitor'],  # Track the score on the validation set
        patience=3,  # Number of epochs in which no improvement is seen.
        verbose=1,
        factor=0.2,  # Factor by which the LR is multiplied.
        min_lr=0.000001  # Don't go below this value for LR.
    )
    
    # Train the model
    history = model.fit(train_generator, epochs=model_params["epochs"], validation_data=valid_generator, callbacks=[learning_rate_reduction, es])
    
    return history,model

