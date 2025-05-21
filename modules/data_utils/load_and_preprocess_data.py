
import os
from modules.data_utils.data_processor_class import DataGenerator_count
import pandas as pd
from scipy.io import loadmat
from sklearn.model_selection import train_test_split
from paths import DATA_PATH
def load_mall_dataset(path):
    # Caricamento del dataset
    
    mall_gt = loadmat(os.path.join(path, 'mall_gt.mat'))
    mall_head_positions_gt = mall_gt['frame'][0]
    
    # Creazione del DataFrame
    df = pd.DataFrame({
        'count': mall_gt['count'].flatten(),
        'annotations': [frame_data[0][0][0] for frame_data in mall_head_positions_gt]})
    
    df = df.reset_index()
    df = df.rename({'index': 'frame_id'}, axis=1)
    df.frame_id = df.index + 1
    df['image_name'] = df['frame_id'].apply(lambda x: f'seq_{x:06d}.jpg')
    
    return df

def get_train_test_val_partition(path):
    df=load_mall_dataset(path)
    train_val_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    train_df, valid_df = train_test_split(train_val_df, test_size=0.2, random_state=42)
    return train_df, valid_df, test_df

def get_new_generators(*, batch_size=32, target_size=(224, 224), model_name=None,augment_data=False):
    """
    Create data generators for training, validation, and testing.
    
    Args:
        batch_size (int): Size of the batches of data.
        target_size (tuple): Target size for resizing images.
        model_name (str): Type of model to use for preprocessing (required).
        
    Returns:
        tuple: Training, validation, and test data generators.
    """
    if model_name is None:
        raise ValueError("Parameter 'model_name' must be specified (e.g., 'vgg16', 'resnet', etc.).")

    train_df, valid_df, test_df = get_train_test_val_partition(DATA_PATH)
    
    train_generator = DataGenerator_count(train_df, batch_size=batch_size, target_size=target_size, model_name=model_name,augment=True if augment_data else False)
    valid_generator = DataGenerator_count(valid_df, batch_size=batch_size, target_size=target_size, model_name=model_name,shuffle=False)
    test_generator = DataGenerator_count(test_df, batch_size=batch_size, target_size=target_size, model_name=model_name,shuffle=False)
    
    return train_generator, valid_generator, test_generator

