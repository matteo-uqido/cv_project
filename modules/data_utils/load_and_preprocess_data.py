
import os
from modules.data_utils.data_processor_class import DataGenerator_crowd_counting_models, DataGenerator_Csrnet
import pandas as pd
from scipy.io import loadmat
from sklearn.model_selection import train_test_split
from paths import DATA_PATH
def load_mall_dataset(path):
    
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

def create_generator(generator_class, df, batch_size, **kwargs):
    return generator_class(df, batch_size=batch_size, **kwargs)

def get_new_generators(batch_size=32, model_name=None, augment_data=False):
    if model_name is None:
        raise ValueError("Parameter 'model_name' must be specified (e.g., 'vgg16', 'resnet', etc.).")

    train_df, valid_df, test_df = get_train_test_val_partition(DATA_PATH)

    target_size = (240, 320) if model_name in ["csrnet", "modified_csrnet"] else (224, 224)

    if model_name == "csrnet":
        train_generator = create_generator(
            DataGenerator_Csrnet,
            train_df,
            batch_size,
            shuffle=True,
            target_size=target_size
        )
        valid_generator = create_generator(
            DataGenerator_Csrnet,
            valid_df,
            batch_size,
            shuffle=False,
            target_size=target_size
        )
        test_generator = create_generator(
            DataGenerator_Csrnet,
            test_df,
            batch_size,
            shuffle=False,
            target_size=target_size
        )

    elif model_name == "modified_csrnet":
        train_generator = create_generator(
            DataGenerator_Csrnet,
            train_df,
            batch_size,
            target_size=target_size,
            is_standard_model=False
        )
        valid_generator = create_generator(
            DataGenerator_Csrnet,
            valid_df,
            batch_size,
            target_size=target_size,
            shuffle=False,
            is_standard_model=True
        )
        test_generator = create_generator(
            DataGenerator_Csrnet,
            test_df,
            batch_size,
            target_size=target_size,
            shuffle=False,
            is_standard_model=True
        )

    else:
        train_generator = create_generator(
            DataGenerator_crowd_counting_models,
            train_df,
            batch_size,
            target_size=target_size,
            model_name=model_name,
            augment=bool(augment_data)
        )
        valid_generator = create_generator(
            DataGenerator_crowd_counting_models,
            valid_df,
            batch_size,
            target_size=target_size,
            model_name=model_name,
            shuffle=False
        )
        test_generator = create_generator(
            DataGenerator_crowd_counting_models,
            test_df,
            batch_size,
            target_size=target_size,
            model_name=model_name,
            shuffle=False
        )

    return train_generator, valid_generator, test_generator




