import os
import numpy as np
import cv2
from tensorflow.keras.utils import Sequence
from paths import FRAMES_PATH

from tensorflow.keras.applications.vgg16 import preprocess_input as vgg_preprocess
from tensorflow.keras.applications.vgg19 import preprocess_input as vgg19_preprocess
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess
from tensorflow.keras.applications.xception import preprocess_input as xception_preprocess
import tensorflow as tf

from modules.data_utils.density_map_utils import create_density_map

class DataGenerator_count(Sequence):
    def __init__(self, dataframe, batch_size=32, shuffle=True, target_size=(224, 224),
                 model_name="vgg16", augment=False, seed=42):
        self.df = dataframe
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.frames_dir = FRAMES_PATH
        self.target_size = target_size
        self.model_name = model_name
        self.augment = augment
        self.seed = seed

        if self.augment:
            self.augmenter = tf.keras.Sequential([
                tf.keras.layers.RandomFlip("horizontal", seed=self.seed),
                tf.keras.layers.RandomRotation(0.05, seed=self.seed),
                tf.keras.layers.RandomZoom(0.1, seed=self.seed),
                tf.keras.layers.RandomContrast(0.1, seed=self.seed)
            ])

        if self.shuffle:
            self.on_epoch_end()

    def __len__(self):
        return np.ceil(len(self.df) / self.batch_size).astype(int)

    def resize_image(self, image):
        return cv2.resize(image, self.target_size, interpolation=cv2.INTER_AREA)

    def preprocess_image(self, image):
        image = image.astype(np.float32)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.model_name == "resnet":
            return resnet_preprocess(image)
        elif self.model_name == "vgg16":
            return vgg_preprocess(image)
        elif self.model_name == "vgg19":
            return vgg19_preprocess(image)
        elif self.model_name == "xception":
            return xception_preprocess(image)

    def on_epoch_end(self):
        if self.shuffle:
            self.df = self.df.sample(frac=1, random_state=self.seed).reset_index(drop=True)

    def __getitem__(self, idx):
        batch_df = self.df[idx * self.batch_size:(idx + 1) * self.batch_size]
        X, y = [], []

        for _, row in batch_df.iterrows():
            image_path = os.path.join(self.frames_dir, row.image_name)
            image = cv2.imread(image_path)
            if image is not None:
                image = self.resize_image(image)

                if self.augment:
                    image = tf.convert_to_tensor(image)
                    image = self.augmenter(image, training=True)  # <- apply augmentations
                    image = image.numpy()

                image = self.preprocess_image(image)
                X.append(image)
                y.append(row['count'])
            else:
                print(f"Warning: Could not load image at {image_path}")

        return np.array(X), np.array(y).astype(float)
    
class DataGenerator_count_old(Sequence):
    def __init__(self, dataframe, batch_size=32, shuffle=True, target_size=(224, 224), model_name="vgg16"):
        self.df = dataframe
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.frames_dir = FRAMES_PATH
        self.target_size = target_size
        self.model_name = model_name
        if self.shuffle:
            self.on_epoch_end()


    def __len__(self):
        return np.ceil(len(self.df) / self.batch_size).astype(int)

    def resize_image(self, image):
        # Ridimensiona l'immagine alle dimensioni target (e.g., 224x224)
        image_resized = cv2.resize(image, self.target_size)
        return image_resized

    def preprocess_image(self, image):
        image = image.astype(np.float32)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.model_name == "resnet":
            return resnet_preprocess(image)
        elif self.model_name == "vgg16":
            return vgg_preprocess(image)
        elif self.model_name == "vgg19":
            return vgg19_preprocess(image)
        elif self.model_name == "xception":
            return xception_preprocess(image)

    def on_epoch_end(self):
        if self.shuffle:
            self.df = self.df.sample(frac=1).reset_index(drop=True)
    def __getitem__(self, idx):
        # Estrae il batch di immagini
        batch_df = self.df[idx * self.batch_size:(idx + 1) * self.batch_size]
        X, y = [], []

        for _, row in batch_df.iterrows():
            image_path = os.path.join(self.frames_dir, row.image_name)
            image = cv2.imread(image_path)
            if image is not None:
                image = self.preprocess_image(self.resize_image(image))  # Applica resize e normalization  # Normalizza l'immagine
                X.append(image)
                y.append(row['count'])  # Usa il numero di persone come target
            else:
                print(f"Warning: Could not load image at {image_path}")

        X = np.array(X)
        y = np.array(y)
        return np.array(X), np.array(y).astype(float) # Converti y in formato float per la regressione
    
class DataGenerator_density(Sequence):
    def __init__(self, dataframe, batch_size=32, shuffle=True, target_size=(240, 320), seed=42):
        self.df = dataframe
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.frames_dir = FRAMES_PATH
        self.target_size = target_size
        self.seed = seed

        if self.shuffle:
            self.on_epoch_end()

    def __len__(self):
        return np.ceil(len(self.df) / self.batch_size).astype(int)

    def resize_image(self, image):
        return cv2.resize(image, (self.target_size[1], self.target_size[0]), interpolation=cv2.INTER_AREA)

    def preprocess_image(self, image):
        image = image.astype(np.float32)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image=vgg_preprocess(image)
        return image
    def on_epoch_end(self):
        if self.shuffle:
            self.df = self.df.sample(frac=1, random_state=self.seed).reset_index(drop=True)
    def __getitem__(self, idx):
        batch_df = self.df[idx * self.batch_size:(idx + 1) * self.batch_size]
        X, y = [], []

        for _, row in batch_df.iterrows():
            image_annotations = row['annotations']
            image_density_map = create_density_map(image_annotations)

            # Resize density map to 1/8th of target size
            density_map_target_size = (self.target_size[1] // 8, self.target_size[0] // 8)
            resized_density_map = cv2.resize(
                image_density_map,
                density_map_target_size,
                interpolation=cv2.INTER_AREA)

            image_path = os.path.join(self.frames_dir, row.image_name)
            image = cv2.imread(image_path)
            if image is not None:
                image = self.resize_image(image)
                image = self.preprocess_image(image)
                #create an extra channel for the density map so that it matches the image shape
                image_density_map_channel=image_density_map[:, :, np.newaxis]
                image_with_density = np.concatenate((image, image_density_map_channel), axis=-1)

                X.append(image_with_density)
                y.append(resized_density_map)
            else:
                print(f"Warning: Could not load image at {image_path}")
        return np.array(X), np.array(y, dtype=np.float32)  
