import os
import numpy as np
import cv2
from tensorflow.keras.utils import Sequence
from paths import FRAMES_PATH

from tensorflow.keras.applications.vgg16 import preprocess_input as vgg_preprocess
from tensorflow.keras.applications.vgg19 import preprocess_input as vgg19_preprocess
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess
from tensorflow.keras.applications.xception import preprocess_input as xception_preprocess

class DataGenerator_count(Sequence):
    def __init__(self, dataframe, batch_size=32, shuffle=True, target_size=(224, 224), model_type="vgg16"):
        self.df = dataframe.sample(frac=1).reset_index(drop=True) if shuffle else dataframe
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.frames_dir = FRAMES_PATH
        self.target_size = target_size
        self.model_type = model_type


    def __len__(self):
        return np.ceil(len(self.df) / self.batch_size).astype(int)

    def resize_image(self, image):
        # Ridimensiona l'immagine alle dimensioni target (e.g., 224x224)
        image_resized = cv2.resize(image, self.target_size)
        return image_resized

    def preprocess_image(self, image):
        image = image.astype(np.float32)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.model_type == "resnet":
            return resnet_preprocess(image)
        elif self.model_type == "vgg16":
            return vgg_preprocess(image)
        elif self.model_type == "vgg19":
            return vgg19_preprocess(image)
        elif self.model_type == "xception":
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

