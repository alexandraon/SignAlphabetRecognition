import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Cache les avertissements TF

import tensorflow as tf  # Framework de deep learning
import numpy as np      # Pour les calculs mathématiques
from keras.utils import image_dataset_from_directory  # Pour charger et augmenter les images par lots
from keras.models import Sequential  # Pour créer un modèle séquentiel
from keras.layers import Activation, Dense, Flatten, Conv2D, MaxPool2D, Dropout  # Différentes couches du réseau
from keras.optimizers import Adam  # Optimiseur pour l'apprentissage
from keras.callbacks import ReduceLROnPlateau, EarlyStopping  # Pour optimiser l'entraînement


# Chemin vers le dataset
data_path = './dataset'


# Création des datasets
ds = image_dataset_from_directory(data_path,
   image_size=(64, 64),      # Redimensionne les images en 64x64
   batch_size=32,            # Traite 32 images à la fois
   label_mode='categorical',  # Mode pour la classification
   validation_split=0.3,      # 30% pour la validation
   subset='training',        # Sous-ensemble d'entraînement
   seed=123                  # Pour la reproductibilité
)

test_ds = image_dataset_from_directory(data_path, image_size=(64, 64), batch_size=32, label_mode='categorical', validation_split=0.3, subset='validation', seed=123)

# Prétraitement des images
train_ds = ds.map(lambda x, y: (tf.keras.applications.vgg16.preprocess_input(x), y))
test_ds = test_ds.map(lambda x, y: (tf.keras.applications.vgg16.preprocess_input(x), y))


# Création du modèle CNN
model = Sequential([
   # Première couche de convolution
   Conv2D(64, (3, 3), activation='relu', input_shape=(64, 64, 3)),  # 32 filtres 3x3
   MaxPool2D(2, 2),  # Réduit dimensions
   Dropout(0.3),

   # Deuxième couche
   Conv2D(128, (3, 3), activation='relu'),
   MaxPool2D(2, 2),
   Dropout(0.3),

   # Troisième couche
   Conv2D(256, (3, 3), activation='relu'),
   MaxPool2D(2, 2),
   Dropout(0.3),

   # Couches denses pour la classification
   Flatten(),  # Convertit en 1D
   Dense(512, activation='relu'),
   Dropout(0.5),
   Dense(256, activation='relu'),
   Dropout(0.5),
   Dense(26, activation='softmax')  # Couche de sortie : 26 classes (A-Z)
])


# Configuration de l'apprentissage
model.compile(
   optimizer=Adam(learning_rate=0.001),  # Optimiseur avec taux d'apprentissage
   loss='categorical_crossentropy',      # Fonction de perte
   metrics=['accuracy']                  # Métrique suivie
)

# Configuration des callbacks pour optimiser l'entraînement
reduce_lr = ReduceLROnPlateau(
   monitor='val_loss',  # Surveille la perte de validation
   factor=0.2,         # Réduit le learning rate de 20%
   patience=2,         # Attend 3 epochs avant réduction
   min_lr=0.0001      # Learning rate minimum
)

early_stop = EarlyStopping(
   monitor='val_loss',          # Surveille la perte de validation
   patience=3,                  # Attend 5 epochs avant arrêt
   restore_best_weights=True    # Restaure les meilleurs poids
)


# Entraînement du modèle
history = model.fit(
   train_ds,                        # Données d'entraînement
   epochs=20,                       # Nombre total d'epochs
   validation_data=test_ds,         # Données de validation
   callbacks=[reduce_lr, early_stop] # Callbacks d'optimisation
)


# Sauvegarde le modèle entraîné
model.save('asl_model.keras')


# Crée dictionnaire pour convertir indices en lettres
lettres = {i: chr(65 + i) for i in range(26)}  # A=0, B=1, etc.


# Test du modèle et affichage des prédictions
for images, labels in test_ds.take(1):  # Prend un batch
   predictions = model.predict(images)  # Fait les prédictions
   for i, pred in enumerate(predictions):
       vraie_lettre = chr(65 + np.argmax(labels[i]))  # Lettre réelle
       lettre_predite = lettres[np.argmax(pred)]      # Lettre prédite
       print(f"Image {i}: Prédiction -> {lettre_predite} (Vraie lettre: {vraie_lettre})")