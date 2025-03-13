import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Cache les avertissements TF

import numpy as np  # Pour les calculs mathématiques
import cv2         # Pour le traitement d'images
import tensorflow as tf  # Pour le deep learning
from keras.models import load_model  # Pour charger le modèle entraîné


# Charge le modèle CNN préalablement entraîné
model = load_model('asl_model.keras')

# Charge l'image montrant tous les signes ASL
image_reference = cv2.imread("asl_reference.jpg")
# Redimensionne pour l'affichage
image_reference = cv2.resize(image_reference, (640, 480))


# Variables pour la détection de la main
arriere_plan = None  # Stocke l'arrière-plan sans main
poids_accumule = 0.5  # Pour mise à jour progressive du fond


# Rectangle où la main doit être placée
ROI_haut = 100     # Distance depuis le haut
ROI_bas = 300      # Distance jusqu'en bas
ROI_droite = 150   # Distance depuis la droite
ROI_gauche = 350   # Distance depuis la gauche


def calculer_moyenne_accumulee(frame, poids_accumule):
   """Met à jour l'arrière-plan progressivement"""
   global arriere_plan
   if arriere_plan is None:
       arriere_plan = frame.copy().astype("float")
       return None
   cv2.accumulateWeighted(frame, arriere_plan, poids_accumule)


def segmenter_main(frame, seuil=25):
   """Isole la main de l'arrière-plan"""
   global arriere_plan
   # Soustraction d'arrière-plan
   diff = cv2.absdiff(arriere_plan.astype("uint8"), frame)
   # Binarisation
   _, image_seuil = cv2.threshold(diff, seuil, 255, cv2.THRESH_BINARY)
   # Détection des contours
   contours, _ = cv2.findContours(image_seuil.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

   if len(contours) == 0:
       return None
   else:
       # Garde le plus grand contour (la main)
       main_contour = max(contours, key=cv2.contourArea)
       return image_seuil, main_contour


# Dictionnaire pour convertir indices en lettres
lettres = {i: chr(65 + i) for i in range(26)}  # A=0, B=1, etc.


# Initialise la webcam
camera = cv2.VideoCapture(0)
nb_frames = 0  # Compteur d'images


while True:
   # Capture et prépare l'image
   ret, frame = camera.read()
   frame = cv2.flip(frame, 1)  # Miroir pour faciliter les gestes
   frame_copie = frame.copy()

   # Extrait la zone d'intérêt
   roi = frame[ROI_haut:ROI_bas, ROI_droite:ROI_gauche]
   frame_gris = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
   frame_gris = cv2.GaussianBlur(frame_gris, (9, 9), 0)

   # Phase d'initialisation
   if nb_frames < 70:
       calculer_moyenne_accumulee(frame_gris, poids_accumule)
       cv2.putText(frame_copie, "INITIALISATION...", (80, 400), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
   else:
       # Détection de la main
       main = segmenter_main(frame_gris)

       if main is not None:
           image_seuil, contour_main = main
           # Dessine le contour de la main
           cv2.drawContours(frame_copie, [contour_main + (ROI_droite, ROI_haut)], -1, (255, 0, 0), 1)

           cv2.imshow("Main Segmentée", image_seuil)

           # Prépare l'image pour la prédiction
           image_pred = cv2.resize(image_seuil, (64, 64))
           image_pred = cv2.cvtColor(image_pred, cv2.COLOR_GRAY2RGB)
           image_pred = np.expand_dims(image_pred, axis=0)
           image_pred = tf.keras.applications.vgg16.preprocess_input(image_pred)

           # Prédit la lettre avec le modèle
           prediction = model.predict(image_pred, verbose=0)
           lettre_predite = lettres[np.argmax(prediction)]
           cv2.putText(frame_copie, f"Lettre: {lettre_predite}", (70, 45), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

   # Affiche le rectangle de détection
   cv2.rectangle(frame_copie, (ROI_gauche, ROI_haut), (ROI_droite, ROI_bas), (255, 128, 0), 3)
   cv2.putText(frame_copie, "Reconnaissance ASL", (10, 20), cv2.FONT_ITALIC, 0.5, (51, 255, 51), 1)

   # Prépare et affiche l'interface
   frame_copie_redim = cv2.resize(frame_copie, (640, 480))
   images_combinees = np.hstack((frame_copie_redim, image_reference))
   cv2.imshow("Detection ASL", images_combinees)

   nb_frames += 1

   # Quitte si ESC pressé
   if cv2.waitKey(1) & 0xFF == 27:
       break

# Nettoyage final
camera.release()
cv2.destroyAllWindows()