import cv2  # OpenCV : bibliothèque pour traitement d'images et vision par ordinateur
import numpy as np  # NumPy : pour les calculs mathématiques sur les images
import os  # Pour créer/gérer les dossiers et fichiers


# Crée un dossier principal 'dataset' pour stocker toutes les images
dataset_dir = "dataset"
if not os.path.exists(dataset_dir):
    os.makedirs(dataset_dir)

# Crée 26 sous-dossiers (A-Z) pour chaque lettre de l'alphabet
for lettre in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
    lettre_dir = os.path.join(dataset_dir, lettre)
    if not os.path.exists(lettre_dir):
        os.makedirs(lettre_dir)


# Charge l'image montrant tous les signes ASL
image_reference = cv2.imread("asl_reference.jpg")
# Redimensionne l'image pour l'afficher à côté du flux vidéo
image_reference = cv2.resize(image_reference, (640, 480))


# Ces variables servent à la détection du mouvement
arriere_plan = None  # Stocke l'image de fond sans main
poids_accumule = 0.5  # Contrôle la vitesse de mise à jour du fond


# Rectangle où la main doit être placée pour être détectée
ROI_haut = 100  # Distance depuis le haut
ROI_bas = 300  # Distance jusqu'en bas
ROI_droite = 150  # Distance depuis la droite
ROI_gauche = 350  # Distance depuis la gauche


def calculer_moyenne_accumulee(image, poids_accumule):
    """
    Fonction qui met à jour progressivement l'image d'arrière-plan
    en fusionnant chaque nouvelle image avec l'ancienne.
    Cela permet d'avoir un arrière-plan stable même avec de petits mouvements
    """
    global arriere_plan
    if arriere_plan is None:
        # Premier appel : initialise l'arrière-plan
        arriere_plan = image.copy().astype("float")
        return None
    # Fusionne la nouvelle image avec l'arrière-plan existant
    cv2.accumulateWeighted(image, arriere_plan, poids_accumule)


def segmenter_main(image, seuil=25):
    """
    Fonction qui isole la main du reste de l'image.
    Elle compare l'image actuelle avec l'arrière-plan pour détecter les changements,
    puis trouve les contours de la main.
    """
    global arriere_plan
    # Calcule la différence entre l'image actuelle et l'arrière-plan
    diff = cv2.absdiff(arriere_plan.astype("uint8"), image)
    # Convertit l'image en noir et blanc avec un seuil
    _, image_seuil = cv2.threshold(diff, seuil, 255, cv2.THRESH_BINARY)
    # Trouve les contours dans l'image
    contours, _ = cv2.findContours(image_seuil.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Si aucun contour n'est trouvé, pas de main
    if len(contours) == 0:
        return None
    else:
        # Prend le plus grand contour (supposé être la main)
        main_contour = max(contours, key=cv2.contourArea)
        return image_seuil, main_contour


# Initialisation de la caméra
camera = cv2.VideoCapture(0)  # 0 = première webcam disponible


# Variables de contrôle
lettre_courante = 'A'  # On commence par la lettre A
nb_images = 0  # Compte les images capturées pour chaque lettre
nb_frames = 0  # Compte total d'images traitées
capture_active = False  # Indique si on capture actuellement des images
nb_images_objectif = 250  # Nombre d'images à capturer par lettre


while True:
    # Capture une image de la webcam
    ret, frame = camera.read()
    frame = cv2.flip(frame, 1)  # Effet miroir pour faciliter le positionnement
    frame_copie = frame.copy()  # Copie pour dessiner dessus sans modifier l'original


    # Extrait la zone d'intérêt (ROI) où la main doit être placée
    roi = frame[ROI_haut:ROI_bas, ROI_droite:ROI_gauche]
    # Convertit en niveaux de gris pour simplifier le traitement
    frame_gris = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    # Applique un flou pour réduire le bruit
    frame_gris = cv2.GaussianBlur(frame_gris, (9, 9), 0)

    # Phase d'initialisation (60 premières frames)
    if nb_frames < 60:
        # Capture l'arrière-plan sans main
        calculer_moyenne_accumulee(frame_gris, poids_accumule)
        if nb_frames <= 59:
            # Affiche message d'attente
            cv2.putText(frame_copie, "Capture de l'arriere-plan...", (80, 400),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)


    # Phase de détection et capture
    else:
        # Tente de détecter une main
        main = segmenter_main(frame_gris)

        if main is not None:  # Si une main est détectée
            image_seuil, segment_main = main

            # Dessine le contour de la main détectée
            cv2.drawContours(frame_copie, [segment_main + (ROI_droite, ROI_haut)], -1, (255, 0, 0), 1)

            # Affiche la lettre en cours et le nombre d'images
            cv2.putText(frame_copie, f"Lettre {lettre_courante} ({nb_images}/250)", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            # Message pour varier les positions
            if capture_active:
                cv2.putText(frame_copie, "Variez les angles et positions de la main!",(10, 380), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # Si capture activée et pas encore 250 images pour cette lettre
            if capture_active and nb_images < nb_images_objectif:
                # Sauvegarde l'image de la main
                cv2.imwrite(os.path.join(dataset_dir, lettre_courante, f"{nb_images}.jpg"), image_seuil)
                nb_images += 1
                # Arrête la capture après 250 images
                if nb_images == nb_images_objectif:
                    capture_active = False

            # Affiche l'image de la main isolée
            cv2.imshow("Image Seuil Main", image_seuil)
        else:
            # Affiche un message si pas de main détectée
            cv2.putText(frame_copie, 'Pas de main detectee...', (200, 400), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)


    # Dessine le rectangle de la zone de détection
    cv2.rectangle(frame_copie, (ROI_gauche, ROI_haut), (ROI_droite, ROI_bas), (255, 128, 0), 3)

    # Prépare et affiche l'interface complète
    frame_copie_redim = cv2.resize(frame_copie, (640, 480))
    # Combine la vue caméra et l'image de référence côte à côte
    images_combinees = np.hstack((frame_copie_redim, image_reference))
    cv2.imshow("Detection ASL", images_combinees)

    nb_frames += 1


    # Gestion des touches
    k = cv2.waitKey(1) & 0xFF
    if k == 27:  # Touche ESC pour quitter
        break
    elif k == 32:  # Touche ESPACE pour contrôler la capture
        if not capture_active and nb_images == nb_images_objectif:  # Si 250 images capturées
            if lettre_courante < 'Z':  # Passe à la lettre suivante
                lettre_courante = chr(ord(lettre_courante) + 1)
                nb_images = 0
                capture_active = True
        elif not capture_active:  # Démarre la capture
            capture_active = True


# Ferme toutes les fenêtres et libère la caméra
cv2.destroyAllWindows()
camera.release()
