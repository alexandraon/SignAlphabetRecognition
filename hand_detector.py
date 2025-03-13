import cv2  # Pour la capture vidéo et le traitement d'image
import mediapipe as mp  # Pour la détection des mains
import time  # Pour calculer les FPS


class HandDetector:
    """
    Classe pour détecter et suivre les mains dans une image ou un flux vidéo
    """

    def __init__(self, mode=False, max_hands=2, detection_confidence=0.5, track_confidence=0.5):
        """
        Initialisation du détecteur de mains
        :param mode: False pour traitement vidéo (plus rapide), True pour images statiques
        :param max_hands: Nombre maximum de mains à détecter
        :param detection_confidence: Seuil de confiance pour la détection (0-1)
        :param track_confidence: Seuil de confiance pour le suivi (0-1)
        """
        self.mode = mode
        self.max_hands = max_hands
        self.detection_confidence = detection_confidence
        self.track_confidence = track_confidence

        # Initialisation du module hands de MediaPipe
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=self.mode,
            max_num_hands=self.max_hands,
            min_detection_confidence=self.detection_confidence,
            min_tracking_confidence=self.track_confidence
        )
        # Utilitaire de dessin de MediaPipe
        self.mp_draw = mp.solutions.drawing_utils

    def find_hands(self, img, draw=True):
        """
        Détecte les mains dans une image
        :param img: Image à analyser (format BGR)
        :param draw: Si True, dessine les points de repère sur l'image
        :return: Image avec les marqueurs dessinés si draw=True
        """
        # Conversion de BGR à RGB (MediaPipe utilise RGB)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # Traitement de l'image pour détecter les mains
        self.results = self.hands.process(img_rgb)

        # Si des mains sont détectées
        if self.results.multi_hand_landmarks:
            for hand_landmarks in self.results.multi_hand_landmarks:
                if draw:
                    # Dessine les points de repère et leurs connexions
                    self.mp_draw.draw_landmarks(
                        img,
                        hand_landmarks,
                        self.mp_hands.HAND_CONNECTIONS
                    )
        return img

    def find_position(self, img, hand_no=0):
        """
        Trouve les coordonnées des points de repère d'une main spécifique
        :param img: Image source
        :param hand_no: Index de la main à analyser (si plusieurs mains détectées)
        :return: Liste des coordonnées [id, x, y] pour chaque point de repère
        """
        landmark_list = []
        if self.results.multi_hand_landmarks:
            # Vérifie si la main demandée existe
            if len(self.results.multi_hand_landmarks) > hand_no:
                my_hand = self.results.multi_hand_landmarks[hand_no]
                # Pour chaque point de repère
                for id, lm in enumerate(my_hand.landmark):
                    # Conversion des coordonnées relatives (0-1) en pixels
                    h, w, c = img.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    landmark_list.append([id, cx, cy])
        return landmark_list


def main():
    """
    Fonction principale qui gère la capture vidéo et l'affichage
    """
    # Initialiser la capture vidéo (0 pour la webcam par défaut)
    cap = cv2.VideoCapture(0)

    # Créer l'instance du détecteur de main
    detector = HandDetector()

    # Variables pour calculer les FPS
    p_time = 0  # temps précédent
    c_time = 0  # temps courant

    # Boucle principale
    while True:
        # Capture d'une frame depuis la webcam
        success, img = cap.read()
        if not success:
            print("Échec de la capture vidéo")
            break

        # Détecter les mains dans l'image
        img = detector.find_hands(img)

        # Obtenir la position des points de la main
        landmark_list = detector.find_position(img)

        # Calculer et afficher les FPS
        c_time = time.time()
        fps = 1 / (c_time - p_time)
        p_time = c_time

        # Afficher le texte des FPS sur l'image
        cv2.putText(img, f'FPS: {int(fps)}', (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)

        # Afficher l'image dans une fenêtre
        cv2.imshow("Image", img)

        # Vérifier si la touche 'q' est pressée pour quitter
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Libérer les ressources
    cap.release()
    cv2.destroyAllWindows()


# Point d'entrée du programme
if __name__ == "__main__":
    main()