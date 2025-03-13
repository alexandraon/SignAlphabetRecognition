import cv2
import numpy as np
import random

def nothing(x):
    pass

# Créer une fenêtre pour les curseurs
cv2.namedWindow('Trackbars')

# Créer les curseurs pour les seuils HSV bas et haut
cv2.createTrackbar('Lower H', 'Trackbars', 0, 179, nothing)
cv2.createTrackbar('Lower S', 'Trackbars', 0, 255, nothing)
cv2.createTrackbar('Lower V', 'Trackbars', 0, 255, nothing)
cv2.createTrackbar('Upper H', 'Trackbars', 179, 179, nothing)
cv2.createTrackbar('Upper S', 'Trackbars', 255, 255, nothing)
cv2.createTrackbar('Upper V', 'Trackbars', 255, 255, nothing)

cv2.setTrackbarPos('Lower H', 'Trackbars', 0)
cv2.setTrackbarPos('Lower S', 'Trackbars', 40)
cv2.setTrackbarPos('Lower V', 'Trackbars', 70)
cv2.setTrackbarPos('Upper H', 'Trackbars', 10)
cv2.setTrackbarPos('Upper S', 'Trackbars', 220)
cv2.setTrackbarPos('Upper V', 'Trackbars', 255)

threshold = 60
cap = cv2.VideoCapture(0)

def calculate_curvature(approx_contour):
    # réduction de la dimension des points : [[x, y]] -> [x, y]
    points = approx_contour.reshape(-1, 2)
    n = len(points)
    curvatures = []
    
    for i in range(n):
        p_prev = points[i - 1]
        p_curr = points[i]
        p_next = points[(i + 1) % n]
        
        # vecteurs entre trois points
        vec1 = p_curr - p_prev
        vec2 = p_next - p_curr
        
        # normes et produit scalaire
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        if norm1 == 0 or norm2 == 0:  # cas où deux points consécutifs sont identiques
            curvatures.append(0)
            continue
        
        dot_product = np.dot(vec1, vec2)
        
        # angle entre les vecteurs
        angle = np.arccos(np.clip(dot_product / (norm1 * norm2), -1.0, 1.0))  # clip pour éviter les erreurs numériques
        angle = np.degrees(angle)
        curvatures.append(angle)
    
    return curvatures

# Détection des pics, ici on essaie de détecter le bout des doigts (où il y a des angles forts)

def detect_peaks(approx):
    peaks = []
    n = len(approx)
    for i in range(n):
        p1 = approx[i - 1][0]
        p2 = approx[i][0]
        p3 = approx[(i + 1) % n][0]

        # vecteurs
        v1 = np.array(p2) - np.array(p1)
        v2 = np.array(p3) - np.array(p2)

        # angle entre les vecteurs
        cos_theta = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        angle = np.degrees(np.arccos(np.clip(cos_theta, -1.0, 1.0)))

        # détection de "pics" (par exemple angle < 70°)
        if angle < 70:
            peaks.append(p2)

    return peaks

# Pour détecter la main on essaie de détecter les polygones "en forme d'étoile", cad avec des variations fortes de courbures sur le polyline approximé par la bibliothèque

def find_star_shape(peaks, contour):
    # calcul du centre approximatif (barycentre)
    M = cv2.moments(contour)
    cx = int(M['m10'] / M['m00'])
    cy = int(M['m01'] / M['m00'])

    # vérifier la distribution des pics
    radial_distances = [np.linalg.norm(np.array(p) - np.array([cx, cy])) for p in peaks]
    
    # critère pour détecter une "forme étoile" : variations significatives des distances radiales
    if radial_distances and max(radial_distances) - min(radial_distances) > threshold:  # Définir un seuil
        return True
    return False


while True:
    # Lire une image de la webcam
    ret, frame = cap.read()
    
    if not ret:
        break
    
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Lire les valeurs des curseurs
    lower_h = cv2.getTrackbarPos('Lower H', 'Trackbars')
    lower_s = cv2.getTrackbarPos('Lower S', 'Trackbars')
    lower_v = cv2.getTrackbarPos('Lower V', 'Trackbars')
    upper_h = cv2.getTrackbarPos('Upper H', 'Trackbars')
    upper_s = cv2.getTrackbarPos('Upper S', 'Trackbars')
    upper_v = cv2.getTrackbarPos('Upper V', 'Trackbars')
    
    # Nettoyage de l'image pour faciliter la détection de contours
    
    # Définir les plages de couleur pour les tons de peau
    lower_skin = np.array([lower_h, lower_s, lower_v], dtype=np.uint8)
    upper_skin = np.array([upper_h, upper_s, upper_v], dtype=np.uint8)

    # Créer un masque pour les tons de peau
    skin_mask = cv2.inRange(hsv_frame, lower_skin, upper_skin)

    # Appliquer le masque à l'image originale
    skin = cv2.bitwise_and(frame, frame, mask=skin_mask)
    
    # On passe les teints de peau en noir et blanc
    gray_skin = cv2.cvtColor(skin, cv2.COLOR_BGR2GRAY)
    
    # Flou pour réduire l'impact du bruit
    blurred = cv2.medianBlur(gray_skin, 15)
    
    # Filtre de Canny pour faire sortir les contours
    edges = cv2.Canny(blurred, 300, float('inf'))
    
    ###################################
    
    affected_frame = edges
    
    # Détection des contours par OpenCV
    
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
    # Travail dans les contours détectés
        
    if contours:
        for contour in contours:
            if not cv2.contourArea(contour):
                print("no contour")
                continue
            # Générer une couleur aléatoire
            color = tuple(np.random.randint(0, 255, 3).tolist())
            
            # Calculer l'enveloppe convexe
            hull = cv2.convexHull(contour)
            hull_area = cv2.contourArea(hull)
            area = cv2.contourArea(contour)
            
            # Ratio entre l'aire du contour et son enveloppe convexe (pas utilisé pour le moment)
            ratio = area/hull_area if hull_area else 0
            
            # Approximation des contours par des droites (c'est ce sur quoi on fait les tests de détection de forme de la main)
            epsilon = 0.01 * cv2.arcLength(contour, True) 
            approx = cv2.approxPolyDP(contour, epsilon, True)
            # cv2.polylines(frame, [approx], isClosed=True, color=color, thickness=3)
            
            peaks = detect_peaks(approx)
            if find_star_shape(peaks, contour):
                cv2.drawContours(frame, [contour], -1, color, 2)
            
    else:
        print("no contour")
 
    cv2.imshow('Reel', frame)
    cv2.imshow('Contours', edges)

    # Quitter la boucle si la touche 'q' est pressée
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
