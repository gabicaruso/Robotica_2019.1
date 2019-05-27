import numpy as np
import cv2
from matplotlib import pyplot as plt
from math import pi
import matplotlib.cm as cm

# Número mínimo de pontos correspondentes
MIN_MATCH_COUNT = 5

imagem = cv2.imread('insperlogo.jpeg',0) # Imagem do cenario - puxe do video para fazer isto





cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Cria o detector SIFT
sift = cv2.xfeatures2d.SIFT_create()

# Encontra os pontos únicos (keypoints) nas duas imagems
kp1, des1 = sift.detectAndCompute(imagem ,None)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Imagem de saída
    out = frame_rgb.copy()
    
    kp2, des2 = sift.detectAndCompute(frame_gray, None)
    
    cv2.imshow(out)
    
    # Configurações do algoritmo FLANN que compara keypoints e ver correspondências - não se preocupe com isso
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks = 50)

    # Configura o algoritmo de casamento de features que vê *como* o objeto que deve ser encontrado aparece na imagem
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    # Tenta fazer a melhor comparacao usando o algoritmo
    matches = flann.knnMatch(des1,k=2)

    # store all the good matches as per Lowe's ratio test.
    good = []
    for m,n in matches:
        if m.distance < 0.7*n.distance:
            good.append(m)


    if len(good)>MIN_MATCH_COUNT:
        # Separa os bons matches na origem e no destino
        src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
        dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)


        # Tenta achar uma trasformacao composta de rotacao, translacao e escala que situe uma imagem na outra
        # Esta transformação é chamada de homografia 
        # Para saber mais veja 
        # https://docs.opencv.org/3.4/d9/dab/tutorial_homography.html
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
        matchesMask = mask.ravel().tolist()



        h,w = imagem.shape
        # Um retângulo com as dimensões da imagem original
        pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)

        # Transforma os pontos do retângulo para onde estao na imagem destino usando a homografia encontrada
        dst = cv2.perspectiveTransform(pts,M)


        # Desenha um contorno em vermelho ao redor de onde o objeto foi encontrado
        img2b = cv2.polylines(frame,[np.int32(dst)],True,(255,0,0),3, cv2.LINE_AA)

    else:
        print("Not enough matches are found - %d/%d" % (len(good),MIN_MATCH_COUNT))
        matchesMask = None

    draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                       singlePointColor = None,
                       matchesMask = matchesMask, # draw only inliers
                       flags = 2)
    
    # Display the resulting frame
    cv2.imshow('SIFT',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
        
# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
