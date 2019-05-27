import cv2
import numpy as np

cap = cv2.VideoCapture(0)

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    
    print("Codigo de retorno", ret)

    # Our operations on the frame come here
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    minm = np.array([142, 30, 30])
    maxm = np.array([192, 255, 255])
    
    maskm = cv2.inRange(hsv, minm, maxm) 
    
    minc = np.array([92, 40, 40])
    maxc = np.array([122, 255, 255])
    
    maskc = cv2.inRange(hsv, minc, maxc) 

    # Display the resulting frame
    cv2.imshow('mask', maskm)
    cv2.imshow('mask', maskc)
    
#    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
