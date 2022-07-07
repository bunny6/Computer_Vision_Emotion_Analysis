#installing open_cv whole package
pip install opencv-contrib-python


#downloading and installing deepface for using pre-trained model.
pip install deepface

#importing libraries
import cv2
from deepface import DeepFace #inorder to detect a person is happy or sad, we have deepface library
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

#importing single image
img=cv2.imread("happy.jpeg")

plt.imshow(img)

#displaying image in RGB
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

#Analyzing the emotion from the picture
pred=DeepFace.analyze(img)

pred

pred['dominant_emotion']

#making the rectangle across the face
faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
faces = faceCascade.detectMultiScale(gray,1.1,4)

for (x,y,w, h) in faces:
    cv2.rectangle(img, (x,y), (x+w, y+h), (0,255,0), 2)

plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

#making the emotion display above the rectangle box
font = cv2.FONT_HERSHEY_SIMPLEX

cv2.putText(img,pred['dominant_emotion'],
           (0,50),
           font, 1,
           (0,0,255),
           2,
           cv2.LINE_4);

plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

#Real_time prediction using web-cam
import cv2 
from deepface import DeepFace
#making the rectangle across the face
faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

cap=cv2.VideoCapture(1)
if not cap.isOpened():
    cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise IOError("Cannot open Webcam")
    
while True:
    ret,frame = cap.read()
    result  = DeepFace.analyze(frame, actions = ['emotion'], enforce_detection=False)
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(gray,1.1,4)
    
    for(x,y,w, h) in faces:
        cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 2)
        
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    cv2.putText(frame,result['dominant_emotion'],
           (50,50),
           font, 3,
           (0,0,255),
           2,
           cv2.LINE_4)
    cv2.imshow("Original video", frame)
    
    if cv2.waitKey(2) & 0xFF == ord('q'):
        break
        
cap.release() 
cv2.destroyAllWindows() 





 

