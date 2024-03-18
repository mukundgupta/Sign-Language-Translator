import pickle
import cv2
import mediapipe as mp
import numpy as np
import pyttsx3 as speech
from time import sleep



cap = cv2.VideoCapture(0)

model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']
speaker = speech.init()
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

labels_dict = {0:'Hello',1:'Yes',2:'No',3:'Y',4:'O'}
next_ch = ''



def get_results():
    
    data_aux = []
    x_ = []
    y_ = []
    ret, frame = cap.read()
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    cv2.waitKey(40)
    
    results = hands.process(frame_rgb)
    if results.multi_hand_landmarks:
        for landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )
            
            for landmarks in results.multi_hand_landmarks:
                for i in range(len(landmarks.landmark)):
                    x = landmarks.landmark[i].x
                    y = landmarks.landmark[i].y

                    x_.append(x)
                    y_.append(y)

                for i in range(len(landmarks.landmark)):
                    x = landmarks.landmark[i].x
                    y = landmarks.landmark[i].y
                    data_aux.append(x - min(x_))
                    data_aux.append(y - min(y_))

        prediction = model.predict([np.asarray(data_aux)])
            

        p_label = labels_dict[int(prediction[0])]

        
        cv2.imshow('frame', frame)
        
        
        return p_label
        

        
    

def print_result(p_label):
    global next_ch
    print(p_label)
    if p_label != next_ch and p_label != None:
         #print(predicted_character)
         speaker.say(p_label)
         speaker.runAndWait()
         next_ch = p_label

while True:

    while True:

        try:
 
            ch = get_results()
            print_result(ch)

        except:
            print("An exception was detected!\n\nThere may be multiple hands in the frame\nThis program was made to handle one hand.\n")
            print("Input 'e' to exit, any other key to restart: ",end = '')
            
            break
    i = input()
    if i == 'e':
        break
            
    