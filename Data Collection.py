import os
import cv2
import pickle
import mediapipe as mp

mphands = mp.solutions.hands
hands = mphands.Hands()
mpDraw = mp.solutions.drawing_utils

#add your folder name here and place it with main.py
cl = ['Ok','Palm']

#Extracting features and labels from images
def read_image():
    deleted_img = 0
    hand_landmarks_data = []
    lab = []
    for ct in cl:
        for img_name in os.listdir(ct):
            shape_data = []
            cim = os.path.join(ct,img_name)
            #print(cim)
            img = cv2.imread(cim)
            imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 
            imgRGB = cv2.resize(imgRGB,(640,240))
            results = hands.process(imgRGB)
            #print(results.multi_hand_landmarks)
            if results.multi_hand_landmarks:
                for handLms in results.multi_hand_landmarks:
                    #mpDraw.draw_landmarks(imgRGB,handLms,mphands.HAND_CONNECTIONS)
                    for landmark in handLms.landmark:
                        x = landmark.x
                        y = landmark.y
                        shape_data.append(x)
                        shape_data.append(y)
                    
                    if len(shape_data) == 42:  # Check if all 42 coordinates are present
                        hand_landmarks_data.append(shape_data)
                        lab.append(ct)
                    else:
                        deleted_img+=1
                        print(f"Incomplete landmarks in {cim}. Deleting.")  # Or delete the image
                        os.remove(cim)  # Uncomment to delete the image if needed 
            else:
                deleted_img+=1
                print(f"No landmarks in {cim}. Deleting.")
                os.remove(cim) 
    return hand_landmarks_data,lab,deleted_img


deleted_img = -1

#we use this to delete all images with missing landmarks to get perfect dataset
while deleted_img!=0:
    ds,ls,deleted_img = read_image()
    
f_data_set = open('dataset.pickle','wb')
pickle.dump({'data':ds,'label':ls},f_data_set)
f_data_set.close()