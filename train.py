import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
import cv2
import mediapipe as mp

mphands = mp.solutions.hands
hands = mphands.Hands()
mpDraw = mp.solutions.drawing_utils

cam = cv2.VideoCapture(0)
clsfr = RandomForestClassifier()
data_Set = pickle.load(open('dataset.pickle','rb'))
with open('dataset.pickle', 'rb') as f:
    data_Set = pickle.load(f)


ds = np.asarray(data_Set['data'])
lab = np.asarray(data_Set['label'])

#80%-20% train test split ratio 
xtrain,xtest,ytrain,ytest = train_test_split(ds,lab,test_size=0.2,shuffle=True,stratify=lab)

clsfr.fit(xtrain,ytrain)

#predicting
'''
To test the accuracy of model
pr = clsfr.predict(xtest)
accuracy = accuracy_score(ytest, pr)
print(f"Accuracy: {accuracy}")'''

while True:
    #read from camera and use the output frane to detect hands
    frm_flg,frame = cam.read()
    if not frm_flg:
        print("unable to capture camera")
        break
    imgRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) 
    results = hands.process(imgRGB)
    #print(results.multi_hand_landmarks)

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            shape_data = []
            for landmark in handLms.landmark:
                        x = landmark.x
                        y = landmark.y
                        shape_data.append(x)
                        shape_data.append(y)

            if len(shape_data) == 42:  # Check for complete landmarks
                shape_data = np.array(shape_data).reshape(1, -1)  # Reshape for prediction
                prediction = clsfr.predict(shape_data)[0]  # Get the prediction (string)
                cv2.putText(frame, str(prediction), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
    else:
        cv2.putText(frame, "No hand detected", (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 255), 3)

    cv2.imshow("video",frame)
    #getting the key we are using 27 since it's ascii of esc
    key = cv2.waitKey(1)
    if key==27:
        break


cam.release()
cv2.destroyAllWindows