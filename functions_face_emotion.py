import streamlit as st
from keras.models import load_model
from keras.preprocessing.image import img_to_array
import cv2
import numpy as np

def display_model():
    face_classifier = cv2.CascadeClassifier(r'haarcascade_frontalface_default.xml')
    classifier =load_model(r'Emotion_Recognition_Classifier.h5')

    emotion_labels = ['Angry','Disgust','Fear','Happy','Neutral', 'Sad', 'Surprise']


    recording_state = st.session_state.get("recording_state", False)
    start_button_pressed = st.button("Start Webcam")
    stop_button_pressed=st.button("Stop")

    if start_button_pressed:
        st.session_state.recording_state = True

        cap=cv2.VideoCapture(0)
        
        frame_placeholder=st.empty()
        
        while cap.isOpened() and not stop_button_pressed:


            _, frame = cap.read()
            labels = []
            gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
            faces = face_classifier.detectMultiScale(gray)

            for (x,y,w,h) in faces:
                cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,255),2)
                roi_gray = gray[y:y+h,x:x+w]
                roi_gray = cv2.resize(roi_gray,(48,48),interpolation=cv2.INTER_AREA)



                if np.sum([roi_gray])!=0:
                    roi = roi_gray.astype('float')/255.0
                    roi = img_to_array(roi)
                    roi = np.expand_dims(roi,axis=0)

                    prediction = classifier.predict(roi)[0]
                    label=emotion_labels[prediction.argmax()]
                    

                        
                    label_position = (x,y)
                    cv2.putText(frame,label,label_position,cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
                else:
                    cv2.putText(frame,'No Faces',(30,80),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
            frame_placeholder.image(frame,'RGB')
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        

        cap.release()
        cv2.destroyAllWindows()

    if stop_button_pressed and recording_state:
        st.session_state.recording_state = False

    if st.button("Restart"):
        st.session_state.recording_state = False

