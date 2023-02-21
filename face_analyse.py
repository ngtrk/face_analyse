import tensorflow as tf
from keras.models import load_model
import cv2
import numpy as np
from keras.preprocessing.image import img_to_array
import os


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

emotion_model = load_model('trained_models/emotion_model.h5')
age_model = load_model('trained_models/age_model.h5')
gender_model = load_model('trained_models/gender_model.h5')


class_names = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
genders = ['Male', 'Female']


if not os.path.exists('trained_models/emotion_model.tflite'):
    converter = tf.lite.TFLiteConverter.from_keras_model(emotion_model)
    tflite_model = converter.convert()
    open('trained_models/emotion_model.tflite', 'wb').write(tflite_model)

if not os.path.exists('trained_models/age_model.tflite'):
    converter = tf.lite.TFLiteConverter.from_keras_model(age_model)
    tflite_model = converter.convert()
    open('trained_models/age_model.tflite', 'wb').write(tflite_model)


if not os.path.exists('trained_models/gender_model.tflite'):
    converter = tf.lite.TFLiteConverter.from_keras_model(gender_model)
    tflite_model = converter.convert()
    open('trained_models/gender_model.tflite', 'wb').write(tflite_model)


emotion_interpreter = tf.lite.Interpreter(model_path='trained_models/emotion_model.tflite')
emotion_interpreter.allocate_tensors()

age_interpreter = tf.lite.Interpreter(model_path='trained_models/age_model.tflite')
age_interpreter.allocate_tensors()

gender_interpreter = tf.lite.Interpreter(model_path='trained_models/gender_model.tflite')
gender_interpreter.allocate_tensors()


emotion_input = emotion_interpreter.get_input_details()
emotion_output = emotion_interpreter.get_output_details()

age_input = age_interpreter.get_input_details()
age_output = age_interpreter.get_output_details()


gender_input = gender_interpreter.get_input_details()
gender_output = gender_interpreter.get_output_details()


cap = cv2.VideoCapture(0)

while True:
    _, frame = cap.read()
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)
    
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_gray = cv2.resize(roi_gray, (160, 160), interpolation=cv2.INTER_AREA)
        
        roi = roi_gray.astype('float') / 255.0
        roi = img_to_array(roi)
        roi = np.expand_dims(roi, axis=0)
        
        emotion_interpreter.set_tensor(emotion_input[0]['index'], roi)
        emotion_interpreter.invoke()
        emotion_preds = emotion_interpreter.get_tensor(emotion_output[0]['index'])
        
        emotion_label = class_names[emotion_preds.argmax()]
        emotion_label_pos = (x + h, y + h - 40)
        cv2.putText(frame, emotion_label, emotion_label_pos, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # gender
        roi_color = frame[y:y+h, x:x+w]
        roi_color = cv2.resize(roi_color, (96, 96), interpolation=cv2.INTER_AREA)
        roi_color = np.array(roi_color).reshape(-1, 96, 96, 3).astype(np.float32)
        
        gender_interpreter.set_tensor(gender_input[0]['index'], roi_color)
        gender_interpreter.invoke()
        gender_preds = gender_interpreter.get_tensor(gender_output[0]['index'])
        
        gender_preds = (gender_preds >= .5).astype(int)[:, 0]
        gender_label = genders[gender_preds[0]]
        gender_label_pos = (x + h, y + h + 40)
        cv2.putText(frame, gender_label, gender_label_pos, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # age
        roi_color = frame[y:y+h, x:x+w]
        roi_color = cv2.resize(roi_color, (64, 64), interpolation=cv2.INTER_AREA)
        roi_color = np.array(roi_color).reshape(-1, 64, 64, 3).astype(np.float32)

        age_interpreter.set_tensor(age_input[0]['index'], roi_color)
        age_interpreter.invoke()
        age_preds = age_interpreter.get_tensor(age_output[0]['index'])
        
        age = round(age_preds[0, 0])
        age_label_pos = (x + h, y + h)
        cv2.putText(frame, 'Age='+ str(age), age_label_pos, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    
    cv2.imshow('Emotion Detector', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()


