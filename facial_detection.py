import numpy as np
from tensorflow.keras.models import load_model
import cv2
from tensorflow.keras.preprocessing.image import img_to_array

def detect_facial_depression():
    """
    Detects facial emotions and predicts depression level based on emotion detection.
    
    Returns:
        int: Predicted depression level (0-5).
    """
    # Load the face detection and emotion classification models
    face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    classifier = load_model('face_emotion_model.h5')

    # Emotion labels
    emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

    # Initialize video capture
    cap = cv2.VideoCapture(0)
    depression_score = 0  # To track depression-related emotions

    print("Press 'q' to stop facial detection.")

    while True:
        _, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_classifier.detectMultiScale(gray)

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
            roi_gray = gray[y:y + h, x:x + w]
            roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)

            if np.sum([roi_gray]) != 0:
                roi = roi_gray.astype('float') / 255.0
                roi = img_to_array(roi)
                roi = np.expand_dims(roi, axis=0)

                prediction = classifier.predict(roi)[0]
                label = emotion_labels[prediction.argmax()]
                label_position = (x, y - 10)
                cv2.putText(frame, label, label_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                # Increment depression score for specific emotions
                if label in ['Sad', 'Angry', 'Fear']:
                    depression_score += 1
            else:
                cv2.putText(frame, 'No Faces', (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow('Facial Depression Detector', frame)

        # Exit loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    # Normalize depression score to a 0-5 scale
    depression_level = min(5, depression_score // 10)  # Adjust divisor as needed
    return depression_level
# def detect_facial_depression():
#     import numpy as np
#     from tensorflow.keras.models import load_model
#     import cv2
#     from tensorflow.keras.preprocessing.image import img_to_array

#     face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
#     classifier = load_model('face_emotion_model.h5')
#     emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

#     cap = cv2.VideoCapture(0)
#     depression_score = 0
#     frame_captured = False

#     for _ in range(10):  # Try 10 frames max
#         ret, frame = cap.read()
#         if not ret:
#             continue

#         gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#         faces = face_classifier.detectMultiScale(gray)

#         for (x, y, w, h) in faces:
#             roi_gray = gray[y:y + h, x:x + w]
#             roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)

#             if np.sum([roi_gray]) != 0:
#                 roi = roi_gray.astype('float') / 255.0
#                 roi = img_to_array(roi)
#                 roi = np.expand_dims(roi, axis=0)

#                 prediction = classifier.predict(roi)[0]
#                 label = emotion_labels[prediction.argmax()]

#                 if label in ['Sad', 'Angry', 'Fear']:
#                     depression_score += 1

#                 frame_captured = True
#                 break
#         if frame_captured:
#             break

#     cap.release()

#     depression_level = min(5, depression_score)  # 0–5 scale
#     return depression_level
