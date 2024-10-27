from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
from keras.models import load_model
from collections import deque
import numpy as np
import pickle
import cv2
import pygame
pygame.mixer.init()

app = Flask(__name__)

Sports_Labels= set(['badminton','chess','football'])
print("image being loaded")
data=[]
labels=[]
data = np.array(data)
labels = np.array(labels)
alarm_sound = pygame.mixer.Sound("alarm.mp3")

# Load the model and label binarizer
model = load_model("vc\\VideoClassificationModel.keras")
with open("vc\\videoclassificationbinarizer.pickle", "rb") as f:
    lb = pickle.load(f)
mean = np.array([123.68, 116.779, 103.939], dtype="float32")
Queue = deque(maxlen=128)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    if file:
        filename = secure_filename(file.filename)
        file.save(filename)
        return redirect(url_for('predict', filename=filename))

@app.route('/predict/<filename>')
def predict(filename):
    capture_video = cv2.VideoCapture(filename)
    if not capture_video.isOpened():
        return "Error opening video file"
    writer = None
    Width, Height = None, None
    Queue = []
    label = None # Initialize label with a default value

    while True:
        taken, frame = capture_video.read()
        if not taken:
            break
        if Width is None or Height is None:
            Width, Height = frame.shape[:2]
        output = frame.copy()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (224, 224)).astype("float32")
        frame -= mean
        preds = model.predict(np.expand_dims(frame, axis=0))[0]
        Queue.append(preds)
        results = np.array(Queue).mean(axis=0)
        i = np.argmax(results)
        label = lb.classes_[i] # label is assigned a value here
        text = "They are Playing : {} ".format(label)
        cv2.putText(output, text, (45, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.25, (255, 0, 0), 5)
        if writer is None:
            fourcc = cv2.VideoWriter_fourcc(*"XVID")
            writer = cv2.VideoWriter("outputvideo.avi", fourcc, 30, (Width, Height), True)
        writer.write(output)
        cv2.imshow("In Progress", output)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

        # Play alarm sound if the predicted label is "badminton"
        if label == '1':
            alarm_sound.play()

    writer.release()
    capture_video.release()
    cv2.destroyAllWindows()
    return "Video processed and result saved as outputvideo.avi"

if __name__ == '__main__':
    app.run(debug=True)
