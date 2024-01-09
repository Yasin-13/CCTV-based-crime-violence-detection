import numpy as np
import argparse
import cv2
import os
from keras.models import load_model
from collections import deque

def print_results(video):
    print("Loading model ...")
    model = load_model('modelnew.h5')
    Q = deque(maxlen=128)
    vs = cv2.VideoCapture(video)
    (W, H) = (None, None)
    count = 0

    while True:
        (grabbed, frame) = vs.read()

        if not grabbed:
            break

        if W is None or H is None:
            (H, W) = frame.shape[:2]

        output = frame.copy()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (128, 128)).astype("float32")
        frame = frame.reshape(128, 128, 3) / 255

        preds = model.predict(np.expand_dims(frame, axis=0))[0]
        Q.append(preds)

        results = np.array(Q).mean(axis=0)
        i = (preds > 0.50)[0]
        label = i

        text_color = (0, 255, 0) 

        if label:  
            text_color = (0, 0, 255)  

        text = "Violence: {}".format(label)
        FONT = cv2.FONT_HERSHEY_SIMPLEX

        cv2.putText(output, text, (35, 50), FONT, 1.25, text_color, 3)

        cv2.imshow("Violence Detection", output)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

    print("[INFO] Cleaning up...")
    vs.release()
    cv2.destroyAllWindows()

video_file = 'nv60.mp4'
print_results(video_file)
