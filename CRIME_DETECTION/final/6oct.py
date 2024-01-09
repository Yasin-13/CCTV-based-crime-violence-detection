import numpy as np
import cv2
from keras.models import load_model
from collections import deque
import telepot
from datetime import datetime
import os

bot = telepot.Bot('6679358098:AAFmpDc7o4MwqDywDahyAK0Qq89IVZqNr04')

def save_annotated_video(input_video, telegram_group_id, output_video_path):
    print("Loading model ...")
    model = load_model('modelnew.h5')
    Q = deque(maxlen=128)

    vs = cv2.VideoCapture(input_video)
    (W, H) = (None, None)
    frame_count = 0

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_video_path, fourcc, 30.0, (1920, 1080))  # Change the resolution and frame rate as needed

    while True:
        (grabbed, frame) = vs.read()

        if not grabbed:
            break

        if W is None or H is None:
            (H, W) = frame.shape[:2]

        output = frame.copy()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (128, 128)).astype("float32")
        frame = frame / 255.0 

        preds = model.predict(np.expand_dims(frame, axis=0))[0]
        i = (preds > 0.50)[0]

        text = "Violence: {}".format(i)
        FONT = cv2.FONT_HERSHEY_SIMPLEX

        cv2.putText(output, text, (35, 50), FONT, 1.25, (0, 0, 255) if i else (0, 255, 0), 3)

        cv2.imshow("Violence Detection", output)  # Show the detection window

        out.write(output)  # Write annotated frame to the output video

        frame_count += 1

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    print("[INFO] Cleaning up...")
    vs.release()
    out.release()
    cv2.destroyAllWindows()

def send_video_to_telegram(telegram_group_id, output_video_path):
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    video_message = f"Violence detected at {current_time}"

    with open(output_video_path, 'rb') as f:
        bot.sendVideo(telegram_group_id, f, caption=video_message)
    os.remove(output_video_path)

if __name__ == "__main__":
    input_video = 'WV112.mp4'
    telegram_group_id = '-949413618'  
    output_video_path = 'annotated_video.avi'  

    save_annotated_video(input_video, telegram_group_id, output_video_path)

    send_video_to_telegram(telegram_group_id, output_video_path)
