import numpy as np
import cv2
from keras.models import load_model
from collections import deque
import telepot
import time
from datetime import datetime
import os

bot = telepot.Bot('6679358098:AAFmpDc7o4MwqDywDahyAK0Qq89IVZqNr04')  # Replace with your Telegram bot token

def save_annotated_frame(input_video, telegram_group_id):
    print("Loading model ...")
    model = load_model('modelnew.h5')
    Q = deque(maxlen=128)

    vs = cv2.VideoCapture(input_video)
    (W, H) = (None, None)
    violence_detected = False
    frame_count = 0

    fourcc = cv2.VideoWriter_fourcc(*'H264')
    output_video = cv2.VideoWriter('output_videos/output.mp4', fourcc, 20.0, (1920, 1080))
  # Modify the codec and frame rate as needed

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

        if i and not violence_detected: 
            frame_image = (frame * 255).astype(np.uint8) 
            violence_detected = True

        text = "Violence: {}".format(i)
        FONT = cv2.FONT_HERSHEY_SIMPLEX

        cv2.putText(output, text, (35, 50), FONT, 1.25, (0, 0, 255) if i else (0, 255, 0), 3)

        cv2.imshow("Crime Detection", output)
        output_video.write(frame)  # Write the frame to the output video

        key = cv2.waitKey(1) & 0xFF
        frame_count += 1

        if key == ord("q"):
            break

    print("[INFO] Cleaning up...")
    vs.release()
    output_video.release()
    cv2.destroyAllWindows()

    return frame_image

def convert_to_hd_black_and_white(frame_image):
    frame_bw = cv2.cvtColor(frame_image, cv2.COLOR_BGR2GRAY)

    frame_hd = cv2.resize(frame_bw, (1920, 1080), interpolation=cv2.INTER_CUBIC)

    return frame_hd

def send_frame_to_telegram(frame_image, telegram_group_id):
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    frame_message = f"Violence detected at {current_time} "

    frame_bw = convert_to_hd_black_and_white(frame_image)
    frame_filename = 'frame_image_bw.jpg'
    cv2.imwrite(frame_filename, frame_bw)
    with open(frame_filename, 'rb') as f:
        bot.sendPhoto(telegram_group_id, f, caption=frame_message)
    os.remove(frame_filename)

if __name__ == "__main__":
    input_video = 'WV82.mp4' 
    telegram_group_id = '-949413618' 
    frame_image = save_annotated_frame(input_video, telegram_group_id)
    
    if frame_image is not None:
        send_frame_to_telegram(frame_image, telegram_group_id)
