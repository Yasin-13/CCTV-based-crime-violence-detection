import numpy as np
import argparse
import cv2
from keras.models import load_model
from collections import deque
import telepot
import time
from datetime import datetime
import tkinter as tk
from tkinter import filedialog

# Initialize the Telegram bot with your bot's API token
bot = telepot.Bot('YOUR_BOT_TOKEN')  # Replace with your actual token

def open_file_dialog(entry, title):
    file_path = filedialog.askopenfilename(title=title)
    entry.delete(0, tk.END)
    entry.insert(0, file_path)

def save_annotated_video(input_video, output_video, telegram_group_id):
    # Rest of your code...

# Create the main application window
app = tk.Tk(main.py)
app.title("Violence Detection App")

# Create and set up input fields for video source, output file, and Telegram group ID
input_video_label = tk.Label(app, text="Input Video Source:")
input_video_entry = tk.Entry(app)
output_video_label = tk.Label(app, text="Output Video File:")
output_video_entry = tk.Entry(app)
telegram_group_label = tk.Label(app, text="Telegram Group ID:")
telegram_group_entry = tk.Entry(app)

input_video_label.grid(row=0, column=0)
input_video_entry.grid(row=0, column=1)
output_video_label.grid(row=1, column=0)
output_video_entry.grid(row=1, column=1)
telegram_group_label.grid(row=2, column=0)
telegram_group_entry.grid(row=2, column=1)

browse_button = tk.Button(app, text="Browse", command=lambda: open_file_dialog(input_video_entry, "Select Input Video"))
browse_button.grid(row=0, column=2)

# Create a start button to begin the detection
start_button = tk.Button(app, text="Start Detection", command=lambda: save_annotated_video(input_video_entry.get(), output_video_entry.get(), telegram_group_entry.get()))
start_button.grid(row=3, column=0, columnspan=3)

app.mainloop()
