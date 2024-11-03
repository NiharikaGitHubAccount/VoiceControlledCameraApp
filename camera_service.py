import cv2
import numpy as np
import time
import threading
from multiprocessing import Queue
from gtts import gTTS
import playsound
import os  # Importing os for file operations

feedback_playing = False  # Global variable to track if voice feedback is playing

def speak(text):
    """ Function for voice feedback """
    global feedback_playing
    feedback_playing = True
    tts = gTTS(text=text, lang='en')
    tts.save("feedback.mp3")
    playsound.playsound("feedback.mp3")
    os.remove("feedback.mp3")  # Remove the temporary audio file after playing
    feedback_playing = False  # Reset flag after playback

def threaded_speak(text):
    """ Run speak in a separate thread to avoid blocking the main thread. """
    thread = threading.Thread(target=speak, args=(text,))
    thread.start()

def apply_warm_filter(frame):
    warm_filter = np.array([0.8, 0.9, 1.2])
    return cv2.convertScaleAbs(frame * warm_filter)

def apply_cool_filter(frame):
    cool_filter = np.array([1.2, 1.0, 0.8])
    return cv2.convertScaleAbs(frame * cool_filter)

def apply_cyberpunk_filter(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hsv[..., 1] = hsv[..., 1] * 1.5  # Increase saturation
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

def apply_vivid_filter(frame):
    return cv2.convertScaleAbs(frame, alpha=1.3, beta=40)

def capture_photo(frame, filename, current_filter=None):
    """Capture and save the photo with the applied filter."""
    if current_filter:
        frame = current_filter(frame)
    cv2.imwrite(filename, frame)
    print(f"Photo captured: {filename}")

def set_timer(frame, duration, command_queue, current_filter=None):
    """Capture photo after a timer with the applied filter."""
    for remaining in range(duration, 0, -1):
        command_queue.put(remaining)  # Send remaining time to the main loop
        time.sleep(1)
    capture_photo(frame, 'timed_image.jpg', current_filter)
    print("Timer photo captured!")
    command_queue.put("timer_done")
    threaded_speak("Timer photo captured!")

def burst_mode(frame, current_filter=None):
    """Capture multiple photos in burst mode with the applied filter."""
    print("Burst mode activated!")
    threaded_speak("Burst mode activated!")
    for i in range(5):
        capture_photo(frame, f'burst_image_{i}.jpg', current_filter)
        time.sleep(0.5)

def start_camera(command_queue, frame_queue):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Camera is not accessible")
        return

    portrait_mode_active = False
    night_mode_active = False
    zoom_level = 1.0
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    timer_active = False
    timer_duration = 0
    current_filter = None  # Initialize current filter

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        original_frame = frame.copy()  # Keep a copy of the original frame without drawings

        # Process command from queue
        while not command_queue.empty():
            command = command_queue.get()

            # Handle each command
            if isinstance(command, int):
                timer_duration = command
            elif command == "capture" and not feedback_playing:
                capture_photo(original_frame, 'captured_image.jpg', current_filter)
            elif command == "zoom_in":
                zoom_level *= 1.5
            elif command == "zoom_out":
                zoom_level /= 1.5
            elif command == "burst_mode":
                threading.Thread(target=burst_mode, args=(original_frame, current_filter)).start()
            elif command.startswith("set_timer"):
                duration = 5
                if len(command.split()) > 2:
                    duration = int(command.split()[-1])
                timer_active = True
                threading.Thread(target=set_timer, args=(original_frame, duration, command_queue, current_filter)).start()
            elif command == "portrait_mode":
                night_mode_active = False
                portrait_mode_active = True
                if not feedback_playing:
                    threaded_speak("Portrait mode activated.")
            elif command == "night_mode":
                portrait_mode_active = False
                night_mode_active = True
                if not feedback_playing:
                    threaded_speak("Night mode activated.")
            elif command == "quit":
                print("Quitting...")
                threaded_speak("Quitting the application.")
                cap.release()
                cv2.destroyAllWindows()
                return
            elif command == "timer_done":
                timer_active = False
            elif command == "warm_filter":
                current_filter = apply_warm_filter
            elif command == "cool_filter":
                current_filter = apply_cool_filter
            elif command == "cyberpunk_filter":
                current_filter = apply_cyberpunk_filter
            elif command == "vivid_filter":
                current_filter = apply_vivid_filter

        # Apply zoom
        if zoom_level != 1.0:
            height, width = frame.shape[:2]
            new_width = int(width / zoom_level)
            new_height = int(height / zoom_level)
            x1 = int((width - new_width) / 2)
            y1 = int((height - new_height) / 2)
            x2 = x1 + new_width
            y2 = y1 + new_height
            frame = frame[y1:y2, x1:x2]
            frame = cv2.resize(frame, (width, height))
            cv2.putText(frame, "Zoom Level: {:.2f}".format(zoom_level), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # Apply the current filter
        if current_filter is not None:
            frame = current_filter(frame)

        # Portrait mode
        if portrait_mode_active:
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5)
            blurred_frame = cv2.GaussianBlur(frame, (21, 21), 0)
            mask = np.zeros_like(frame)
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                mask[y:y + h, x:x + w] = frame[y:y + h, x:x + w]
            frame = np.where(mask == 0, blurred_frame, frame)
            cv2.putText(frame, "Portrait Mode", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # Night mode
        if night_mode_active:
            frame = cv2.convertScaleAbs(frame, alpha=1.2, beta=30)
            cv2.putText(frame, "Night Mode", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # Timer display
        if timer_active:
            cv2.putText(frame, f"Timer: {timer_duration}", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)

        # Send frame to queue for web display
        if frame_queue.full():
            frame_queue.get()  # Remove the oldest frame to maintain queue size
        frame_queue.put(frame)

        cv2.imshow('Camera Feed', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            threaded_speak("Quitting the application.")
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    command_queue = Queue()
    frame_queue = Queue(maxsize=1)  # Limit to 1 frame to prevent overflow
    start_camera(command_queue, frame_queue)