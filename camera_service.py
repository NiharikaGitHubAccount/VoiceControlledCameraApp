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
    # Apply a warm filter by adjusting the colors
    warm_filter = np.array([1.2, 1.0, 0.8])
    return cv2.convertScaleAbs(frame * warm_filter)

def apply_cool_filter(frame):
    # Apply a cool filter by adjusting the colors
    cool_filter = np.array([0.8, 0.9, 1.2])
    return cv2.convertScaleAbs(frame * cool_filter)

def apply_gray_filter(frame):
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

def apply_cyberpunk_filter(frame):
    # Cyberpunk effect by increasing saturation and contrast
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hsv[..., 1] = hsv[..., 1] * 1.5  # Increase saturation
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

def apply_vivid_filter(frame):
    # Increase brightness and contrast
    return cv2.convertScaleAbs(frame, alpha=1.3, beta=40)

def capture_photo(frame, filename):
    cv2.imwrite(filename, frame)
    print(f"Photo captured: {filename}")

def set_timer(frame, duration, command_queue):
    """ Function to set a timer and capture a photo after the duration """
    for remaining in range(duration, 0, -1):
        command_queue.put(remaining)  # Send remaining time to the main loop
        time.sleep(1)  # Wait for 1 second
    capture_photo(frame, 'timed_image.jpg')
    print("Timer photo captured!")
    command_queue.put("timer_done")  # Notify that the timer is done
    threaded_speak("Timer photo captured!")

def burst_mode(frame):
    """ Function to handle burst mode photo capture """
    print("Burst mode activated!")
    threaded_speak("Burst mode activated!")  # Only feedback for activation
    for i in range(5):
        capture_photo(frame, f'burst_image_{i}.jpg')
        time.sleep(0.5)  # Short delay between captures
    print("Burst mode photos captured!")  # Removed voice feedback for this line

def start_camera(command_queue):
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

            # Check if command is an int for timer
            if isinstance(command, int):
                timer_duration = command
            elif command == "capture" and not feedback_playing:  # Prevents capturing during voice feedback
                capture_photo(original_frame, 'captured_image.jpg')  # Save original frame without rectangle
            elif command == "zoom_in":
                zoom_level *= 1.5
            elif command == "zoom_out":
                zoom_level /= 1.5
            elif command == "burst_mode":
                threading.Thread(target=burst_mode, args=(original_frame,)).start()
            elif command.startswith("set_timer"):
                duration = 5  # Default to 5 seconds
                if len(command.split()) > 2:  # E.g., "set timer for 10"
                    duration = int(command.split()[-1])
                timer_active = True
                threading.Thread(target=set_timer, args=(original_frame, duration, command_queue)).start()
            elif command == "portrait_mode":
                night_mode_active = False  # Remove night mode if active
                portrait_mode_active = True
                if not feedback_playing:
                    threaded_speak("Portrait mode activated.")
            elif command == "night_mode":
                portrait_mode_active = False  # Remove portrait mode if active
                night_mode_active = True
                if not feedback_playing:
                    threaded_speak("Night mode activated.")
            elif command == "quit":
                print("Quitting...")
                threaded_speak("Quitting the application.")
                cap.release()  # Release the camera
                cv2.destroyAllWindows()  # Close all OpenCV windows
                return  # Exit the function to stop the application
            elif command == "timer_done":
                timer_active = False  # Reset timer status
            elif command == "warm_filter":
                current_filter = apply_warm_filter
            elif command == "cool_filter":
                current_filter = apply_cool_filter
            elif command == "gray_filter":
                current_filter = apply_gray_filter
            elif command == "cyberpunk_filter":
                current_filter = apply_cyberpunk_filter
            elif command == "vivid_filter":
                current_filter = apply_vivid_filter

        # Apply zoom if needed
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

        # Apply the current filter if set
        if current_filter is not None:
            frame = current_filter(frame)

        # Portrait mode processing
        if portrait_mode_active:
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5)

            blurred_frame = cv2.GaussianBlur(frame, (21, 21), 0)

            # Create a mask for the detected face
            mask = np.zeros_like(frame)

            for (x, y, w, h) in faces:
                # Draw a rectangle around the face for live feed only
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                # Keep the face sharp by adding it to the mask
                mask[y:y + h, x:x + w] = frame[y:y + h, x:x + w]

            # Combine the original frame with the blurred frame using the mask
            frame = np.where(mask == 0, blurred_frame, frame)

            cv2.putText(frame, "Portrait Mode", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # Night mode processing
        if night_mode_active:
            frame = cv2.convertScaleAbs(frame, alpha=1.2, beta=30)
            cv2.putText(frame, "Night Mode", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # Display countdown timer
        if timer_active:
            cv2.putText(frame, f"Timer: {timer_duration}", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)

        cv2.imshow('Camera Feed', frame)  # Show frame with rectangles and text

        if cv2.waitKey(1) & 0xFF == ord('q'):
            threaded_speak("Quitting the application.")  # Added feedback for quitting when 'q' is pressed
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    command_queue = Queue()
    start_camera(command_queue)