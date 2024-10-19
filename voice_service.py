import speech_recognition as sr
from multiprocessing import Queue
import time

def recognize_voice(command_queue):
    recognizer = sr.Recognizer()
    mic = sr.Microphone()

    while True:
        with mic as source:
            print("Listening for voice commands...")
            recognizer.adjust_for_ambient_noise(source)
            audio = recognizer.listen(source)

        try:
            command = recognizer.recognize_google(audio).lower()
            print(f"Recognized command: {command}")

            if "capture" in command:
                command_queue.put("capture")
            elif "quit" in command:
                command_queue.put("quit")
                break
            elif "zoom in" in command:
                command_queue.put("zoom_in")
            elif "zoom out" in command:
                command_queue.put("zoom_out")
            elif "burst mode" in command:
                command_queue.put("burst_mode")
            elif "set timer" in command:
                command_queue.put("set_timer")
            elif "portrait mode" in command:
                command_queue.put("portrait_mode")
            elif "night mode" in command:
                command_queue.put("night_mode")
            elif "apply warm filter" in command:
                command_queue.put("warm_filter")
            elif "apply cool filter" in command:
                command_queue.put("cool_filter")
            elif "apply gray filter" in command:
                command_queue.put("gray_filter")
            elif "apply cyberpunk filter" in command:
                command_queue.put("cyberpunk_filter")
            elif "apply vivid filter" in command:
                command_queue.put("vivid_filter")

        except sr.UnknownValueError:
            print("Could not understand the audio")
        except sr.RequestError:
            print("Error with the recognition service")

if __name__ == "__main__":
    command_queue = Queue()
    recognize_voice(command_queue)