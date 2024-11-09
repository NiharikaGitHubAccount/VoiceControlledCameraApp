from fastapi import FastAPI, WebSocket, BackgroundTasks
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.responses import StreamingResponse
from multiprocessing import Process, Queue
from voice_service import recognize_voice
from camera_service import start_camera
import cv2
import uvicorn

app = FastAPI()

# Serve static files (like your HTML)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Shared command queue and frame queue between camera and voice services
command_queue = Queue()
frame_queue = Queue()

@app.get("/")
async def get():
    return HTMLResponse(open("static/index.html").read())

def generate_frame():
    """Function to generate frames from the frame queue."""
    while True:
        # Check if there is a frame in the queue
        if not frame_queue.empty():
            frame = frame_queue.get()

            # Encode the frame in JPEG format
            _, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()

            # Yield the frame as a byte stream
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.get("/video_feed")
async def video_feed():
    return StreamingResponse(generate_frame(), media_type='multipart/x-mixed-replace; boundary=frame')

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    while True:
        data = await websocket.receive_text()
        command_queue.put(data)  # Directly send received command to the command queue
        await websocket.send_text(f"Command received: {data}")

@app.on_event("startup")
def start_services():
    # Start both services in parallel processes
    camera_process = Process(target=start_camera, args=(command_queue, frame_queue))
    voice_process = Process(target=recognize_voice, args=(command_queue,))

    camera_process.start()
    voice_process.start()

# Run the FastAPI app if executing locally
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)