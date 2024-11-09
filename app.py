from fastapi import FastAPI, WebSocket, BackgroundTasks
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from multiprocessing import Process, Queue
import uvicorn
import cv2

from voice_service import recognize_voice
from camera_service import start_camera

app = FastAPI()

# Serve static files (like your HTML)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Shared command queue and frame queue between camera and voice services
command_queue = Queue()
frame_queue = Queue()

@app.get("/")
async def get():
    """Serve the main HTML interface."""
    return HTMLResponse(open("static/index.html").read())

def generate_frame():
    """Generate frames from the frame queue for video streaming."""
    while True:
        if not frame_queue.empty():
            frame = frame_queue.get()
            _, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.get("/video_feed")
async def video_feed():
    """Endpoint for serving video feed to the frontend."""
    return StreamingResponse(generate_frame(), media_type='multipart/x-mixed-replace; boundary=frame')

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time command handling."""
    await websocket.accept()
    while True:
        data = await websocket.receive_text()
        command_queue.put(data)
        await websocket.send_text(f"Command received: {data}")

@app.on_event("startup")
def start_services():
    """Start the camera and voice services in parallel processes."""
    camera_process = Process(target=start_camera, args=(command_queue, frame_queue))
    voice_process = Process(target=recognize_voice, args=(command_queue,))

    camera_process.start()
    voice_process.start()

# Run the FastAPI app if executing locally
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
