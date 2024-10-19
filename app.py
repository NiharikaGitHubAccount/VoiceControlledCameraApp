from fastapi import FastAPI, WebSocket
from multiprocessing import Process, Queue
from voice_service import recognize_voice
from camera_service import start_camera

app = FastAPI()

# Shared command queue between camera and voice services
command_queue = Queue()

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    while True:
        data = await websocket.receive_text()
        if data == "capture":
            command_queue.put("capture")
        elif data == "quit":
            command_queue.put("quit")
            break
        elif data == "zoom in":
            command_queue.put("zoom_in")
        elif data == "zoom out":
            command_queue.put("zoom_out")
        elif data == "burst mode":
            command_queue.put("burst_mode")
        elif data == "set timer":
            command_queue.put("set_timer")
        elif data == "portrait mode":
            command_queue.put("portrait_mode")
        elif data == "night mode":
            command_queue.put("night_mode")
        elif data == "apply warm filter":
            command_queue.put("warm_filter")
        elif data == "apply cool filter":
            command_queue.put("cool_filter")
        elif data == "apply gray filter":
            command_queue.put("gray_filter")
        elif data == "apply cyberpunk filter":
            command_queue.put("cyberpunk_filter")
        elif data == "apply vivid filter":
            command_queue.put("vivid_filter")

        await websocket.send_text(f"Command received: {data}")
    await websocket.close()

if __name__ == "__main__":
    # Start both services in parallel processes
    camera_process = Process(target=start_camera, args=(command_queue,))
    voice_process = Process(target=recognize_voice, args=(command_queue,))

    camera_process.start()
    voice_process.start()

    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)

    camera_process.join()
    voice_process.join()