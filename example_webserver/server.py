

WAIT_FOR_START_COMMAND = False

if __name__ == '__main__':
    server = "0.0.0.0"
    port = 5025

    print(f"STT speech to text server")
    print(f"runs on http://{server}:{port}")
    print()
    print("starting")
    print("└─ ... ", end='', flush=True)


    from RealtimeSTT import AudioToTextRecorder
    from fastapi.responses import HTMLResponse
    from fastapi import FastAPI, WebSocket
    from colorama import Fore, Style
    from pyngrok import ngrok
    import threading as th
    import nest_asyncio
    import numpy as np
    import websockets
    import colorama
    import asyncio
    import uvicorn
    import shutil
    import queue
    import json
    import time
    import os

    colorama.init()

    first_chunk = True
    full_sentences = []
    displayed_text = ""
    message_queue = queue.Queue()
    start_recording_event = th.Event()
    start_transcription_event = th.Event()
    connected_clients = set()
    audio_chunks = queue.Queue()

    def clear_console():
        os.system('clear' if os.name == 'posix' else 'cls')

    def add_message_to_queue(type: str, content):
        message = {
            "type": type,
            "content": content
        }
        message_queue.put(message)

    def fill_cli_line(text):
        columns, _ = shutil.get_terminal_size()
        return text.ljust(columns)[-columns:]

    def text_detected(text):
        global displayed_text, first_chunk

        if text != displayed_text:
            first_chunk = False
            displayed_text = text
            add_message_to_queue("realtime", text)

            message = fill_cli_line(text)

            message = "└─ " + Fore.CYAN + message[:-3] + Style.RESET_ALL
            print(f"\r{message}", end='', flush=True)

    async def broadcast(message_obj):
        if connected_clients:
            for client in connected_clients:
                await client.send(json.dumps(message_obj))

    async def send_handler():
        while True:
            while not message_queue.empty():
                message = message_queue.get()
                await broadcast(message)
            await asyncio.sleep(0.02)

    def recording_started():
        add_message_to_queue("record_start", "")

    def vad_detect_started():
        add_message_to_queue("vad_start", "")

    def wakeword_detect_started():
        add_message_to_queue("wakeword_start", "")

    def transcription_started():
        add_message_to_queue("transcript_start", "")

    recorder_config = {
        'spinner': False,
        'model': 'small.en',
        'language': 'en',
        'silero_sensitivity': 0.01,
        'webrtc_sensitivity': 3,
        'silero_use_onnx': False,
        'post_speech_silence_duration': 1.2,
        'min_length_of_recording': 0.2,
        'min_gap_between_recordings': 0,
        'enable_realtime_transcription': True,
        'realtime_processing_pause': 0,
        'use_microphone': False,
        'realtime_model_type': 'tiny.en',
        'on_realtime_transcription_stabilized': text_detected,
        'on_recording_start': recording_started,
        'on_vad_detect_start': vad_detect_started,
        'on_wakeword_detection_start': wakeword_detect_started,
        'on_transcription_start': transcription_started,
    }

    recorder = AudioToTextRecorder(**recorder_config)

    def transcriber_thread():
        while True:
            start_transcription_event.wait()
            text = "└─ transcribing ... "
            text = fill_cli_line(text)
            print(f"\r{text}", end='', flush=True)
            while not audio_chunks.empty():
                chunk = audio_chunks.get()
                # convert bytes to ndarray
                chunk = np.frombuffer(chunk, dtype=np.int16)
                chunk = chunk.reshape(-1, 1)
                recorder.feed_audio(chunk)
            sentence = recorder.text()
            print(Style.RESET_ALL + "\r└─ " + Fore.YELLOW + sentence + Style.RESET_ALL)
            add_message_to_queue("full", sentence)
            start_transcription_event.clear()
            if WAIT_FOR_START_COMMAND:
                print("waiting for start command")
                print("└─ ... ", end='', flush=True)

    th.Thread(target=transcriber_thread, daemon=True).start()

    ngrok_tunnel = ngrok.connect(port, url="dolphin-rare-buck.ngrok-free.app")
    print('Public URL:', ngrok_tunnel.public_url)
    nest_asyncio.apply()

    app = FastAPI()

    @app.websocket("/ws")
    async def websocket_endpoint(websocket: WebSocket):
        await websocket.accept()
        print("\r└─ OK")
        if WAIT_FOR_START_COMMAND:
            print("waiting for start command")
            print("└─ ... ", end='', flush=True)

        connected_clients.add(websocket) 
        try:
            while True:
                message = await websocket.receive()
                if message["type"] == "websocket.receive":
                    if "text" in message:
                        data = json.loads(message["text"])
                        audio_format = data.get("audio_format")
                        transcription_config = data.get("transcription_config")
                        print(f"Audio Format: {audio_format}")
                        print(f"Transcription Config: {transcription_config}")
                        print("\r└─ OK")
                        start_recording_event.set()
                        start_transcription_event.set()
                    elif "bytes" in message:
                        audio_chunks.put(message["bytes"])

        except json.JSONDecodeError:
            print(Fore.RED + "STT Received an invalid JSON message." + Style.RESET_ALL)
        except websockets.ConnectionClosedError:
            print(Fore.RED + "connection closed unexpectedly by the client" + Style.RESET_ALL)
        except websockets.exceptions.ConnectionClosedOK:
            print("connection closed.")
        finally:
            print("client disconnected")
            connected_clients.remove(websocket)
            print("waiting for clients")
            print("└─ ... ", end='', flush=True)

    print("Server ready")
    uvicorn.run(app, host=server, port=port)
    loop = asyncio.get_event_loop()

    print("\r└─ OK")
    print("waiting for clients")
    print("└─ ... ", end='', flush=True)

    loop.create_task(send_handler())
    loop.run_forever()