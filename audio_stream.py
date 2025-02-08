from flask import Flask, Response
import pyaudio

app = Flask(__name__)

# Audio settings
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
CHUNK = 1024

audio = pyaudio.PyAudio()
stream = audio.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)

def generate_audio():
    while True:
        data = stream.read(CHUNK)
        yield data

@app.route('/audio_feed')
def audio_feed():
    return Response(generate_audio(), mimetype="audio/wav")

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5001, debug=False, threaded=True)
