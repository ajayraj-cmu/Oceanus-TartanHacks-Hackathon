import pyaudio
import boto3
import time
import re

# Configuration
WAKE_WORD = "computer"  # Wake word to trigger listening
AWS_REGION = "us-west-2"  # AWS region for Transcribe
CHUNK = 1024  # Audio chunk size
FORMAT = pyaudio.paInt16  # Audio format
CHANNELS = 1  # Number of audio channels
RATE = 16000  # Sampling rate

# Initialize AWS clients
transcribe_client = boto3.client('transcribestreaming', region_name=AWS_REGION)

class TranscribeStream:
    def __init__(self):
        self.wake_detected = False
        self.command_buffer = []
        self.object_name = None

    def get_audio_stream(self):
        """Generator that yields audio chunks from microphone"""
        p = pyaudio.PyAudio()
        self.stream = p.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=RATE,
            input=True,
            frames_per_buffer=CHUNK
        )
        
        try:
            while True:
                data = self.stream.read(CHUNK, exception_on_overflow=False)
                yield {'AudioEvent': {'AudioChunk': data}}
        finally:
            self.stream.stop_stream()
            self.stream.close()
            p.terminate()

    def parse_command(self, text):
        """Extract object name from command text"""
        patterns = [
            r"where is my (.+?)[\.\?]?$",
            r"locate (.+?)[\.\?]?$",
            r"find (.+?)[\.\?]?$"
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        return None

    def handle_transcript(self, transcript):
        """Process transcript results"""
        if not self.wake_detected:
            if WAKE_WORD in transcript.lower():
                print(f"Wake word '{WAKE_WORD}' detected")
                self.wake_detected = True
        else:
            self.command_buffer.append(transcript)
            full_command = " ".join(self.command_buffer).strip()
            
            # Check for end of command (question mark or period)
            if re.search(r"[\.\?]$", full_command):
                print(f"Processing command: {full_command}")
                self.object_name = self.parse_command(full_command)
                
                # Reset state
                self.wake_detected = False
                self.command_buffer = []

    def listen_for_command(self):
        """Main method to start transcription and processing"""
        audio_stream = self.get_audio_stream()
        
        response = transcribe_client.start_stream_transcription(
            LanguageCode='en-US',
            MediaSampleRateHertz=RATE,
            MediaEncoding='pcm',
            AudioStream=audio_stream
        )
        
        try:
            # Process transcription events
            for event in response['TranscriptResultStream']:
                if 'TranscriptEvent' in event:
                    results = event['TranscriptEvent']['Transcript']['Results']
                    for result in results:
                        if result['IsPartial']:
                            continue
                        transcript = result['Alternatives'][0]['Transcript']
                        self.handle_transcript(transcript)
                        if self.object_name:
                            return self.object_name
        except Exception as e:
            print(f"Error: {str(e)}")
        finally:
            audio_stream.close()

# Usage example
if __name__ == "__main__":
    stream_handler = TranscribeStream()
    print("Listening for wake word...")
    object_name = stream_handler.listen_for_command()
    if object_name:
        print(f"Object to locate: {object_name}")
    else:
        print("No valid command detected")
