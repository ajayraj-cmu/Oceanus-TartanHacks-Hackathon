import pyttsx3
import wave
import struct
import math
import pathfinder

def generate_tts(text, filename, rate=150):
    """
    Generate a WAV file from the given text using pyttsx3.
    The file will be saved as 'filename'. Adjust the rate as desired.
    Note: The output should be mono and 16-bit PCM for the next stage.
    """
    engine = pyttsx3.init()
    engine.setProperty('rate', rate)
    engine.save_to_file(text, filename)
    engine.runAndWait()

def channel_gains(az):
    """
    Compute relative gains for 6 channels (FL, FR, C, LFE, SL, SR)
    based on an azimuth angle (in radians).
    """
    abs_az_deg = abs(math.degrees(az) % 360)
    if abs_az_deg > 180:
        abs_az_deg = 360 - abs_az_deg  
    front_factor = max(0.0, 1.0 - abs_az_deg / 180.0)
    surround_factor = 1.0 - front_factor

    # Compute left/right factors based on azimuth
    left_factor = 0.5 * (1.0 - (az / (math.pi / 2)))
    right_factor = 0.5 * (1.0 + (az / (math.pi / 2)))
    left_factor = max(0.0, min(1.0, left_factor))
    right_factor = max(0.0, min(1.0, right_factor))

    # Center gain decreases as azimuth deviates from straight ahead
    center = max(0.0, 1.0 - (abs_az_deg / 90.0))
    lfe = 0.2  # Fixed low-frequency effect gain

    fl = front_factor * left_factor
    fr = front_factor * right_factor
    sl = surround_factor * left_factor
    sr = surround_factor * right_factor
    c  = front_factor * center

    # Normalize if any value exceeds 1.0
    max_val = max(fl, fr, c, lfe, sl, sr)
    if max_val > 1.0:
        fl /= max_val
        fr /= max_val
        sl /= max_val
        sr /= max_val
        c  /= max_val

    return (fl, fr, c, lfe, sl, sr)

def create_surround_words(input_filename, output_filename, azimuth_degrees=0.0, distance=1.0):
    """
    Reads a mono, 16-bit PCM WAV file (input_filename) containing spoken words,
    applies surround panning based on the specified azimuth (degrees) and distance,
    and writes a new 6-channel WAV file (output_filename).
    """
    num_channels_out = 6
    distance_amplitude = 1.0 / (1.0 + distance)
    azimuth_radians = math.radians(azimuth_degrees)
    gains = channel_gains(azimuth_radians)

    # Open the input file (must be mono, 16-bit PCM)
    with wave.open(input_filename, 'rb') as wf_in:
        num_channels_in = wf_in.getnchannels()
        sample_width = wf_in.getsampwidth()
        frame_rate = wf_in.getframerate()
        num_frames = wf_in.getnframes()

        if num_channels_in != 1 or sample_width != 2:
            raise ValueError("Input file must be mono and 16-bit PCM")
            
        frames = wf_in.readframes(num_frames)
        input_format = '<' + ('h' * num_frames)
        samples = struct.unpack(input_format, frames)

    # Open the output file for writing 6-channel audio
    with wave.open(output_filename, 'wb') as wf_out:
        wf_out.setnchannels(num_channels_out)
        wf_out.setsampwidth(2)
        wf_out.setframerate(frame_rate)
        
        for s in samples:
            sample_norm = s / 32767.0
            sample_norm *= distance_amplitude
            channel_samples = []
            for g in gains:
                val = sample_norm * g
                sample_int = int(val * 32767)
                sample_int = max(-32768, min(32767, sample_int))
                channel_samples.append(sample_int)
            data = struct.pack('<hhhhhh', *channel_samples)
            wf_out.writeframesraw(data)

if __name__ == "__main__":
    # Define the text you want to convert to speech
    text = "Hello, this is a test of our surround sound system using text-to-speech."
    
    # Filenames for the TTS output and the final surround output
    tts_filename = "words.wav"
    surround_filename = "surround_words.wav"
    
    # Generate the TTS WAV file (ensure it's mono, 16-bit PCM)
    generate_tts(text, tts_filename, rate=150)
    
    # Process the TTS file to create a surround-sound version
    # Adjust azimuth_degrees and distance as desired
    create_surround_words(
        input_filename=tts_filename,
        output_filename=surround_filename,
        azimuth_degrees=270,
        distance=2.0
    )
    
    print("Surround words file created as:", surround_filename)
