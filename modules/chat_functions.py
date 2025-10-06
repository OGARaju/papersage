import io
import struct
import requests
import time
import os
import tempfile
import base64
import json

from scripts.chat import TTS_API_BASE_URL, TTS_MODEL, API_KEY


def pcm_to_wav_bytes(pcm16_bytes: bytes, sample_rate: int, num_channels: int = 1) -> bytes:
    """Converts raw 16-bit PCM bytes to a WAV file format."""
    buffer = io.BytesIO()
    data_size = len(pcm16_bytes)

    # 1. RIFF chunk
    buffer.write(b'RIFF')
    buffer.write(struct.pack('<I', 36 + data_size))  # File size
    buffer.write(b'WAVE')

    # 2. FMT chunk
    buffer.write(b'fmt ')
    buffer.write(struct.pack('<I', 16))  # Sub-chunk size
    buffer.write(struct.pack('<H', 1))  # Audio format (1=PCM)
    buffer.write(struct.pack('<H', num_channels))
    buffer.write(struct.pack('<I', sample_rate))
    byte_rate = sample_rate * num_channels * 2  # 2 bytes/sample
    buffer.write(struct.pack('<I', byte_rate))
    block_align = num_channels * 2
    buffer.write(struct.pack('<H', block_align))
    buffer.write(struct.pack('<H', 16))  # Bits per sample

    # 3. DATA chunk
    buffer.write(b'data')
    buffer.write(struct.pack('<I', data_size))  # Data size

    # 4. PCM data
    buffer.write(pcm16_bytes)

    return buffer.getvalue()

def call_tts_api_and_save(text: str) -> str:
    """Calls the TTS API, converts PCM to WAV, and saves to a temporary file."""
    url = f"{TTS_API_BASE_URL}/{TTS_MODEL}:generateContent?key={API_KEY}"

    # Voice: Orus (Firm)
    payload = {
        "contents": [{"parts": [{"text": text}]}],
        "generationConfig": {
            "responseModalities": ["AUDIO"],
            "speechConfig": {
                "voiceConfig": {
                    "prebuiltVoiceConfig": {"voiceName": "Orus"}
                }
            }
        }
    }

    max_retries = 4
    for attempt in range(max_retries):
        try:
            response = requests.post(url, headers={'Content-Type': 'application/json'}, data=json.dumps(payload))
            response.raise_for_status()
            result = response.json()

            part = result.get('candidates', [{}])[0].get('content', {}).get('parts', [{}])[0]
            audio_data_b64 = part.get('inlineData', {}).get('data')
            mime_type = part.get('inlineData', {}).get('mimeType')

            if audio_data_b64 and mime_type and mime_type.startswith("audio/L16"):
                pcm_bytes = base64.b64decode(audio_data_b64)

                import re
                match = re.search(r'rate=(\d+)', mime_type)
                if not match:
                    raise ValueError(f"Could not parse sample rate from MIME type: {mime_type}")
                sample_rate = int(match.group(1))

                wav_bytes = pcm_to_wav_bytes(pcm_bytes, sample_rate)

                temp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
                temp_file.write(wav_bytes)
                temp_file.close()
                return temp_file.name

            raise Exception("TTS API did not return valid audio data.")

        except requests.exceptions.HTTPError as http_err:
            if response.status_code == 429 and attempt < max_retries - 1:
                sleep_time = 2 ** attempt
                time.sleep(sleep_time)
            else:
                print(f"TTS HTTP Error: {http_err} - {response.text}")
                return "Error: Failed to generate audio."
        except Exception as err:
            print(f"TTS Error: {err}")
            return "Error: Failed to process audio."

    return "Error: TTS failed after retries."

def clean_temp_file(audio_path):
    """Utility to clean up the temporary audio file after playback."""
    if audio_path and os.path.exists(audio_path):
        try:
            os.remove(audio_path)
        except Exception as e:
            print(f"Error cleaning up temp file {audio_path}: {e}")
    return None