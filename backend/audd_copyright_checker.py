import requests
import tempfile
import subprocess
import os

def extract_audio_from_video(video_path, output_format="mp3"):
    """Extract audio from video using ffmpeg and return path to audio file."""
    tmp_dir = tempfile.gettempdir()
    audio_path = os.path.join(tmp_dir, f"audd_audio_extract.{output_format}")
    cmd = [
        "ffmpeg", "-y", "-i", video_path,
        "-vn", "-acodec", "libmp3lame", "-ar", "44100", "-ac", "2", audio_path
    ]
    subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return audio_path

def check_copyright_with_audd(audio_file_path, api_key="0f9a5d36da40b8cbdfe4809c2145b2ad"):
    url = "https://api.audd.io/"
    with open(audio_file_path, "rb") as f:
        files = {"file": f}
        data = {"api_token": api_key, "return": "apple_music,spotify"}
        response = requests.post(url, data=data, files=files)
    return response.json()

def check_video_copyright(video_path, api_key="0f9a5d36da40b8cbdfe4809c2145b2ad"):
    audio_path = extract_audio_from_video(video_path)
    result = check_copyright_with_audd(audio_path, api_key)
    # Clean up temp audio file
    if os.path.exists(audio_path):
        os.remove(audio_path)
    # Determine violation: true if song detected, false otherwise
    violation = False
    # AudD returns 'result' key with song info if found
    if result and isinstance(result, dict):
        audd_result = result.get('result')
        if audd_result and isinstance(audd_result, dict):
            # Song detected
            violation = True
    return {
        'violation': violation,
        'result': result.get('result') if isinstance(result, dict) else result,
        'raw': result
    }

# Example usage:
if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python audd_copyright_checker.py <video_path>")
        sys.exit(1)
    video_path = sys.argv[1]
    result = check_video_copyright(video_path)
    print(result)
