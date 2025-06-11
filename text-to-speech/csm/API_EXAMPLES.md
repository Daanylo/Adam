# TTS API Examples

## Starting the API Server

```powershell
# Start with default settings (localhost:5000)
.\csm\start_api.ps1

# Start on different port
.\csm\start_api.ps1 -Port 8080

# Start with CPU instead of CUDA
.\csm\start_api.ps1 -Device cpu

# Start with debug mode
.\csm\start_api.ps1 -Debug
```

## API Endpoints

### Health Check
```bash
curl http://127.0.0.1:5000/health
```

### Check if Model is Ready
```bash
curl http://127.0.0.1:5000/ready
```

### Generate Speech
```bash
curl -X POST http://127.0.0.1:5000/generate \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Hello world, this is a test!",
    "speaker": 0,
    "max_length_ms": 10000
  }'
```

### Generate Speech with Different Speaker
```bash
curl -X POST http://127.0.0.1:5000/generate \
  -H "Content-Type: application/json" \
  -d '{
    "text": "This is speaker number one speaking.",
    "speaker": 1,
    "max_length_ms": 15000
  }'
```

### Download Audio File
```bash
# Replace 'filename.wav' with actual filename from generate response
curl -O http://127.0.0.1:5000/audio/filename.wav
```

### List All Audio Files
```bash
curl http://127.0.0.1:5000/files
```

### Get Server Info
```bash
curl http://127.0.0.1:5000/info
```

## PowerShell Examples

### Generate Speech
```powershell
$body = @{
    text = "Hello from PowerShell!"
    speaker = 0
    max_length_ms = 8000
} | ConvertTo-Json

$response = Invoke-RestMethod -Uri "http://127.0.0.1:5000/generate" -Method Post -Body $body -ContentType "application/json"
Write-Host "Generated: $($response.filename)"
```

### Download Audio File
```powershell
$filename = "tts_12345678_1234567890.wav"  # Use actual filename
Invoke-WebRequest -Uri "http://127.0.0.1:5000/audio/$filename" -OutFile $filename
```

## Python Examples

### Simple Generation
```python
import requests

response = requests.post("http://127.0.0.1:5000/generate", json={
    "text": "Hello from Python!",
    "speaker": 0
})

result = response.json()
print(f"Audio saved as: {result['filename']}")
```

### Using the Client Class
```python
from client_example import TTSClient

client = TTSClient()
client.wait_until_ready()

result = client.generate_speech(
    "This is much easier to use!",
    speaker=0,
    download=True
)

print(f"Saved to: {result['local_path']}")
```

## LLM Integration Examples

### OpenAI Function Call Schema
```json
{
  "name": "speak_text",
  "description": "Convert text to speech using TTS API",
  "parameters": {
    "type": "object",
    "properties": {
      "text": {
        "type": "string",
        "description": "The text to convert to speech"
      },
      "speaker": {
        "type": "integer",
        "description": "Speaker ID (0, 1, 2, etc.)",
        "default": 0
      }
    },
    "required": ["text"]
  }
}
```

### Simple LLM Integration Function
```python
def speak_response(text: str) -> str:
    """Function that LLM can call to speak its response."""
    import requests
    
    try:
        response = requests.post("http://127.0.0.1:5000/generate", json={
            "text": text,
            "speaker": 0
        })
        
        if response.status_code == 200:
            result = response.json()
            return f"🔊 Speaking: '{text}' (saved as {result['filename']})"
        else:
            return f"❌ TTS Error: {response.status_code}"
    except Exception as e:
        return f"❌ TTS Error: {e}"
```

## Response Format

### Successful Generation
```json
{
  "status": "success",
  "text": "Hello world",
  "speaker": 0,
  "duration_seconds": 2.5,
  "generation_time_seconds": 3.2,
  "sample_rate": 24000,
  "filename": "tts_a1b2c3d4_1234567890.wav",
  "filepath": "./audio_output/tts_a1b2c3d4_1234567890.wav"
}
```

### Error Response
```json
{
  "error": "Model not ready yet",
  "status": "loading",
  "message": "Please wait for the model to finish loading"
}
```
