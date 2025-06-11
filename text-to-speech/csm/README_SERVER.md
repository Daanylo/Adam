# Sesame CSM TTS Server Setup Guide

## 🚀 Quick Start

You now have multiple ways to run the TTS server with improved performance:

### Option 1: PowerShell Script (Recommended)
```powershell
cd C:\Users\Daan\Documents\Sesame
.\csm\start_server.ps1
```

### Option 2: Direct Python (Manual)
```powershell
cd C:\Users\Daan\Documents\Sesame
venv\Scripts\activate
python .\csm\server.py
```

### Option 3: Single Generation
```powershell
cd C:\Users\Daan\Documents\Sesame
.\csm\start_server.ps1 -Text "Hello world"
```

## 🎯 Features

### ✅ Solved Issues:
- **No more model reloading**: Model stays in memory
- **Smart caching**: Downloads cached in `./model_cache/`
- **Interactive mode**: Generate multiple files without restarting
- **Fast generation**: ~2-5 seconds after initial load

### 🎮 Interactive Commands:
Once the server is running, you can use:
- `generate Hello world` - Generate audio
- `speaker 1` - Change speaker ID
- `length 10000` - Set max length (10 seconds)
- `output my_file.wav` - Change output filename
- `context` - Show conversation context
- `clear` - Clear context
- `help` - Show all commands
- `quit` - Exit server

## 📊 Performance:
- **First run**: ~15-30 seconds (download + load)
- **Subsequent generations**: ~2-5 seconds
- **Model size**: Downloads once, cached permanently
- **Memory**: Model stays loaded for instant generation

## 🔧 Troubleshooting:

### If you get "virtual environment not found":
Make sure you have activated your venv and installed dependencies:
```powershell
venv\Scripts\activate
pip install -r csm\requirements.txt
```

### If downloads keep happening:
The cache directory should be created automatically at `csm\model_cache\`. 
If issues persist, manually create it or check disk space.

### If CUDA errors occur:
Try using CPU mode:
```powershell
.\csm\start_server.ps1 -Device cpu
```

## 💡 Tips:
- The server keeps running - generate as many files as you want!
- Use different speakers (0, 1, 2, etc.) for variety
- Adjust length for longer/shorter audio
- Context feature allows conversation-style generation
- All files are cached, so restarts are much faster
