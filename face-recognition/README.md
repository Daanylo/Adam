# Face Recognition Application

A Python-based face recognition application that uses your webcam to recognize faces from a local database. When an unknown face is detected, the application prompts you to label it and stores it for future recognition.

## 🎯 Features

- **Real-time face recognition** using webcam
- **Local SQLite database** for storing face data
- **Interactive face labeling** for unknown faces
- **Live video feed** with bounding boxes and name labels
- **Database management tools** for adding, removing, and listing faces
- **Simple template matching** for face recognition (no complex dependencies)

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install opencv-python opencv-contrib-python numpy
```

### 2. Run the Application
```bash
python basic_face_recognition.py
```

## 📋 Available Applications

This repository contains multiple face recognition implementations:

### 1. Basic Face Recognition (`basic_face_recognition.py`) - **RECOMMENDED**
- Uses OpenCV's built-in face detection
- Template matching for face recognition
- Works without complex dependencies
- Most reliable and easy to set up

**Controls:**
- Press `q` to quit
- Press `c` to manually capture and label a face
- When unknown face detected, check console for labeling prompt

### 2. Advanced Face Recognition (`face_recognition_app.py`)
- Uses the `face_recognition` library for high accuracy
- Requires `dlib` installation (can be complex on some systems)
- Better recognition accuracy but harder to install

### 3. Simple OpenCV Version (`simple_face_recognition.py`)
- Basic implementation using only OpenCV
- Good for learning and understanding the concepts

## 🛠 Database Management

Use the database management tool for the basic version:

```bash
# List all faces in database
python manage_simple_db.py list

# Add face from image file
python manage_simple_db.py add --name "John Doe" --image "photo.jpg"

# Delete a face
python manage_simple_db.py delete --name "John Doe"

# Show a face from database
python manage_simple_db.py show --name "John Doe"

# Clear entire database
python manage_simple_db.py clear
```

## 📁 File Structure

```
deepface/
├── basic_face_recognition.py      # Main application (RECOMMENDED)
├── simple_face_recognition.py     # Simple OpenCV version
├── face_recognition_app.py        # Advanced version (requires dlib)
├── face_database.py              # Database operations (advanced)
├── manage_simple_db.py           # Database manager (basic)
├── manage_database.py            # Database manager (advanced)
├── requirements.txt              # Python dependencies
├── simple_faces.db              # SQLite database (created automatically)
└── README.md                    # This file
```

## 🔧 How It Works

### Basic Version (Recommended)
1. **Face Detection**: Uses OpenCV's Haar cascades for face detection
2. **Template Matching**: Compares detected faces with stored templates
3. **Database Storage**: Stores normalized face images in SQLite database
4. **Recognition**: Uses correlation coefficient for face matching

### Process Flow:
1. Camera captures video frames
2. Each frame is processed for face detection
3. Detected faces are compared with known faces in database
4. If no match found, user is prompted to label the face
5. New faces are stored and future frames will recognize them

## 💡 Usage Tips

### For Better Recognition:
- **Good lighting**: Ensure adequate lighting when adding faces
- **Face the camera**: Look directly at the camera when being recorded
- **Clear images**: Use high-quality images when adding via management tool
- **Multiple samples**: Capture faces from different angles for better recognition

### Troubleshooting:
1. **Camera not working**: Make sure webcam is connected and not used by another app
2. **Face not detected**: Ensure good lighting and face the camera directly
3. **Poor recognition**: Try adjusting the recognition threshold in the code
4. **Performance issues**: Close other applications to free up CPU/memory

## ⚙️ Configuration

You can adjust recognition sensitivity by modifying the threshold in `basic_face_recognition.py`:

```python
def compare_faces(self, face1, face2, threshold=0.8):  # Adjust this value
```

- Lower threshold (0.6-0.7): More sensitive, may have false positives
- Higher threshold (0.8-0.9): Less sensitive, may miss valid matches

## 🔒 Privacy & Ethics

- This application stores facial data locally on your computer
- Always obtain consent before storing someone's facial data
- Be mindful of privacy laws in your jurisdiction
- Use responsibly and ethically

## 📝 License

This project is for educational and personal use. Please respect privacy and obtain consent before storing anyone's facial data.
