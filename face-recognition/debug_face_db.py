"""
Database Inspection Tool for Face Recognition
============================================

This tool helps debug face recognition issues by showing what's stored in the database
and testing face comparison directly.
"""

import cv2
import numpy as np
from improved_face_recognition import SimpleFaceDatabase, FaceRecognitionApp
import argparse

def show_stored_face(db, name):
    """Display a face stored in the database"""
    faces = db.get_all_faces()
    
    if name not in faces:
        print(f"No face named '{name}' found in database")
        return
    
    face_data = faces[name]
    print(f"Face '{name}' - Shape: {face_data.shape}, Type: {face_data.dtype}")
    
    # Display the face
    face_display = cv2.resize(face_data, (300, 300))
    cv2.imshow(f'Stored Face: {name} - Press any key to close', face_display)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def test_face_comparison(db, name, image_path):
    """Test comparing a stored face with a new image"""
    faces = db.get_all_faces()
    
    if name not in faces:
        print(f"No face named '{name}' found in database")
        return
    
    # Load test image
    test_image = cv2.imread(image_path)
    if test_image is None:
        print(f"Could not load image: {image_path}")
        return
    
    # Convert to grayscale and detect faces
    gray = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces_detected = face_cascade.detectMultiScale(gray, 1.1, 4)
    
    if len(faces_detected) == 0:
        print("No faces detected in the test image")
        return
    
    # Use the largest face
    largest_face = max(faces_detected, key=lambda f: f[2] * f[3])
    x, y, w, h = largest_face
    
    # Extract and normalize face
    face_roi = gray[y:y+h, x:x+w]
    face_normalized = cv2.resize(face_roi, (100, 100))
    face_normalized = cv2.equalizeHist(face_normalized)
    
    # Get stored face
    stored_face = faces[name]
    
    # Create app instance for comparison
    app = FaceRecognitionApp(debug=True)
    
    # Test comparison
    print(f"\nTesting comparison between stored '{name}' and image '{image_path}':")
    is_match, confidence = app.compare_faces(face_normalized, stored_face)
    
    print(f"Match: {is_match}")
    print(f"Confidence: {confidence:.3f}")
    print(f"Threshold: 0.6")
    
    # Show both images side by side
    stored_display = cv2.resize(stored_face, (200, 200))
    test_display = cv2.resize(face_normalized, (200, 200))
    
    # Create side-by-side comparison
    comparison = np.hstack([stored_display, test_display])
    
    # Add labels
    cv2.putText(comparison, f'Stored: {name}', (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    cv2.putText(comparison, f'Test Image', (210, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    cv2.putText(comparison, f'Confidence: {confidence:.3f}', (10, 220), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    cv2.putText(comparison, f'Match: {is_match}', (210, 220), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    
    cv2.imshow('Face Comparison - Press any key to close', comparison)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def test_with_camera(db, name):
    """Test recognition with live camera feed"""
    faces = db.get_all_faces()
    
    if name not in faces:
        print(f"No face named '{name}' found in database")
        return
    
    print(f"Testing recognition of '{name}' with camera...")
    print("Press 'q' to quit, 'space' to test current frame")
    
    # Create app instance
    app = FaceRecognitionApp(debug=True)
    
    # Initialize camera
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Could not open camera")
        return
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Detect faces
            faces_detected, gray = app.detect_faces(frame)
            
            for (x, y, w, h) in faces_detected:
                # Extract and normalize face
                face_roi = gray[y:y+h, x:x+w]
                face_normalized = cv2.resize(face_roi, (100, 100))
                face_normalized = cv2.equalizeHist(face_normalized)
                
                # Test recognition
                recognized_name, confidence = app.recognize_face(face_normalized)
                
                # Draw results
                color = (0, 255, 0) if recognized_name == name else (0, 0, 255)
                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                
                display_text = f"{recognized_name or 'Unknown'} ({confidence:.3f})"
                cv2.putText(frame, display_text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            # Add instructions
            cv2.putText(frame, "Press 'q' to quit, 'space' to test", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            
            cv2.imshow('Camera Test', frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord(' '):
                print(f"Current recognition results shown above")
    
    finally:
        cap.release()
        cv2.destroyAllWindows()

def main():
    parser = argparse.ArgumentParser(description='Face Recognition Database Inspector')
    parser.add_argument('action', choices=['show', 'test', 'camera'], 
                       help='Action to perform')
    parser.add_argument('--name', required=True, help='Name of person in database')
    parser.add_argument('--image', help='Path to test image (for test action)')
    parser.add_argument('--db', default='simple_faces.db', help='Database file path')
    
    args = parser.parse_args()
    
    db = SimpleFaceDatabase(args.db)
    
    if args.action == 'show':
        show_stored_face(db, args.name)
    
    elif args.action == 'test':
        if not args.image:
            print("Error: --image is required for test action")
            return
        test_face_comparison(db, args.name, args.image)
    
    elif args.action == 'camera':
        test_with_camera(db, args.name)

if __name__ == "__main__":
    main()
