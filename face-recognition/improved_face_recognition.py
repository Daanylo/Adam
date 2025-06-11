import cv2
import numpy as np
import sqlite3
import pickle
import os
from datetime import datetime
import threading
import time
import requests
# --- Add Flask for registration endpoint ---
from flask import Flask, request, jsonify

class SimpleFaceDatabase:
    def __init__(self, db_path="simple_faces.db"):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize the SQLite database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS simple_faces (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT UNIQUE NOT NULL,
                face_data BLOB NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def add_face(self, name, face_data):
        """Add a new face to the database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            face_bytes = pickle.dumps(face_data)
            
            cursor.execute('''
                INSERT OR REPLACE INTO simple_faces (name, face_data) 
                VALUES (?, ?)
            ''', (name, face_bytes))
            
            conn.commit()
            conn.close()
            return True
        except Exception as e:
            print(f"Error adding face to database: {e}")
            return False
    
    def get_all_faces(self):
        """Retrieve all faces from the database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('SELECT name, face_data FROM simple_faces')
            results = cursor.fetchall()
            
            faces = {}
            for name, face_bytes in results:
                face_data = pickle.loads(face_bytes)
                faces[name] = face_data
            
            conn.close()
            return faces
        except Exception as e:
            print(f"Error retrieving faces from database: {e}")
            return {}

class FaceRecognitionApp:
    def __init__(self, debug=True):
        self.db = SimpleFaceDatabase()
        self.known_faces = self.db.get_all_faces()
        self.debug_mode = debug
        self.recognition_threshold = 0.4  # Adjustable recognition threshold
        
        # Load OpenCV's pre-trained face detection model
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # UI state
        self.waiting_for_label = False
        self.last_unknown_time = 0
        self.unknown_cooldown = 5  # seconds between unknown face prompts
        
        # Face event sending state
        self.last_sent_face_id = None
        self.last_sent_time = 0
        self.send_cooldown = 3  # seconds between sending the same face event
        
        # --- Pending faces for registration via API ---
        self.pending_unknown_faces = {}  # face_id -> (face_normalized, frame)
        self.last_unknown_face_id = None
        self.last_unknown_face_data = None  # (face_normalized, frame)
        
        # --- Flask app for registration ---
        self.flask_app = Flask(__name__)
        self._setup_api()
        self._start_api_thread()
        
        # --- Debounce unknown face events ---
        self.unknown_face_counter = 0
        self.unknown_face_streak = 0
        self.unknown_face_id = None
        self.unknown_face_streak_threshold = 10  # Require 10 consecutive unknowns

        print(f"Loaded {len(self.known_faces)} faces from database")
        if self.debug_mode:
            print("Debug mode enabled - detailed recognition info will be shown")
    
    def detect_faces(self, frame):
        """Detect faces in the frame"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 4, minSize=(50, 50))
        return faces, gray
    
    def compare_faces(self, face1, face2, threshold=0.4):
        """Compare two face images using multiple methods"""
        try:
            # Resize both faces to same size
            face1_resized = cv2.resize(face1, (100, 100))
            face2_resized = cv2.resize(face2, (100, 100))
            
            # Apply histogram equalization for better comparison
            face1_eq = cv2.equalizeHist(face1_resized)
            face2_eq = cv2.equalizeHist(face2_resized)
            
            # Method 1: Template matching with normalized correlation
            result1 = cv2.matchTemplate(face1_eq, face2_eq, cv2.TM_CCOEFF_NORMED)
            max_val1 = np.max(result1)
            
            # Method 2: Template matching with correlation coefficient
            result2 = cv2.matchTemplate(face1_eq, face2_eq, cv2.TM_CCORR_NORMED)
            max_val2 = np.max(result2)
            
            # Method 3: Simple structural similarity
            mse = np.mean((face1_eq.astype(np.float32) - face2_eq.astype(np.float32)) ** 2)
            mse_similarity = 1.0 / (1.0 + mse / 1000.0)
            
            # Combine all methods with weights
            combined_score = (max_val1 * 0.5) + (max_val2 * 0.3) + (mse_similarity * 0.2)
            
            # Debug output
            if self.debug_mode:
                print(f"    TM1: {max_val1:.3f}, TM2: {max_val2:.3f}, MSE: {mse_similarity:.3f}, Combined: {combined_score:.3f}")
            
            return combined_score > threshold, combined_score
        except Exception as e:
            print(f"Error comparing faces: {e}")
            return False, 0
    
    def recognize_face(self, face_roi):
        """Recognize a face by comparing with known faces"""
        best_match = None
        best_confidence = 0
        
        if self.debug_mode and len(self.known_faces) > 0:
            print(f"  Comparing against {len(self.known_faces)} known faces:")
        for name, known_face in self.known_faces.items():
            is_match, confidence = self.compare_faces(face_roi, known_face, self.recognition_threshold)
            
            if self.debug_mode:
                print(f"    {name}: {confidence:.3f} {'✓' if is_match else '✗'}")
            
            if is_match and confidence > best_confidence:
                best_match = name
                best_confidence = confidence
        
        if self.debug_mode:
            print(f"  → Best match: {best_match or 'None'} ({best_confidence:.3f})")
        
        return best_match, best_confidence
    
    def add_new_face_interactive(self, face_roi, frame):
        """Add a new face to the database interactively"""
        print("\n" + "="*50)
        print("UNKNOWN FACE DETECTED!")
        print("="*50)
        
        # Save the current frame for reference
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"unknown_face_{timestamp}.jpg"
        cv2.imwrite(filename, frame)
        print(f"Face image saved as '{filename}' for reference")
        
        while True:
            name = input("\nEnter a name for this person (or 'skip' to ignore): ").strip()
            if name.lower() == 'skip':
                print("Skipping face registration")
                self.waiting_for_label = False
                self.unknown_face_id = None  # Reset so new unknowns can be sent
                self.unknown_face_streak = 0
                return False
            
            if name and len(name) > 0:
                # Normalize the face
                face_normalized = cv2.resize(face_roi, (100, 100))
                face_normalized = cv2.equalizeHist(face_normalized)
                
                if self.db.add_face(name, face_normalized):
                    print(f"Successfully added {name} to the database!")
                    self.known_faces[name] = face_normalized
                    self.waiting_for_label = False
                    self.unknown_face_id = None  # Reset so new unknowns can be sent
                    self.unknown_face_streak = 0
                    return True
                else:
                    print("Error adding face to database. Please try again.")
            else:
                print("Please enter a valid name.")
    
    def send_face_event(self, face_id, name=None, face_normalized=None, frame=None):
        """Send a face event to the chat tool API if not sent recently. Store unknown face for registration."""
        now = time.time()
        if face_id == self.last_sent_face_id and (now - self.last_sent_time) < self.send_cooldown:
            return  # Debounce: don't send the same face too often
        self.last_sent_face_id = face_id
        self.last_sent_time = now
        payload = {"face_id": face_id}
        if name:
            payload["name"] = name
        try:
            response = requests.post("http://localhost:5100/face_event", json=payload, timeout=2)
            if self.debug_mode:
                print(f"[FaceRecognition] Sent face event: {payload}, Response: {response.status_code}")
            # --- If unknown, store for registration ---
            if name is None and face_normalized is not None and frame is not None:
                self.pending_unknown_faces[face_id] = (face_normalized, frame)
                self.last_unknown_face_id = face_id
                self.last_unknown_face_data = (face_normalized, frame)
        except Exception as e:
            if self.debug_mode:
                print(f"[FaceRecognition] Failed to send face event: {e}")

    def _setup_api(self):
        @self.flask_app.route('/register_face', methods=['POST'])
        def register_face():
            data = request.get_json()
            face_id = data.get('face_id')
            name = data.get('name')
            print(f"[API] /register_face called with face_id={face_id}, name={name}")
            if not face_id or not name:
                print(f"[API] Missing face_id or name in request: {data}")
                return jsonify({'error': 'face_id and name required'}), 400
            # Find the pending unknown face
            if face_id in self.pending_unknown_faces:
                face_normalized, frame = self.pending_unknown_faces.pop(face_id)
                if self.db.add_face(name, face_normalized):
                    self.known_faces[name] = face_normalized
                    print(f"[API] Registered new face: {name} (face_id={face_id})")
                    return jsonify({'status': 'success', 'name': name})
                else:
                    print(f"[API] Failed to add face to DB for {name}")
                    return jsonify({'error': 'Failed to add face to DB'}), 500
            elif face_id == self.last_unknown_face_id and self.last_unknown_face_data is not None:
                face_normalized, frame = self.last_unknown_face_data
                if self.db.add_face(name, face_normalized):
                    self.known_faces[name] = face_normalized
                    print(f"[API] Registered new face (fallback): {name} (face_id={face_id})")
                    return jsonify({'status': 'success', 'name': name})
                else:
                    print(f"[API] Failed to add face to DB for {name} (fallback)")
                    return jsonify({'error': 'Failed to add face to DB (fallback)'}), 500
            else:
                print(f"[API] No pending face for face_id={face_id}")
                print(f"[API] Current pending_unknown_faces keys: {list(self.pending_unknown_faces.keys())}")
                return jsonify({'error': 'No pending face for this face_id'}), 404

    def _start_api_thread(self):
        def run_flask():
            import logging
            log = logging.getLogger('werkzeug')
            log.setLevel(logging.WARNING)
            self.flask_app.run(host='127.0.0.1', port=5200, debug=False, use_reloader=False)
        t = threading.Thread(target=run_flask, daemon=True)
        t.start()

    def process_frame(self, frame):
        faces, gray = self.detect_faces(frame)
        current_time = time.time()
        recognized_this_frame = False
        unknown_face_present = False
        for (x, y, w, h) in faces:
            # Extract face region
            face_roi = gray[y:y+h, x:x+w]
            if w < 50 or h < 50:
                continue
            face_normalized = cv2.resize(face_roi, (100, 100))
            face_normalized = cv2.equalizeHist(face_normalized)
            if self.debug_mode:
                print(f"Processing face at ({x}, {y}, {w}, {h})")
            # --- Initialize name and confidence to avoid unbound errors ---
            name = None
            confidence = 0
            # Recognize face
            best_match = None
            best_confidence = 0
            is_match = False
            for name_candidate, known_face in self.known_faces.items():
                match, conf = self.compare_faces(face_normalized, known_face, self.recognition_threshold)
                if self.debug_mode:
                    print(f"    {name_candidate}: {conf:.3f} {'✓' if match else '✗'}")
                if conf > best_confidence:
                    best_match = name_candidate
                    best_confidence = conf
            # Only recognize the best match if above threshold
            if best_match is not None and best_confidence > self.recognition_threshold:
                name = best_match
                confidence = best_confidence
                self.unknown_face_streak = 0  # Reset streak if recognized
                recognized_this_frame = True
                self.unknown_face_id = None  # Reset unknown face id
                self.send_face_event(face_id=best_match, name=best_match)
            elif not self.waiting_for_label and (current_time - self.last_unknown_time) > self.unknown_cooldown:
                unknown_face_present = True
                if self.unknown_face_id is None:
                    self.unknown_face_id = f"unknown_{int(current_time)}"
                self.unknown_face_streak += 1
                if self.debug_mode:
                    print(f"  Unknown face streak: {self.unknown_face_streak}")
                if self.unknown_face_streak >= self.unknown_face_streak_threshold:
                    self.send_face_event(face_id=self.unknown_face_id, face_normalized=face_normalized, frame=frame)
                    self.unknown_face_streak = 0  # Reset after sending
            
            # If no match found and not recently prompted
            if name is None and not self.waiting_for_label and (current_time - self.last_unknown_time) > self.unknown_cooldown:
                self.waiting_for_label = True
                self.last_unknown_time = current_time
                
                # Start face labeling in a separate thread
                threading.Thread(
                    target=self.add_new_face_interactive,
                    args=(face_normalized, frame),
                    daemon=True
                ).start()
                
                name = "Unknown"
                confidence = 0
            elif name is None:
                name = "Unknown"
                confidence = 0
            
            # Choose color based on recognition
            if name == "Unknown":
                color = (0, 0, 255)  # Red
                display_text = f"Unknown"
            else:
                color = (0, 255, 0)  # Green
                display_text = f"{name} ({confidence:.2f})"
            
            # Draw rectangle and text
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            cv2.rectangle(frame, (x, y-30), (x+w, y), color, cv2.FILLED)
            cv2.putText(frame, display_text, (x+5, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        if recognized_this_frame:
            self.unknown_face_streak = 0
            self.unknown_face_id = None
        if not unknown_face_present:
            self.unknown_face_id = None
            self.unknown_face_streak = 0
        
        return frame
    
    def add_status_info(self, frame):
        """Add status information to the frame"""
        height, width = frame.shape[:2]
        
        # Add database info
        db_count = len(self.known_faces)
        status_text = f"Database: {db_count} faces | Press 'q' to quit | Press 'c' to capture | Press 'd' for debug"
        
        # Background for text
        cv2.rectangle(frame, (10, 10), (width - 10, 80), (0, 0, 0), cv2.FILLED)
        cv2.putText(frame, status_text, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        if self.waiting_for_label:
            cv2.putText(frame, "Check console for face labeling prompt", (15, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        
        debug_text = f"Debug: {'ON' if self.debug_mode else 'OFF'} | Threshold: 0.6"
        cv2.putText(frame, debug_text, (15, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        return frame
    
    def capture_face_manually(self, frame):
        """Manually capture and label a face"""
        faces, gray = self.detect_faces(frame)
        
        if len(faces) == 0:
            print("No faces detected in the current frame")
            return
        
        # Use the largest face
        largest_face = max(faces, key=lambda f: f[2] * f[3])
        x, y, w, h = largest_face
        
        face_roi = gray[y:y+h, x:x+w]
        face_normalized = cv2.resize(face_roi, (100, 100))
        face_normalized = cv2.equalizeHist(face_normalized)
        
        # Show the captured face
        face_display = cv2.resize(face_normalized, (200, 200))
        cv2.imshow('Captured Face - Close this window to continue', face_display)
        cv2.waitKey(2000)  # Show for 2 seconds
        cv2.destroyWindow('Captured Face - Close this window to continue')
        
        # Get name from user
        name = input("Enter a name for this person: ").strip()
        if name:
            if self.db.add_face(name, face_normalized):
                print(f"Successfully added {name} to the database!")
                self.known_faces[name] = face_normalized
            else:
                print("Error adding face to database")
    
    def run(self):
        """Main application loop"""
        print("Starting Face Recognition Application...")
        print("Controls:")
        print("  - Press 'q' to quit")
        print("  - Press 'c' to manually capture and label a face")
        print("  - Press 'd' to toggle debug mode")
        print("  - When unknown face is detected, check console for labeling prompt")
        print("\nInitializing camera...")
        
        # Initialize webcam
        video_capture = cv2.VideoCapture(0)
        
        if not video_capture.isOpened():
            print("Error: Could not open webcam")
            return
        
        # Set camera properties
        video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        print("Camera initialized successfully!")
        print("Application is now running...\n")
        
        try:
            while True:
                ret, frame = video_capture.read()
                
                if not ret:
                    print("Error: Could not read frame from webcam")
                    break
                
                # Process frame
                frame = self.process_frame(frame)
                
                # Add status info
                frame = self.add_status_info(frame)
                
                # Display frame
                cv2.imshow('Face Recognition App', frame)
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("\nQuitting application...")
                    break
                elif key == ord('c'):
                    self.capture_face_manually(frame)
                elif key == ord('d'):
                    self.debug_mode = not self.debug_mode
                    print(f"Debug mode {'enabled' if self.debug_mode else 'disabled'}")
                elif key == ord('r'):
                    # Refresh database
                    self.known_faces = self.db.get_all_faces()
                    print(f"Refreshed database: {len(self.known_faces)} faces loaded")
        
        except KeyboardInterrupt:
            print("\nApplication interrupted by user")
        
        finally:
            video_capture.release()
            cv2.destroyAllWindows()
            print("Application closed successfully")

def main():
    """Main function"""
    try:
        app = FaceRecognitionApp(debug=True)  # Start with debug mode on
        app.run()
    except Exception as e:
        print(f"Error starting application: {e}")
        print("Make sure you have installed OpenCV: pip install opencv-python")

if __name__ == "__main__":
    main()
