"""
Simple Face Recognition Database Management Tool
================================================

This script provides command-line tools for managing the simple face recognition database.
"""

import argparse
import os
import cv2
import numpy as np
from improved_face_recognition import SimpleFaceDatabase

def list_faces(db):
    """List all faces in the database"""
    faces = db.get_all_faces()
    
    if not faces:
        print("No faces found in the database.")
        return
    
    print(f"\nFound {len(faces)} faces in the database:")
    print("-" * 40)
    for i, name in enumerate(faces.keys(), 1):
        print(f"{i}. {name}")
    print("-" * 40)

def delete_face(db, name):
    """Delete a face from the database"""
    import sqlite3
    try:
        conn = sqlite3.connect(db.db_path)
        cursor = conn.cursor()
        
        cursor.execute('DELETE FROM simple_faces WHERE name = ?', (name,))
        
        if cursor.rowcount > 0:
            conn.commit()
            print(f"Successfully deleted '{name}' from the database.")
        else:
            print(f"No face named '{name}' found in the database.")
        
        conn.close()
    except Exception as e:
        print(f"Error deleting face from database: {e}")

def add_face_from_image(db, image_path, name):
    """Add a face from an image file"""
    if not os.path.exists(image_path):
        print(f"Error: Image file '{image_path}' not found.")
        return
    
    try:
        # Load the image
        image = cv2.imread(image_path)
        if image is None:
            print(f"Error: Could not load image '{image_path}'.")
            return
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        
        if len(faces) == 0:
            print(f"No faces found in the image '{image_path}'.")
            return
        
        if len(faces) > 1:
            print(f"Multiple faces found in the image. Using the largest one.")
        
        # Use the largest face
        largest_face = max(faces, key=lambda f: f[2] * f[3])
        x, y, w, h = largest_face
        
        # Extract and process face
        face_roi = gray[y:y+h, x:x+w]
        face_normalized = cv2.resize(face_roi, (100, 100))
        face_normalized = cv2.equalizeHist(face_normalized)
        
        if db.add_face(name, face_normalized):
            print(f"Successfully added '{name}' to the database from '{image_path}'.")
        else:
            print(f"Failed to add '{name}' to the database.")
    
    except Exception as e:
        print(f"Error processing image: {e}")

def show_face(db, name):
    """Display a face from the database"""
    faces = db.get_all_faces()
    
    if name not in faces:
        print(f"No face named '{name}' found in the database.")
        return
    
    face_data = faces[name]
    
    # Display the face
    face_display = cv2.resize(face_data, (200, 200))
    cv2.imshow(f'Face: {name} - Press any key to close', face_display)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def clear_database(db):
    """Clear all faces from the database"""
    response = input("Are you sure you want to delete ALL faces from the database? (yes/no): ")
    if response.lower() in ['yes', 'y']:
        try:
            import sqlite3
            conn = sqlite3.connect(db.db_path)
            cursor = conn.cursor()
            cursor.execute('DELETE FROM simple_faces')
            conn.commit()
            conn.close()
            print("Database cleared successfully.")
        except Exception as e:
            print(f"Error clearing database: {e}")
    else:
        print("Operation cancelled.")

def main():
    parser = argparse.ArgumentParser(description='Simple Face Recognition Database Management Tool')
    parser.add_argument('action', choices=['list', 'add', 'delete', 'show', 'clear'], 
                       help='Action to perform')
    parser.add_argument('--name', help='Name of the person (for add/delete/show actions)')
    parser.add_argument('--image', help='Path to image file (for add action)')
    parser.add_argument('--db', default='simple_faces.db', help='Database file path')
    
    args = parser.parse_args()
    
    db = SimpleFaceDatabase(args.db)
    
    if args.action == 'list':
        list_faces(db)
    
    elif args.action == 'add':
        if not args.name or not args.image:
            print("Error: --name and --image are required for add action")
            return
        add_face_from_image(db, args.image, args.name)
    
    elif args.action == 'delete':
        if not args.name:
            print("Error: --name is required for delete action")
            return
        delete_face(db, args.name)
    
    elif args.action == 'show':
        if not args.name:
            print("Error: --name is required for show action")
            return
        show_face(db, args.name)
    
    elif args.action == 'clear':
        clear_database(db)

if __name__ == "__main__":
    main()
