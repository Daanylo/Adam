"""
Test script to verify the face recognition setup
"""

import cv2
import numpy as np
import sys

def test_camera():
    """Test if camera is accessible"""
    print("Testing camera access...")
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("❌ Camera not accessible")
        return False
    
    ret, frame = cap.read()
    if not ret:
        print("❌ Cannot read from camera")
        cap.release()
        return False
    
    print("✅ Camera working")
    cap.release()
    return True

def test_face_detection():
    """Test face detection capabilities"""
    print("Testing face detection...")
    
    try:
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        if face_cascade.empty():
            print("❌ Face detection classifier not loaded")
            return False
        
        print("✅ Face detection ready")
        return True
    except Exception as e:
        print(f"❌ Face detection error: {e}")
        return False

def test_database():
    """Test database functionality"""
    print("Testing database...")
    
    try:
        import sqlite3
        
        # Test database creation
        conn = sqlite3.connect(':memory:')
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE test (
                id INTEGER PRIMARY KEY,
                data BLOB
            )
        ''')
        
        # Test data insertion
        test_data = np.array([1, 2, 3])
        import pickle
        cursor.execute('INSERT INTO test (data) VALUES (?)', (pickle.dumps(test_data),))
        
        # Test data retrieval
        cursor.execute('SELECT data FROM test')
        retrieved = pickle.loads(cursor.fetchone()[0])
        
        if np.array_equal(test_data, retrieved):
            print("✅ Database functionality working")
            conn.close()
            return True
        else:
            print("❌ Database data integrity issue")
            conn.close()
            return False
            
    except Exception as e:
        print(f"❌ Database error: {e}")
        return False

def test_opencv_version():
    """Test OpenCV version and features"""
    print(f"OpenCV version: {cv2.__version__}")
    
    required_features = [
        'CascadeClassifier',
        'VideoCapture',
        'TM_CCOEFF_NORMED'
    ]
    
    for feature in required_features:
        if hasattr(cv2, feature):
            print(f"✅ {feature} available")
        else:
            print(f"❌ {feature} not available")
            return False
    
    return True

def main():
    """Run all tests"""
    print("🔍 Face Recognition Setup Test")
    print("=" * 40)
    
    tests = [
        ("OpenCV Version & Features", test_opencv_version),
        ("Camera Access", test_camera),
        ("Face Detection", test_face_detection),
        ("Database Functionality", test_database),
    ]
    
    all_passed = True
    
    for test_name, test_func in tests:
        print(f"\n📋 {test_name}")
        print("-" * 30)
        
        try:
            if not test_func():
                all_passed = False
        except Exception as e:
            print(f"❌ Test failed with exception: {e}")
            all_passed = False
    
    print("\n" + "=" * 40)
    if all_passed:
        print("🎉 All tests passed! Your setup is ready.")
        print("\nYou can now run:")
        print("  python basic_face_recognition.py")
    else:
        print("❌ Some tests failed. Please check the errors above.")
        print("\nTry installing missing dependencies:")
        print("  pip install opencv-python opencv-contrib-python numpy")
    
    print("=" * 40)

if __name__ == "__main__":
    main()
