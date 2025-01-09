from flask import Flask, render_template, Response
import cv2
import dlib
import threading
from deepface import DeepFace

app = Flask(__name__)

hog_face_detector = dlib.get_frontal_face_detector()
dlib_facelandmark = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
img1 = cv2.imread(r"C:\Users\Varrun\Desktop\Projects\UserVerification\reference.jpg")

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

counter = 0
face_match = False

def check_face(frame):
    global face_match 
    try: 
        if DeepFace.verify(frame, img1.copy())["verified"]: 
            face_match = True
        else: 
            face_match = False
    except ValueError: 
        face_match = False 

def gen_frames():  
    global counter
    global face_match
    while True:
        _, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = hog_face_detector(gray)
        for face in faces: 
            face_landmarks = dlib_facelandmark(gray, face)
            for n in range(0, 68):
                x = face_landmarks.part(n).x
                y = face_landmarks.part(n).y
                cv2.circle(frame, (x, y), 1, (0, 255, 0), 1)

            if counter % 30 == 0:  
                try:
                    thread = threading.Thread(target=check_face, args=(frame.copy(),)).start()
                except ValueError:
                    pass
            counter += 1

            if face_match:  
                cv2.putText(frame, "USER VERIFIED", (20, 450), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
                break
            else:
                cv2.putText(frame, "USER UNVERIFIED", (20, 450), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', threaded=True)
