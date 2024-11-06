from flask import Flask , render_template , Response
from flask import jsonify
import cv2
import keras
from keras.models import model_from_json
import numpy as np 
import base64
import time


###플라스크 시작
app = Flask(__name__, static_folder='static')

emotion_dict = {0: "Angry", 1: "Fear",  2: "Happy", 3: "Neutral", 4: "Surprised"}
# 모델 json load
json_file = open('ProjectModel\Emotion_detection_with_CNN-main\model\model2_2.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
emotion_model = model_from_json(loaded_model_json)

# 모델 load
emotion_model.load_weights("ProjectModel\Emotion_detection_with_CNN-main\model\model2_2.h5")
print("Loaded model from disk")

cap = cv2.VideoCapture(0)


def gen_frames() : 
        
        while True:
            ret, frame = cap.read()
        # frame = cv2.resize(frame, (1280, 720))
            if not ret:
                break
            face_detector = cv2.CascadeClassifier('ProjectModel\Emotion_detection_with_CNN-main\haarcascades\haarcascade_frontalface_default.xml')
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # detect faces available on camera
            num_faces = face_detector.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)

            # take each face available on the camera and Preprocess it
            for (x, y, w, h) in num_faces:
                cv2.rectangle(frame, (x, y-50), (x+w, y+h+10), (0, 255, 0), 4)
                roi_gray_frame = gray_frame[y:y + h, x:x + w]
                cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray_frame, (48, 48)), -1), 0)

                # predict the emotions
                emotion_prediction = emotion_model.predict(cropped_img)
                maxindex = int(np.argmax(emotion_prediction))
                emotion_label = emotion_dict[maxindex]
                cv2.putText(frame, emotion_label, (x+5, y-20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
                
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n') 


emotion_count = {emotion: 0 for emotion in emotion_dict.values()}  # Count of each emotion
start_time = None  # Start time for the 5-second interval```

most_frequent_emotion = None

def gen_frames_1():
    global start_time ,emotion_count , emotion_label ,most_frequent_emotion  # Declare start_time as global
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        face_detector = cv2.CascadeClassifier('ProjectModel\Emotion_detection_with_CNN-main\haarcascades\haarcascade_frontalface_default.xml')
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        num_faces = face_detector.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)
        
        current_time = time.time()  # Current time
        if start_time is None:
            start_time = current_time

        for (x, y, w, h) in num_faces:
            cv2.rectangle(frame, (x, y-50), (x+w, y+h+10), (0, 255, 0), 4)
            roi_gray_frame = gray_frame[y:y + h, x:x + w]
            cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray_frame, (48, 48)), -1), 0)

            # Predict the emotions
            emotion_prediction = emotion_model.predict(cropped_img)
            maxindex = int(np.argmax(emotion_prediction))
            emotion_label = emotion_dict[maxindex]
            cv2.putText(frame, emotion_label, (x+5, y-20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

            if current_time - start_time <= 5:  # Within the 5-second interval
                emotion_count[emotion_label] += 1  # Increment the count of detected emotion
        
        if current_time - start_time > 5:  # After the 5-second interval
            # Get the most frequent emotion within the 5-second interval
            most_frequent_emotion = max(emotion_count, key=emotion_count.get)
            
            # Reset the count dictionary and start time for the next interval
            emotion_count = {emotion: 0 for emotion in emotion_dict.values()}
            start_time = current_time
        
        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        frame_base64 = base64.b64encode(frame_bytes).decode('utf-8')

        yield frame_base64, emotion_label, most_frequent_emotion     



@app.route('/index')
def index():
    return render_template('index_1.html')

@app.route('/video_feed')
def video_feed():
    return  Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/get_prediction')
def get_prediction():
    frame_base64, emotion_label, most_frequent_emotion = next(gen_frames_1())

    prediction_dict = {
        'frame': frame_base64,
        'emotion': emotion_label,
        'most_frequent_emotion': most_frequent_emotion
    }

    return jsonify(prediction_dict)

####모델!!!!!
@app.route('/Model')
def Model():
    # 
    emotion_dict = {0: "Angry", 1:"Happy" ,  2: "Neutral", 3: "Sad"}

    # load json and create model
    json_file = open('ProjectModel\Emotion_detection_with_CNN-main\model\model1_1.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    emotion_model = model_from_json(loaded_model_json)

    # load weights into new model
    emotion_model.load_weights("ProjectModel\Emotion_detection_with_CNN-main\model\model1_1.h5")
    print("Loaded model from disk")

    # start the webcam feed
    cap = cv2.VideoCapture(0)

    # pass here your video path
    # you may download one from here : https://www.pexels.com/video/three-girls-laughing-5273028/
    # cap = cv2.VideoCapture("C:\\JustDoIt\\ML\\Sample_videos\\emotion_sample6.mp4")

    while True:
        # Find haar cascade to draw bounding box around face
        ret, frame = cap.read()
        # frame = cv2.resize(frame, (1280, 720))
        if not ret:
            break
        face_detector = cv2.CascadeClassifier('ProjectModel\Emotion_detection_with_CNN-main\haarcascades\haarcascade_frontalface_default.xml')
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # detect faces available on camera
        num_faces = face_detector.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)

        # take each face available on the camera and Preprocess it
        for (x, y, w, h) in num_faces:
            cv2.rectangle(frame, (x, y-50), (x+w, y+h+10), (0, 255, 0), 4)
            roi_gray_frame = gray_frame[y:y + h, x:x + w]
            cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray_frame, (48, 48)), -1), 0)

            # predict the emotions
            emotion_prediction = emotion_model.predict(cropped_img)
            maxindex = int(np.argmax(emotion_prediction))
            cv2.putText(frame, emotion_dict[maxindex], (x+5, y-20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

        cv2.imshow('Emotion Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

 

    cap.release()
    cv2.destroyAllWindows()

    return emotion_dict[maxindex]


#감정인식결과 텍스트로 제시할 때 쓸 거!!
@app.route('/detect_emotion')
def detect_emotion():
    # Model 함수 실행
    emotion_result = Model()
    return emotion_result

# 로그인 정보 저장 예시
login_info = {
    'username': '일취월장',
    'password': '1111',
    'image': 'gyj.jpg'  # 로그인 정보에 맞는 이미지 파일명
}

@app.route('/')
def login():
    return render_template('mainloginhtmlcssjs.html')

@app.route('/success')
def success():
    username = login_info['username']
    password = login_info['password']
    image_filename = login_info['image']
    image_path = f'/static/images/{image_filename}'  # 이미지 파일의 경로

    return render_template('success.html', username=username, password=password, image_path=image_path)

@app.route('/workstart')
def workstart():
    username = login_info['username']
    password = login_info['password']

    return render_template('workstart.html',username=username, password=password)


if __name__ == '__main__':
    app.run(debug=True)