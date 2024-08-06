import os
import cv2
import numpy as np
import insightface
from sklearn import preprocessing
from PIL import Image, ImageDraw, ImageFont
import time
import threading
import queue

class FaceRecognition:
    def __init__(self, gpu_id=0, face_db='face_db', threshold=1.24, det_thresh=0.50, det_size=(640, 640)):
        self.gpu_id = gpu_id
        self.face_db = face_db
        self.threshold = threshold
        self.det_thresh = det_thresh
        self.det_size = det_size

        self.model = insightface.app.FaceAnalysis(providers=['CUDAExecutionProvider'])
        self.model.prepare(ctx_id=self.gpu_id, det_thresh=self.det_thresh, det_size=self.det_size)
        self.faces_embedding = list()
        self.load_faces(self.face_db)

    def load_faces(self, face_db_path):
        if not os.path.exists(face_db_path):
            os.makedirs(face_db_path)
        for root, dirs, files in os.walk(face_db_path):
            for file in files:
                try:
                    input_image = cv2.imdecode(np.fromfile(os.path.join(root, file), dtype=np.uint8), 1)
                    user_name = file.split(".")[0]
                    face = self.model.get(input_image)[0]
                    embedding = np.array(face.embedding).reshape((1, -1))
                    embedding = preprocessing.normalize(embedding)
                    self.faces_embedding.append({
                        "user_name": user_name,
                        "feature": embedding
                    })
                except Exception as e:
                    print(f"Error loading {file}: {e}")

    def detect(self, image):
        try:
            faces = self.model.get(image)
            results = []
            for face in faces:
                embedding = np.array(face.embedding).reshape((1, -1))
                embedding = preprocessing.normalize(embedding)
                user_name = "unknown"
                for com_face in self.faces_embedding:
                    if self.feature_compare(embedding, com_face["feature"], self.threshold):
                        user_name = com_face["user_name"]
                        break
                results.append(user_name)
            return results
        except Exception as e:
            print(f"Detection error: {e}")
            return []

    @staticmethod
    def feature_compare(feature1, feature2, threshold):
        diff = np.subtract(feature1, feature2)
        dist = np.sum(np.square(diff), 1)
        return dist < threshold

    def register(self, image, user_name):
        try:
            faces = self.model.get(image)
            if len(faces) != 1:
                return '图片检测不到人脸'
            embedding = np.array(faces[0].embedding).reshape((1, -1))
            embedding = preprocessing.normalize(embedding)
            for com_face in self.faces_embedding:
                if self.feature_compare(embedding, com_face["feature"], self.threshold):
                    return '该用户已存在'
            cv2.imencode('.png', image)[1].tofile(os.path.join(self.face_db, f'{user_name}.png'))
            self.faces_embedding.append({
                "user_name": user_name,
                "feature": embedding
            })
            return "success"
        except Exception as e:
            print(f"Registration error: {e}")
            return "failure"

def load_font(font_path, font_size):
    try:
        return ImageFont.truetype(font_path, font_size)
    except IOError:
        print(f"Font file not found at {font_path}, using default font.")
        return ImageFont.load_default()
def draw_chinese_text(image, text, position, font,  color=(0, 255, 0)):
    image_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(image_pil)

    draw.text(position, text, font=font, fill=color)
    image = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
    return image

def capture_frames(cap, frame_queue):
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1
        # 每隔一帧处理一次
        if frame_count % 3 == 0:
            if frame_queue.full():
                frame_queue.get()
            frame_queue.put(frame)

def process_frames(face_recognition, frame_queue, processed_queue,font):
    while True:
        if not frame_queue.empty():
            frame = frame_queue.get()
            results = face_recognition.detect(frame)
            for result in results:
                pass

            faces = face_recognition.model.get(frame)
            for face in faces:
                bbox = face.bbox.astype(np.int32)
                cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
                frame = draw_chinese_text(frame, result, (bbox[0], bbox[1] - 30), font=font)

            if processed_queue.full():
                processed_queue.get()
            processed_queue.put(frame)

if __name__ == '__main__':
    face_recognition = FaceRecognition()
    cap = cv2.VideoCapture(0)
    cv2.namedWindow('camera')

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    # 获取摄像头帧率
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"摄像头帧率: {fps} FPS")

    font_path = 'path/to/font/msyh.ttc'
    font_size = 20
    font = load_font(font_path, font_size)


    frame_queue = queue.Queue(maxsize=5)
    processed_queue = queue.Queue(maxsize=5)

    capture_thread = threading.Thread(target=capture_frames, args=(cap, frame_queue))
    process_thread = threading.Thread(target=process_frames, args=(face_recognition, frame_queue, processed_queue, font))

    capture_thread.start()
    process_thread.start()

    frame_count = 0
    start_time = time.time()

    while cap.isOpened():
        if not processed_queue.empty():
            frame = processed_queue.get()
            cv2.imshow('Camera', frame)

            # 计算处理后的帧数
            frame_count += 1
            if frame_count >= 10:  # 每处理10帧计算一次FPS
                end_time = time.time()
                elapsed_time = end_time - start_time
                output_fps = frame_count / elapsed_time
                print(f"处理后输出帧率: {output_fps:.2f} FPS")
                frame_count = 0
                start_time = time.time()

    capture_thread.join()
    process_thread.join()
    cap.release()
    cv2.destroyAllWindows()
