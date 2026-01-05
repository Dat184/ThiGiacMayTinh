import cv2
import numpy as np
from mtcnn import MTCNN
from keras_facenet import FaceNet
from sklearn.metrics.pairwise import cosine_similarity

detector = MTCNN()
embedder = FaceNet()
THRESHOLD = 0.7

# ham lay embedding tu khuon mat
def get_embedding(face_img):
    face_img = cv2.resize(face_img, (160, 160))
    face_img = face_img.astype('float32')
    face_img = np.expand_dims(face_img, axis=0)
    embedding = embedder.embeddings(face_img)
    return embedding[0]
# load anh va lay embedding
known_img = cv2.imread("known.jpeg")
known_img = cv2.cvtColor(known_img, cv2.COLOR_BGR2RGB)

faces = detector.detect_faces(known_img)
x, y, w, h = faces[0]['box']
known_face = known_img[y:y+h, x:x+w]

known_embedding = get_embedding(known_face)

cap = cv2.VideoCapture(0)

THRESHOLD = 0.7

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.resize(frame, (640, 480))
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    try:
        faces = detector.detect_faces(rgb_frame)
    except:
        faces = []

    for face in faces:
        x, y, w, h = face['box']
        face_img = rgb_frame[y:y+h, x:x+w]

        embedding = get_embedding(face_img)

        similarity = cosine_similarity(
            [known_embedding],
            [embedding]
        )[0][0]

        if similarity > THRESHOLD:
            label = f"Matched ({similarity:.2f})"
            color = (0, 255, 0)
        else:
            label = f"Unknown ({similarity:.2f})"
            color = (0, 0, 255)

        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        cv2.putText(frame, label, (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    cv2.imshow("Face Recognition - FaceNet", frame)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()