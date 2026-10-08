import cv2
import time


IMAGE_SIZE = 64
FACE_PADDING = 0.20

MIN_FACE_SIZE = 100
MOVEMENT_THRESHOLD = 20
SIZE_CHANGE_THRESHOLD = 0.25
REACQUIRE_DELAY = 1.0

locked_box = None
lost_since = None


def get_face_detector():
    cascade_path = (
        cv2.data.haarcascades +
        "haarcascade_frontalface_default.xml"
    )

    detector = cv2.CascadeClassifier(cascade_path)

    if detector.empty():
        raise RuntimeError("Could not load Haar cascade.")

    return detector


def reset_tracker():
    global locked_box, lost_since

    locked_box = None
    lost_since = None


def is_valid_detection(face, frame_width, frame_height):
    x, y, w, h = face

    if w < MIN_FACE_SIZE or h < MIN_FACE_SIZE:
        return False

    aspect_ratio = w / float(h)

    if aspect_ratio < 0.65 or aspect_ratio > 1.5:
        return False

    if x <= 2 or y <= 2:
        return False

    if x + w >= frame_width - 2:
        return False

    if y + h >= frame_height - 2:
        return False

    return True


def get_center(box):
    x, y, w, h = box
    return x + w / 2, y + h / 2


def center_distance(box1, box2):
    x1, y1 = get_center(box1)
    x2, y2 = get_center(box2)

    dx = x2 - x1
    dy = y2 - y1

    return (dx * dx + dy * dy) ** 0.5


def size_difference(old_box, new_box):
    old_w = old_box[2]
    old_h = old_box[3]

    new_w = new_box[2]
    new_h = new_box[3]

    width_difference = abs(new_w - old_w) / float(old_w)
    height_difference = abs(new_h - old_h) / float(old_h)

    return max(width_difference, height_difference)


def detect_faces(gray, detector):
    faces = detector.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=8,
        minSize=(MIN_FACE_SIZE, MIN_FACE_SIZE)
    )

    frame_height, frame_width = gray.shape
    valid_faces = []

    for face in faces:
        if is_valid_detection(
            face,
            frame_width,
            frame_height
        ):
            valid_faces.append(face)

    return valid_faces


def choose_initial_face(faces):
    if len(faces) == 0:
        return None

    return max(
        faces,
        key=lambda face: face[2] * face[3]
    )


def choose_tracked_face(faces, previous_box):
    if len(faces) == 0:
        return None

    best_face = None
    best_distance = None

    previous_w = previous_box[2]
    previous_h = previous_box[3]

    maximum_distance = max(
        previous_w,
        previous_h
    ) * 0.75

    for face in faces:
        distance = center_distance(
            previous_box,
            face
        )

        if distance > maximum_distance:
            continue

        difference = size_difference(
            previous_box,
            face
        )

        if difference > SIZE_CHANGE_THRESHOLD:
            continue

        if best_distance is None or distance < best_distance:
            best_face = face
            best_distance = distance

    return best_face


def preprocess_frame(frame, detector):
    global locked_box, lost_since

    gray = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY
    )

    frame_height, frame_width = gray.shape

    faces = detect_faces(
        gray,
        detector
    )

    if locked_box is None:
        new_face = choose_initial_face(faces)

        if new_face is None:
            return None, None

        locked_box = new_face
        lost_since = None

    else:
        tracked_face = choose_tracked_face(
            faces,
            locked_box
        )

        if tracked_face is not None:
            lost_since = None

            movement = center_distance(
                locked_box,
                tracked_face
            )

            if movement >= MOVEMENT_THRESHOLD:
                old_w = locked_box[2]
                old_h = locked_box[3]

                center_x, center_y = get_center(
                    tracked_face
                )

                new_x = int(center_x - old_w / 2)
                new_y = int(center_y - old_h / 2)

                new_x = max(
                    0,
                    min(
                        new_x,
                        frame_width - old_w
                    )
                )

                new_y = max(
                    0,
                    min(
                        new_y,
                        frame_height - old_h
                    )
                )

                locked_box = (
                    new_x,
                    new_y,
                    old_w,
                    old_h
                )

        else:
            if lost_since is None:
                lost_since = time.time()

            time_lost = time.time() - lost_since

            if time_lost >= REACQUIRE_DELAY:
                new_face = choose_initial_face(faces)

                if new_face is not None:
                    locked_box = new_face
                    lost_since = None
                else:
                    return None, None

    x, y, w, h = locked_box

    x = max(
        0,
        min(x, frame_width - w)
    )

    y = max(
        0,
        min(y, frame_height - h)
    )

    locked_box = (x, y, w, h)

    padding_x = int(w * FACE_PADDING)
    padding_y = int(h * FACE_PADDING)

    x1 = max(0, x - padding_x)
    y1 = max(0, y - padding_y)
    x2 = min(frame_width, x + w + padding_x)
    y2 = min(frame_height, y + h + padding_y)

    face = gray[y1:y2, x1:x2]

    face = cv2.resize(
        face,
        (IMAGE_SIZE, IMAGE_SIZE),
        interpolation=cv2.INTER_AREA
    )

    vector = []

    for i in range(IMAGE_SIZE):
        for j in range(IMAGE_SIZE):
            vector.append(
                float(face[i, j]) / 255.0
            )

    return vector, locked_box


def start_camera():
    detector = get_face_detector()
    camera = cv2.VideoCapture(0)

    if not camera.isOpened():
        raise RuntimeError("Could not open webcam.")

    reset_tracker()

    print("Webcam started.")
    print()
    print("Face tracking:")
    print("- Initial face = largest detected face")
    print("- Small movement = ignored")
    print("- Large movement = box follows")
    print("- Lost face = wait 1 second")
    print("- Then reacquire largest face")
    print()
    print("Press Q to quit.")

    while True:
        success, frame = camera.read()

        if not success:
            break

        frame = cv2.flip(frame, 1)

        vector, face_box = preprocess_frame(
            frame,
            detector
        )

        if face_box is not None:
            x, y, w, h = face_box

            cv2.rectangle(
                frame,
                (x, y),
                (x + w, y + h),
                (0, 255, 0),
                2
            )

            cv2.putText(
                frame,
                "LOCKED",
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )

        else:
            cv2.putText(
                frame,
                "Looking for face...",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2
            )

        cv2.imshow("Webcam Test", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

    camera.release()
    cv2.destroyAllWindows()
    reset_tracker()


if __name__ == "__main__":
    start_camera()