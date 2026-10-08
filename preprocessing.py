import cv2
import os

IMAGE_SIZE = 64
FACE_PADDING = 0.20

DATASET_DIR = "dataset"
PROCESSED_DIR = "processed"

TESTING_DIR = "testing_data"
PROCESSED_TESTING_DIR = "processed_testing"

CLASSES = [
    "Angry",
    "Cry",
    "Happy",
    "Neutral",
    "Sad",
    "Surprised"
]


def get_face_detector():
    cascade_path = (
        cv2.data.haarcascades +
        "haarcascade_frontalface_default.xml"
    )

    detector = cv2.CascadeClassifier(cascade_path)

    if detector.empty():
        raise RuntimeError("Could not load Haar cascade.")

    return detector


def preprocess_image(image_path, detector):
    image = cv2.imread(image_path)

    if image is None:
        return None

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    faces = detector.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )

    if len(faces) == 0:
        return None

    # Select largest detected face
    x, y, w, h = max(
        faces,
        key=lambda face: face[2] * face[3]
    )

    padding_x = int(w * FACE_PADDING)
    padding_y = int(h * FACE_PADDING)

    x1 = max(0, x - padding_x)
    y1 = max(0, y - padding_y)

    x2 = min(gray.shape[1], x + w + padding_x)
    y2 = min(gray.shape[0], y + h + padding_y)

    face = gray[y1:y2, x1:x2]

    face = cv2.resize(
        face,
        (IMAGE_SIZE, IMAGE_SIZE),
        interpolation=cv2.INTER_AREA
    )

    return face


def process_folder(input_dir, output_dir):
    detector = get_face_detector()

    os.makedirs(output_dir, exist_ok=True)

    total = 0
    successful = 0
    failed = 0

    for class_name in CLASSES:

        input_class_dir = os.path.join(
            input_dir,
            class_name
        )

        output_class_dir = os.path.join(
            output_dir,
            class_name
        )

        os.makedirs(output_class_dir, exist_ok=True)

        if not os.path.exists(input_class_dir):
            print("Missing folder:", input_class_dir)
            continue

        for filename in os.listdir(input_class_dir):

            input_path = os.path.join(
                input_class_dir,
                filename
            )

            if not os.path.isfile(input_path):
                continue

            total += 1

            face = preprocess_image(
                input_path,
                detector
            )

            if face is None:
                print("FAILED:", input_path)
                failed += 1
                continue

            output_path = os.path.join(
                output_class_dir,
                filename
            )

            cv2.imwrite(output_path, face)

            successful += 1

    print()
    print("Preprocessing complete")
    print("----------------------")
    print("Total images :", total)
    print("Successful   :", successful)
    print("Failed       :", failed)


def main():

    print("=== TRAINING DATA ===")
    process_folder(
        DATASET_DIR,
        PROCESSED_DIR
    )

    print()
    print("=== TESTING DATA ===")
    process_folder(
        TESTING_DIR,
        PROCESSED_TESTING_DIR
    )


if __name__ == "__main__":
    main()