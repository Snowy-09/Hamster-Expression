import cv2
import os


IMAGE_SIZE = 64
VECTOR_SIZE = IMAGE_SIZE * IMAGE_SIZE

CLASSES = [
    "Angry",
    "Cry",
    "Happy",
    "Neutral",
    "Sad",
    "Surprised"
]


def load_image(path):

    image = cv2.imread(
        path,
        cv2.IMREAD_GRAYSCALE
    )

    if image is None:
        raise ValueError(
            "Could not read image: " + path
        )

    if image.shape != (IMAGE_SIZE, IMAGE_SIZE):
        raise ValueError(
            "Incorrect image size: " + path
        )

    vector = []

    for i in range(IMAGE_SIZE):

        for j in range(IMAGE_SIZE):

            pixel = image[i, j]

            vector.append(
                float(pixel) / 255.0
            )

    return vector


def load_dataset(directory="processed"):

    images = []
    labels = []

    for class_name in CLASSES:

        class_dir = os.path.join(
            directory,
            class_name
        )

        if not os.path.exists(class_dir):
            continue

        for filename in os.listdir(class_dir):

            path = os.path.join(
                class_dir,
                filename
            )

            if not os.path.isfile(path):
                continue

            try:

                image = load_image(path)

                images.append(image)
                labels.append(class_name)

            except Exception as e:

                print(
                    "Skipping:",
                    path,
                    "|",
                    e
                )

    return images, labels


def print_dataset_info(images, labels):

    print(
        "Number of images:",
        len(images)
    )

    print(
        "Features per image:",
        len(images[0])
    )

    print()

    for class_name in CLASSES:

        count = labels.count(class_name)

        print(
            f"{class_name:10s}: {count}"
        )