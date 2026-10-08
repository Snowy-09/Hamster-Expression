import json

from image_data import load_dataset
from classifier import classify


MODEL_FILE = "model.json"


def load_model(filename=MODEL_FILE):

    with open(filename, "r") as file:
        model = json.load(file)

    return model


def project_image(
    image,
    mean_vector,
    eigenfaces
):

    centered_image = []

    for i in range(len(image)):
        centered_image.append(
            image[i] - mean_vector[i]
        )

    signature = []

    for eigenface in eigenfaces:

        value = 0.0

        for i in range(len(eigenface)):
            value += (
                eigenface[i] *
                centered_image[i]
            )

        signature.append(value)

    return signature


def main():

    print("==============================")
    print("HAMSTER EXPRESSION RECOGNITION")
    print("==============================")
    print()

    print("Loading model...")

    model = load_model()

    mean_vector = model["mean_vector"]
    eigenvalues = model["eigenvalues"]
    eigenfaces = model["eigenfaces"]
    centroids = model["centroids"]

    number_of_eigenfaces = model[
        "number_of_eigenfaces"
    ]

    alpha = model["alpha"]

    classes = model["classes"]

    print(
        "Model loaded successfully."
    )

    print()

    print(
        "Eigenfaces:",
        number_of_eigenfaces
    )

    print(
        "Alpha:",
        alpha
    )

    print(
        "Classes:",
        len(classes)
    )

    print()

    print("Loading training data...")

    images, labels = load_dataset(
        "processed"
    )

    print(
        "Training images:",
        len(images)
    )

    print()

    print("Evaluating saved model...")

    correct = 0

    for i in range(len(images)):

        signature = project_image(
            images[i],
            mean_vector,
            eigenfaces
        )

        predicted, distances = classify(
            signature,
            centroids,
            eigenvalues
        )

        if predicted == labels[i]:
            correct += 1

    total = len(images)

    if total > 0:
        accuracy = (
            correct /
            total
        ) * 100
    else:
        accuracy = 0.0

    print()

    print("==============================")
    print("MODEL RESULTS")
    print("==============================")

    print(
        "Eigenfaces:",
        number_of_eigenfaces
    )

    print(
        "Alpha:",
        alpha
    )

    print(
        "Correct:",
        correct,
        "/",
        total
    )

    print(
        f"Training accuracy: {accuracy:.2f}%"
    )

    print("==============================")

    print()

    print(
        "Run test.py to evaluate "
        "unseen testing images."
    )


if __name__ == "__main__":
    main()