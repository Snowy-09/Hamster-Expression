import json

from image_data import load_dataset
from classifier import classify, calculate_confidence

MODEL_FILE = "model.json"

def load_model(filename=MODEL_FILE):
    with open(filename, "r") as file:
        return json.load(file)

def project_image(image, mean_vector, eigenfaces):
    centered_image = [
        image[j] - mean_vector[j]
        for j in range(len(image))
    ]

    signature = []

    for eigenface in eigenfaces:
        value = 0.0

        for j in range(len(eigenface)):
            value += eigenface[j] * centered_image[j]

        signature.append(value)

    return signature


def main():
    print("===================================")
    print("       HAMSTER EXPRESSION TEST")
    print("===================================\n")

    print("Loading trained model...")
    model = load_model()

    mean_vector = model["mean_vector"]
    eigenvalues = model["eigenvalues"]
    eigenfaces = model["eigenfaces"]
    centroids = model["centroids"]

    number_of_eigenfaces = model["number_of_eigenfaces"]
    alpha = model["alpha"]
    classes = model["classes"]

    print("Model loaded successfully.\n")
    print("Image size:", model["image_size"])
    print("Eigenfaces:", number_of_eigenfaces)
    print("Alpha:", alpha)
    print("Classes:", classes)
    print()

    print("Loading testing data...")

    testing_images, testing_labels = load_dataset(
        "processed_testing"
    )

    print("Testing images:", len(testing_images))
    print()

    if len(testing_images) == 0:
        print("No testing images found.")
        return

    correct = 0
    total = len(testing_images)

    for i in range(total):
        image = testing_images[i]
        actual = testing_labels[i]

        signature = project_image(
            image,
            mean_vector,
            eigenfaces
        )

        predicted, distances = classify(
            signature,
            centroids,
            eigenvalues
        )

        confidence = calculate_confidence(distances)
        confidence_percent = confidence[predicted] * 100

        if predicted == actual:
            correct += 1

        print(
            f"{i + 1:3d}. "
            f"Actual: {actual:10s} | "
            f"Predicted: {predicted:10s} | "
            f"Confidence: {confidence_percent:6.2f}%"
        )

    print()
    print("==============================")
    print("TEST RESULTS")
    print("==============================")
    print("Correct:", correct, "/", total)

    if total > 0:
        accuracy = (correct / total) * 100
        print(f"Accuracy: {accuracy:.2f}%")

    print("==============================")


if __name__ == "__main__":
    main()