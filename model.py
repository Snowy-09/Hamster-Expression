import json

from image_data import load_dataset, CLASSES
from pca import calculate_mean_vector, center_dataset
from eigen import build_small_matrix, find_eigenvectors
from eigenfaces import generate_eigenfaces
from projection import project_dataset
from classifier import calculate_centroids, ALPHA


NUMBER_OF_EIGENFACES = 50
MODEL_FILE = "model.json"


def save_model(
    mean_vector,
    eigenvalues,
    eigenfaces,
    centroids,
    filename=MODEL_FILE
):
    model = {
        "image_size": 64,
        "number_of_eigenfaces": NUMBER_OF_EIGENFACES,
        "alpha": ALPHA,
        "classes": CLASSES,
        "mean_vector": mean_vector,
        "eigenvalues": eigenvalues,
        "eigenfaces": eigenfaces,
        "centroids": centroids
    }

    with open(filename, "w") as file:
        json.dump(model, file)

    print("\nModel saved to:", filename)


def train_model():
    print("===================================")
    print("     HAMSTER EXPRESSION MODEL")
    print("===================================\n")

    print("Loading training dataset...")
    images, labels = load_dataset("processed")

    print("Training images:", len(images))

    if len(images) == 0:
        raise RuntimeError("No training images found.")

    print("\nCalculating mean vector...")
    mean_vector = calculate_mean_vector(images)

    print("Centering dataset...")
    centered_images = center_dataset(images, mean_vector)

    print("Building covariance matrix...")
    small_matrix = build_small_matrix(centered_images)

    print(
        "Small covariance matrix:",
        len(small_matrix),
        "x",
        len(small_matrix[0])
    )

    print(
        "\nFinding",
        NUMBER_OF_EIGENFACES,
        "eigenvectors..."
    )

    eigenvalues, eigenvectors = find_eigenvectors(
        small_matrix,
        NUMBER_OF_EIGENFACES
    )

    print("\nGenerating eigenfaces...")
    eigenfaces = generate_eigenfaces(
        centered_images,
        eigenvectors
    )

    print(
        "Generated",
        len(eigenfaces),
        "eigenfaces."
    )

    print("\nProjecting training images...")
    signatures = project_dataset(
        centered_images,
        eigenfaces
    )

    print("Signature size:", len(signatures[0]))

    print("\nCalculating class centroids...")
    centroids = calculate_centroids(
        signatures,
        labels,
        CLASSES
    )

    print("Centroids created:", len(centroids))

    save_model(
        mean_vector,
        eigenvalues,
        eigenfaces,
        centroids
    )

    print("\n===================================")
    print("        MODEL TRAINING DONE")
    print("===================================")


def load_model(filename=MODEL_FILE):
    with open(filename, "r") as file:
        return json.load(file)


if __name__ == "__main__":
    train_model()