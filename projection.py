from image_data import load_dataset
from pca import calculate_mean_vector, center_dataset
from eigen import build_small_matrix, find_eigenvectors
from eigenfaces import generate_eigenfaces
from matrix import dot_product
NUM_EIGENFACES = 10

def project_image(centered_image, eigenfaces):
    signature = []

    for eigenface in eigenfaces:

        weight = dot_product(
            eigenface,
            centered_image
        )

        signature.append(weight)

    return signature

def project_dataset(centered_images, eigenfaces):
    signatures = []
    for i, image in enumerate(centered_images):

        signature = project_image(
            image,
            eigenfaces
        )

        signatures.append(signature)

        print(
            "Projected image",
            i + 1,
            "of",
            len(centered_images)
        )
    return signatures

if __name__ == "__main__":

    print("Loading dataset...")

    images, labels = load_dataset()

    print()
    print("Calculating mean...")

    mean = calculate_mean_vector(images)

    print()
    print("Centering dataset...")

    centered_images = center_dataset(
        images,
        mean
    )
    print()
    print("Building small PCA matrix...")

    L = build_small_matrix(
        centered_images
    )

    print(
        "Small matrix:",
        len(L),
        "x",
        len(L[0])
    )

    print()
    print(
        "Finding",
        NUM_EIGENFACES,
        "eigenvectors..."
    )

    eigenvalues, eigenvectors = find_eigenvectors(
        L,
        NUM_EIGENFACES
    )

    print()
    print("Generating eigenfaces...")

    eigenfaces = generate_eigenfaces(
        centered_images,
        eigenvectors
    )

    print("Eigenfaces generated.")
    print()
    print("==============================")
    print("PROJECTING DATASET")
    print("==============================")

    signatures = project_dataset(
        centered_images,
        eigenfaces
    )

    print()
    print("==============================")
    print("PROJECTION RESULTS")
    print("==============================")

    print(
        "Number of signatures:",
        len(signatures)
    )

    print(
        "Values per signature:",
        len(signatures[0])
    )

    print()

    # Show first few signatures

    for i in range(min(5, len(signatures))):

        print(
            "Image",
            i + 1,
            "| Label:",
            labels[i]
        )

        print(
            "Signature:",
            signatures[i]
        )

        print()

    print("==============================")
    print("PROJECTION COMPLETE")
    print("==============================")