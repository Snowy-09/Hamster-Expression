from image_data import load_dataset
from pca import calculate_mean_vector, center_dataset, save_vector_as_pgm
from eigen import build_small_matrix, find_eigenvectors
from matrix import (
    transpose,
    matrix_vector_multiply,
    vector_norm,
    normalize_vector,
    dot_product
)

NUM_EIGENFACES = 10
def generate_eigenfaces(centered_images, eigenvectors):
    A_transpose = transpose(centered_images)

    eigenfaces = []

    for i, eigenvector in enumerate(eigenvectors):

        print(
            "Generating eigenface",
            i + 1,
            "of",
            len(eigenvectors)
        )

        # A^T * v
        eigenface = matrix_vector_multiply(
            A_transpose,
            eigenvector
        )

        # Normalize eigenface
        norm = vector_norm(eigenface)

        if norm == 0:
            raise ValueError(
                "Generated eigenface has zero magnitude"
            )

        eigenface = normalize_vector(eigenface)

        eigenfaces.append(eigenface)

    return eigenfaces

if __name__ == "__main__":

    print("Loading dataset...")

    images, labels = load_dataset()

    print("Calculating mean...")

    mean = calculate_mean_vector(images)

    print("Centering dataset...")

    centered_images = center_dataset(
        images,
        mean
    )

    print()
    print("Building small PCA matrix...")

    L = build_small_matrix(centered_images)

    print(
        "Small matrix size:",
        len(L),
        "x",
        len(L[0])
    )

    print()
    print(
        "Calculating",
        NUM_EIGENFACES,
        "eigenvectors..."
    )

    eigenvalues, eigenvectors = find_eigenvectors(
        L,
        NUM_EIGENFACES
    )

    print()
    print("Eigenvectors calculated.")
    print()
    print("Generating eigenfaces...")

    eigenfaces = generate_eigenfaces(
        centered_images,
        eigenvectors
    )

    print()
    print("==============================") 
    print("EIGENFACE ORTHOGONALITY TEST")
    print("==============================")

    for i in range(len(eigenfaces)):
        for j in range(i + 1, len(eigenfaces)):

            value = dot_product(
                eigenfaces[i],
                eigenfaces[j]
        )

            print(
                "Eigenface",
                i + 1,
                "vs",
                j + 1,
                ":",
                value
            )


    print()
    print("Eigenfaces generated.")
    print()
    print("==============================")
    print("EIGENFACE INFORMATION")
    print("==============================")

    print(
        "Number of eigenfaces:",
        len(eigenfaces)
    )

    print(
        "Values per eigenface:",
        len(eigenfaces[0])
    )

    print()
    print("Checking eigenface normalization...")

    for i in range(len(eigenfaces)):

        norm = vector_norm(eigenfaces[i])

        print(
            "Eigenface",
            i + 1,
            "norm:",
            norm
        )
    print()
    print("Saving eigenfaces...")

    for i, eigenface in enumerate(eigenfaces):

        filename = f"eigenface_{i + 1}.pgm"

        save_vector_as_pgm(
            eigenface,
            filename
        )

        print(
            "Saved:",
            filename
        )
    print()
    print("==============================")
    print("EIGENFACE GENERATION COMPLETE")
    print("==============================")