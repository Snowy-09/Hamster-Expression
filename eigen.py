from matrix import (
    identity,
    matrix_multiply,
    matrix_vector_multiply,
    vector_norm,
    normalize_vector,
    dot_product,
    scalar_multiply,
    matrix_subtract
)


MAX_ITERATIONS = 1000
TOLERANCE = 1e-8


def build_small_matrix(centered_images):

    A = centered_images

    A_transpose = []

    for j in range(len(A[0])):

        row = []

        for i in range(len(A)):
            row.append(A[i][j])

        A_transpose.append(row)

    return matrix_multiply(
        A,
        A_transpose
    )


def initial_vector(n):

    vector = [0.0] * n

    for i in range(n):
        vector[i] = 1.0 / (i + 1)

    return normalize_vector(vector)


def power_iteration(A):

    n = len(A)

    vector = initial_vector(n)

    for iteration in range(MAX_ITERATIONS):

        new_vector = matrix_vector_multiply(
            A,
            vector
        )

        new_vector = normalize_vector(
            new_vector
        )

        difference = 0.0

        for i in range(n):

            difference += (
                new_vector[i] - vector[i]
            ) ** 2

        difference = difference ** 0.5

        vector = new_vector

        if difference < TOLERANCE:
            break

    Av = matrix_vector_multiply(
        A,
        vector
    )

    eigenvalue = dot_product(
        vector,
        Av
    )

    return eigenvalue, vector, iteration + 1


def deflate_matrix(
    A,
    eigenvalue,
    eigenvector
):

    n = len(A)

    result = []

    for i in range(n):

        row = []

        for j in range(n):

            value = (
                A[i][j]
                -
                eigenvalue
                * eigenvector[i]
                * eigenvector[j]
            )

            row.append(value)

        result.append(row)

    return result


def find_eigenvectors(A, number):

    current_matrix = A

    eigenvalues = []
    eigenvectors = []

    number = min(
        number,
        len(A)
    )

    for i in range(number):

        print(
            "Finding eigenvector",
            i + 1,
            "of",
            number
        )

        eigenvalue, eigenvector, iterations = (
            power_iteration(current_matrix)
        )

        print(
            "  Eigenvalue:",
            eigenvalue
        )

        print(
            "  Iterations:",
            iterations
        )

        eigenvalues.append(
            eigenvalue
        )

        eigenvectors.append(
            eigenvector
        )

        current_matrix = deflate_matrix(
            current_matrix,
            eigenvalue,
            eigenvector
        )

    return eigenvalues, eigenvectors