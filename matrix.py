def rows(A):
    return len(A)


def cols(A):
    if len(A) == 0:
        return 0

    return len(A[0])


def shape(A):
    return (rows(A), cols(A))

def zeros(r, c):
    result = []
    for i in range(r):
        row = []
        for j in range(c):
            row.append(0.0)
        result.append(row)
    return result


def identity(n):
    result = zeros(n, n)
    for i in range(n):
        result[i][i] = 1.0
    return result

def transpose(A):
    if len(A) == 0:
        return []

    r = rows(A)
    c = cols(A)

    result = zeros(c, r)

    for i in range(r):
        for j in range(c):
            result[j][i] = A[i][j]

    return result

def matrix_add(A, B):
    if shape(A) != shape(B):
        raise ValueError("Matrices must have the same dimensions")

    r = rows(A)
    c = cols(A)

    result = zeros(r, c)

    for i in range(r):
        for j in range(c):
            result[i][j] = A[i][j] + B[i][j]

    return result

def matrix_subtract(A, B):
    """Subtract B from A."""

    if shape(A) != shape(B):
        raise ValueError("Matrices must have the same dimensions")

    r = rows(A)
    c = cols(A)

    result = zeros(r, c)

    for i in range(r):
        for j in range(c):
            result[i][j] = A[i][j] - B[i][j]

    return result

def matrix_multiply(A, B):
    if cols(A) != rows(B):
        raise ValueError(
            "Number of columns in A must equal "
            "number of rows in B"
        )

    rA = rows(A)
    cA = cols(A)
    cB = cols(B)

    result = zeros(rA, cB)
    for i in range(rA):
        for j in range(cB):
            total = 0.0
            for k in range(cA):
                total += A[i][k] * B[k][j]
            result[i][j] = total

    return result

def scalar_multiply(A, scalar):
    r = rows(A)
    c = cols(A)

    result = zeros(r, c)

    for i in range(r):
        for j in range(c):
            result[i][j] = A[i][j] * scalar

    return result

def dot_product(a, b):
    if len(a) != len(b):
        raise ValueError("Vectors must have the same length")
    total = 0.0
    for i in range(len(a)):
        total += a[i] * b[i]

    return total

def vector_add(a, b):
    if len(a) != len(b):
        raise ValueError("Vectors must have the same length")
    result = []
    for i in range(len(a)):
        result.append(a[i] + b[i])
    return result

def vector_subtract(a, b):
    if len(a) != len(b):
        raise ValueError("Vectors must have the same length")
    result = []
    for i in range(len(a)):
        result.append(a[i] - b[i])
    return result

def vector_scalar_multiply(v, scalar):
    result = []
    for value in v:
        result.append(value * scalar)
    return result

def vector_norm(v):
    total = 0.0
    for value in v:
        total += value * value
    return total ** 0.5

def normalize_vector(v):
    magnitude = vector_norm(v)

    if magnitude == 0:
        raise ValueError("Cannot normalize a zero vector")
    result = []
    for value in v:
        result.append(value / magnitude)
    return result

def matrix_vector_multiply(A, v):

    if cols(A) != len(v):
        raise ValueError("Matrix columns must equal vector length")
    result = []
    for i in range(rows(A)):
        total = 0.0
        for j in range(cols(A)):
            total += A[i][j] * v[j]
        result.append(total)
    return result

def vector_to_column(v):
    result = []
    for value in v:
        result.append([value])

    return result

def column_to_vector(A):
    if cols(A) != 1:
        raise ValueError("Matrix must have exactly one column")
    result = []
    for row in A:
        result.append(row[0])
    return result

def copy_matrix(A):
    result = []
    for row in A:
        result.append(row[:])
    return result

if __name__ == "__main__":
    print("===== MATRIX TESTS =====")

    A = [
        [1, 2],
        [3, 4]
    ]

    B = [
        [5, 6],
        [7, 8]
    ]

    print("\nA:")
    print(A)

    print("\nB:")
    print(B)

    print("\nShape of A:")
    print(shape(A))

    print("\nTranspose of A:")
    print(transpose(A))

    print("\nA + B:")
    print(matrix_add(A, B))

    print("\nA - B:")
    print(matrix_subtract(A, B))

    print("\nA × B:")
    print(matrix_multiply(A, B))

    print("\nA × 2:")
    print(scalar_multiply(A, 2))

    v = [3, 4]

    print("\nVector:")
    print(v)

    print("\nDot product:")
    print(dot_product(v, v))

    print("\nVector norm:")
    print(vector_norm(v))

    print("\nNormalized vector:")
    print(normalize_vector(v))

    print("\nA × vector:")
    print(matrix_vector_multiply(A, v))

    print("\nIdentity matrix:")
    print(identity(3))

    print("\n===== TESTS COMPLETE =====")