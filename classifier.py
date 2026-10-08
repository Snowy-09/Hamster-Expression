import math
ALPHA = 1.0

def calculate_centroids(signatures, labels, classes):
    centroids = {}

    for class_name in classes:
        class_signatures = [
            signatures[i]
            for i in range(len(signatures))
            if labels[i] == class_name
        ]

        if len(class_signatures) == 0:
            continue

        dimension = len(class_signatures[0])
        centroid = [0.0] * dimension

        for signature in class_signatures:
            for j in range(dimension):
                centroid[j] += signature[j]

        for j in range(dimension):
            centroid[j] /= len(class_signatures)

        centroids[class_name] = centroid

    return centroids


def weighted_distance(signature, centroid, eigenvalues):
    total = 0.0

    for i in range(len(signature)):
        difference = signature[i] - centroid[i]
        eigenvalue = eigenvalues[i]

        if eigenvalue < 1e-12:
            eigenvalue = 1e-12

        total += (
            (difference * difference) /
            (eigenvalue ** ALPHA)
        )

    return total ** 0.5


def classify(signature, centroids, eigenvalues):
    distances = {}

    for class_name in centroids:
        distances[class_name] = weighted_distance(
            signature,
            centroids[class_name],
            eigenvalues
        )

    predicted_class = min(
        distances,
        key=distances.get
    )

    return predicted_class, distances


def calculate_confidence(distances):
    values = list(distances.values())

    mean_distance = sum(values) / len(values)

    variance = 0.0

    for distance in values:
        variance += (distance - mean_distance) ** 2

    variance /= len(values)

    standard_deviation = variance ** 0.5

    if standard_deviation < 1e-12:
        standard_deviation = 1e-12

    scores = {}
    total_score = 0.0

    for class_name, distance in distances.items():
        z = (distance - mean_distance) / standard_deviation
        score = math.exp(-z)

        scores[class_name] = score
        total_score += score

    confidence = {
        class_name: scores[class_name] / total_score
        for class_name in scores
    }

    return confidence