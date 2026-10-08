from image_data import load_dataset
from matrix import vector_subtract, rows, cols
IMAGE_SIZE = 128

def calculate_mean_vector(images):
    if len(images) == 0:
        raise ValueError("Dataset is empty")

    num_images = len(images)
    num_features = len(images[0])

    mean = [0.0] * num_features

    for image in images:
        for j in range(num_features):
            mean[j] += image[j]

    for j in range(num_features):
        mean[j] /= num_images

    return mean

def center_dataset(images, mean):
    centered_images = []

    for image in images:
        centered_image = vector_subtract(image, mean)
        centered_images.append(centered_image)

    return centered_images


def save_vector_as_pgm(vector, filename):
    if len(vector) != IMAGE_SIZE * IMAGE_SIZE:
        raise ValueError("Vector does not match image size")

    with open(filename, "w") as file:

        # PGM header
        file.write("P2\n")
        file.write(f"{IMAGE_SIZE} {IMAGE_SIZE}\n")
        file.write("255\n")

        index = 0

        for i in range(IMAGE_SIZE):

            row = []

            for j in range(IMAGE_SIZE):

                value = vector[index]

                # Clamp value between 0 and 1
                if value < 0:
                    value = 0

                if value > 1:
                    value = 1

                pixel = round(value * 255)

                row.append(str(pixel))

                index += 1

            file.write(" ".join(row) + "\n")

if __name__ == "__main__":

    print("Loading dataset...")

    images, labels = load_dataset()

    print()
    print("Calculating mean vector...")

    mean = calculate_mean_vector(images)

    print("Mean vector calculated.")

    print()
    print("First 10 mean pixel values:")
    print(mean[:10])

    print()
    print("Centering dataset...")

    centered_images = center_dataset(images, mean)

    print("Dataset centered.")

    print()
    print("==============================")
    print("PCA PREPROCESSING INFORMATION")
    print("==============================")

    print("Original dataset:")
    print(rows(images), "x", cols(images))

    print()
    print("Centered dataset:")
    print(rows(centered_images), "x", cols(centered_images))

    print()
    print("Mean vector length:")
    print(len(mean))

    print()
    print("First centered image - first 10 values:")
    print(centered_images[0][:10])

    print()
    print("Saving mean face...")

    save_vector_as_pgm(mean, "mean_face.pgm")

    print("Saved as: mean_face.pgm")

    print()
    print("==============================")
    print("PCA PREPROCESSING COMPLETE")
    print("==============================")