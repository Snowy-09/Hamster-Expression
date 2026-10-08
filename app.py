import cv2
import os
import random
import time
import tkinter as tk
from collections import deque
import numpy as np

from model import load_model
from webcam import get_face_detector, preprocess_frame
from matrix import dot_product
from classifier import weighted_distance, calculate_confidence
from session_adaptation import SessionAdapter


MODEL_FILE = "model.json"
OUTPUT_DIR = "output_images"
WINDOW_NAME = "Hamster Expression AI"

WINDOW_WIDTH = 1200
WINDOW_HEIGHT = 700

BASE_WIDTH = WINDOW_WIDTH
BASE_HEIGHT = WINDOW_HEIGHT


def get_screen_resolution():
    root = tk.Tk()
    root.withdraw()
    width = root.winfo_screenwidth()
    height = root.winfo_screenheight()
    root.destroy()
    return width, height


SCREEN_WIDTH, SCREEN_HEIGHT = get_screen_resolution()
SCALE_X = SCREEN_WIDTH / BASE_WIDTH
SCALE_Y = SCREEN_HEIGHT / BASE_HEIGHT

CLASSES = [
    "Angry",
    "Cry",
    "Happy",
    "Neutral",
    "Sad",
    "Surprised"
]


SMOOTHING_WINDOW_SECONDS = 2.0
STABILITY_TIME_SECONDS = 0.5
CONFIDENCE_THRESHOLD = 0.20

MAX_ADAPTATION_WEIGHT = 0.40
SAMPLES_FOR_MAX_ADAPTATION = 20

FEEDBACK_Y1 = 625
FEEDBACK_Y2 = 660

YES_X1 = 390
YES_X2 = 510

NO_X1 = 690
NO_X2 = 810


def project_image(centered_image, eigenfaces):
    signature = []
    for eigenface in eigenfaces:
        signature.append(dot_product(eigenface, centered_image))
    return signature


def classify_signature(signature, centroids, eigenvalues):
    distances = {}

    for class_name in centroids:
        distances[class_name] = weighted_distance(
            signature,
            centroids[class_name],
            eigenvalues
        )

    predicted_class = min(distances, key=distances.get)
    return predicted_class, distances


def classify_image(image_vector, model, centroids):
    mean_vector = model["mean_vector"]
    eigenfaces = model["eigenfaces"]
    eigenvalues = model["eigenvalues"]

    centered_image = [
        image_vector[i] - mean_vector[i]
        for i in range(len(image_vector))
    ]

    signature = project_image(centered_image, eigenfaces)

    predicted_class, distances = classify_signature(
        signature,
        centroids,
        eigenvalues
    )

    confidence_values = calculate_confidence(distances)
    confidence = confidence_values[predicted_class]

    return (
        predicted_class,
        confidence,
        confidence_values,
        distances,
        signature
    )


def calculate_smoothed_confidence(confidence_history):
    if len(confidence_history) == 0:
        return {}

    smoothed = {}

    for class_name in CLASSES:
        total = 0.0
        count = 0

        for confidence_values in confidence_history:
            if class_name in confidence_values:
                total += confidence_values[class_name]
                count += 1

        smoothed[class_name] = total / count if count > 0 else 0.0

    return smoothed


def calculate_smoothed_signature(signature_history):
    if len(signature_history) == 0:
        return None

    dimension = len(signature_history[0])
    result = [0.0] * dimension

    for signature in signature_history:
        for i in range(dimension):
            result[i] += signature[i]

    count = len(signature_history)

    for i in range(dimension):
        result[i] /= count

    return result


def get_smoothed_prediction(smoothed_confidence):
    if len(smoothed_confidence) == 0:
        return None, 0.0

    predicted_class = max(
        smoothed_confidence,
        key=smoothed_confidence.get
    )

    confidence = smoothed_confidence[predicted_class]

    if confidence < CONFIDENCE_THRESHOLD:
        return "Miscellaneous", confidence

    return predicted_class, confidence


def load_hamster_images():
    hamster_images = {}
    all_folders = CLASSES + ["Miscellaneous"]

    print()
    print("Loading hamster images...")
    print("--------------------------------")

    for class_name in all_folders:
        folder = os.path.join(OUTPUT_DIR, class_name)
        hamster_images[class_name] = []

        if not os.path.exists(folder):
            print(class_name, ": folder missing")
            continue

        for filename in os.listdir(folder):
            path = os.path.join(folder, filename)

            if not os.path.isfile(path):
                continue

            image = cv2.imread(path, cv2.IMREAD_COLOR)

            if image is not None:
                hamster_images[class_name].append(image)

        print(
            class_name,
            ":",
            len(hamster_images[class_name]),
            "image(s)"
        )

    print("--------------------------------")
    print()

    return hamster_images


def get_random_hamster(hamster_images, class_name):
    images = hamster_images.get(class_name, [])

    if len(images) == 0:
        return None

    return random.choice(images)


def fit_image(image, width, height):
    image_height, image_width = image.shape[:2]

    scale_x = width / image_width
    scale_y = height / image_height
    scale = min(scale_x, scale_y)

    new_width = int(image_width * scale)
    new_height = int(image_height * scale)

    resized = cv2.resize(
        image,
        (new_width, new_height),
        interpolation=cv2.INTER_AREA
    )

    canvas = np.ones(
        (height, width, 3),
        dtype=np.uint8
    ) * 255

    x = (width - new_width) // 2
    y = (height - new_height) // 2

    canvas[y:y + new_height, x:x + new_width] = resized

    return canvas


def draw_text(image, text, position, size=0.7, thickness=2):
    cv2.putText(
        image,
        text,
        position,
        cv2.FONT_HERSHEY_SIMPLEX,
        size,
        (40, 40, 40),
        thickness,
        cv2.LINE_AA
    )


def draw_button(image, x1, y1, x2, y2, text):
    cv2.rectangle(
        image,
        (x1, y1),
        (x2, y2),
        (220, 220, 220),
        -1
    )

    cv2.rectangle(
        image,
        (x1, y1),
        (x2, y2),
        (80, 80, 80),
        2
    )

    text_size = cv2.getTextSize(
        text,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        2
    )[0]

    text_x = x1 + ((x2 - x1) - text_size[0]) // 2
    text_y = y1 + ((y2 - y1) + text_size[1]) // 2

    cv2.putText(
        image,
        text,
        (text_x, text_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (40, 40, 40),
        2,
        cv2.LINE_AA
    )


def mouse_callback(event, x, y, flags, state):
    if event != cv2.EVENT_LBUTTONDOWN:
        return

    x = int(x / SCALE_X)
    y = int(y / SCALE_Y)

    if not state["feedback_enabled"]:
        return

    signature = state["feedback_signature"]

    if signature is None:
        return

    if state["feedback_mode"] == "confirm":

        if (
            YES_X1 <= x <= YES_X2
            and FEEDBACK_Y1 <= y <= FEEDBACK_Y2
        ):
            predicted = state["current_expression"]

            if predicted is not None and predicted != "Miscellaneous":
                added = state["adapter"].add_sample(
                    predicted,
                    signature
                )

                if added:
                    state["feedback_message"] = (
                        "Confirmed as " + predicted
                    )
                    print("\n[FEEDBACK] Confirmed:", predicted)
            else:
                state["feedback_message"] = (
                    "Miscellaneous confirmed"
                )
                print("\n[FEEDBACK] Miscellaneous confirmed")

            state["confidence_history"].clear()
            state["signature_history"].clear()
            state["candidate_expression"] = None
            state["candidate_since"] = None
            state["feedback_enabled"] = False
            state["feedback_cooldown_until"] = time.time() + 2.0
            return

        if (
            NO_X1 <= x <= NO_X2
            and FEEDBACK_Y1 <= y <= FEEDBACK_Y2
        ):
            state["feedback_mode"] = "label"
            state["feedback_message"] = (
                "Select the correct expression"
            )
            return

    if state["feedback_mode"] == "label":
        button_width = 170
        gap = 5
        start_x = 55

        for i, class_name in enumerate(CLASSES):
            x1 = start_x + i * (button_width + gap)
            x2 = x1 + button_width

            if (
                x1 <= x <= x2
                and FEEDBACK_Y1 <= y <= FEEDBACK_Y2
            ):
                added = state["adapter"].add_sample(
                    class_name,
                    signature
                )

                if added:
                    state["feedback_message"] = (
                        "Corrected to " + class_name
                    )
                    print(
                        "\n[FEEDBACK] Corrected:",
                        class_name
                    )

                state["confidence_history"].clear()
                state["signature_history"].clear()
                state["candidate_expression"] = None
                state["candidate_since"] = None
                state["feedback_mode"] = "confirm"
                state["feedback_enabled"] = False
                state["feedback_cooldown_until"] = time.time() + 2.0
                return


def main():
    print("===================================")
    print("     HAMSTER EXPRESSION AI")
    print("===================================")
    print()

    print(
        "Confidence threshold:",
        CONFIDENCE_THRESHOLD * 100,
        "%"
    )

    print(
        "Smoothing window:",
        SMOOTHING_WINDOW_SECONDS,
        "seconds"
    )

    print(
        "Stability time:",
        STABILITY_TIME_SECONDS,
        "seconds"
    )

    print(
        "Maximum session adaptation:",
        MAX_ADAPTATION_WEIGHT * 100,
        "%"
    )

    print()

    print("Loading model...")
    model = load_model(MODEL_FILE)
    print("Model loaded.")

    detector = get_face_detector()
    hamster_images = load_hamster_images()

    adapter = SessionAdapter(
        model["centroids"],
        CLASSES,
        MAX_ADAPTATION_WEIGHT,
        SAMPLES_FOR_MAX_ADAPTATION
    )

    print("Session adaptation ready.")

    camera = cv2.VideoCapture(0)

    if not camera.isOpened():
        raise RuntimeError("Could not open webcam.")

    print("Webcam started.")
    print("Press Q to quit.")
    print()

    current_expression = None
    current_hamster = None
    current_confidence = 0.0

    confidence_history = deque()
    signature_history = deque()

    candidate_expression = None
    candidate_since = None

    frame_counter = 0

    feedback_enabled = False
    feedback_mode = "confirm"
    feedback_signature = None
    feedback_message = ""
    feedback_cooldown_until = 0.0

    state = {
        "adapter": adapter,
        "current_expression": current_expression,
        "feedback_enabled": feedback_enabled,
        "feedback_mode": feedback_mode,
        "feedback_signature": feedback_signature,
        "feedback_message": feedback_message,
        "confidence_history": confidence_history,
        "signature_history": signature_history,
        "candidate_expression": candidate_expression,
        "candidate_since": candidate_since,
        "feedback_cooldown_until": feedback_cooldown_until
    }

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    cv2.setWindowProperty(
        WINDOW_NAME,
        cv2.WND_PROP_FULLSCREEN,
        cv2.WINDOW_FULLSCREEN
    )

    cv2.setMouseCallback(
        WINDOW_NAME,
        mouse_callback,
        state
    )

    while True:
        current_expression = state["current_expression"]
        feedback_enabled = state["feedback_enabled"]
        feedback_mode = state["feedback_mode"]
        feedback_signature = state["feedback_signature"]
        feedback_message = state["feedback_message"]
        candidate_expression = state["candidate_expression"]
        candidate_since = state["candidate_since"]
        feedback_cooldown_until = state["feedback_cooldown_until"]

        success, frame = camera.read()

        if not success:
            break

        frame = cv2.flip(frame, 1)

        image_vector, face_box = preprocess_frame(
            frame,
            detector
        )

        if image_vector is not None:
            centroids = adapter.get_centroids()

            (
                predicted_class,
                confidence,
                confidence_values,
                distances,
                signature
            ) = classify_image(
                image_vector,
                model,
                centroids
            )

            current_time = time.time()

            confidence_history.append(
                (current_time, confidence_values)
            )

            signature_history.append(
                (current_time, signature)
            )

            while (
                len(confidence_history) > 0
                and current_time - confidence_history[0][0]
                > SMOOTHING_WINDOW_SECONDS
            ):
                confidence_history.popleft()

            while (
                len(signature_history) > 0
                and current_time - signature_history[0][0]
                > SMOOTHING_WINDOW_SECONDS
            ):
                signature_history.popleft()

            recent_confidences = [
                values
                for timestamp, values in confidence_history
            ]

            recent_signatures = [
                values
                for timestamp, values in signature_history
            ]

            smoothed_confidence = calculate_smoothed_confidence(
                recent_confidences
            )

            smoothed_signature = calculate_smoothed_signature(
                recent_signatures
            )

            (
                smoothed_expression,
                smoothed_confidence_value
            ) = get_smoothed_prediction(
                smoothed_confidence
            )

            if smoothed_expression is not None:

                if smoothed_expression != candidate_expression:
                    candidate_expression = smoothed_expression
                    candidate_since = current_time

                elif candidate_since is not None:
                    elapsed = current_time - candidate_since

                    if elapsed >= STABILITY_TIME_SECONDS:

                        if candidate_expression != current_expression:
                            current_expression = candidate_expression
                            current_confidence = smoothed_confidence_value

                            current_hamster = get_random_hamster(
                                hamster_images,
                                current_expression
                            )

                            feedback_enabled = True
                            feedback_signature = smoothed_signature
                            feedback_mode = "confirm"
                            feedback_message = ""

                        else:
                            current_confidence = smoothed_confidence_value

                            if smoothed_signature is not None:
                                feedback_signature = smoothed_signature

            if (
                not feedback_enabled
                and current_time >= feedback_cooldown_until
                and current_expression is not None
            ):
                feedback_enabled = True

            state["current_expression"] = current_expression
            state["feedback_enabled"] = feedback_enabled
            state["feedback_mode"] = feedback_mode
            state["feedback_signature"] = feedback_signature
            state["feedback_message"] = feedback_message
            state["candidate_expression"] = candidate_expression
            state["candidate_since"] = candidate_since
            state["feedback_cooldown_until"] = feedback_cooldown_until

            frame_counter += 1

            if frame_counter % 30 == 0:
                print("\nRaw prediction:", predicted_class)
                print(
                    "Raw confidence:",
                    round(confidence * 100, 2),
                    "%"
                )
                print(
                    "Smoothed prediction:",
                    smoothed_expression
                )
                print(
                    "Smoothed confidence:",
                    round(
                        smoothed_confidence_value * 100,
                        2
                    ),
                    "%"
                )
                print(
                    "Session samples:",
                    adapter.get_total_samples()
                )
                print(
                    "Session counts:",
                    adapter.get_counts()
                )

        ui = np.ones(
            (WINDOW_HEIGHT, WINDOW_WIDTH, 3),
            dtype=np.uint8
        ) * 245

        cv2.rectangle(
            ui,
            (0, 0),
            (WINDOW_WIDTH, 80),
            (55, 55, 55),
            -1
        )

        cv2.putText(
            ui,
            "HAMSTER EXPRESSION AI",
            (330, 52),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )

        webcam_x = 40
        webcam_y = 120
        webcam_width = 540
        webcam_height = 400

        display_frame = cv2.resize(
            frame,
            (webcam_width, webcam_height)
        )

        if face_box is not None:
            x, y, w, h = face_box

            original_height, original_width = frame.shape[:2]

            scale_x = webcam_width / original_width
            scale_y = webcam_height / original_height

            x = int(x * scale_x)
            y = int(y * scale_y)
            w = int(w * scale_x)
            h = int(h * scale_y)

            cv2.rectangle(
                display_frame,
                (x, y),
                (x + w, y + h),
                (0, 220, 0),
                3
            )

        ui[
            webcam_y:webcam_y + webcam_height,
            webcam_x:webcam_x + webcam_width
        ] = display_frame

        hamster_x = 620
        hamster_y = 120
        hamster_width = 540
        hamster_height = 400

        if current_hamster is not None:
            hamster_panel = fit_image(
                current_hamster,
                hamster_width,
                hamster_height
            )

        else:
            hamster_panel = np.ones(
                (hamster_height, hamster_width, 3),
                dtype=np.uint8
            ) * 255

            cv2.putText(
                hamster_panel,
                "No hamster reaction",
                (140, 210),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (100, 100, 100),
                2,
                cv2.LINE_AA
            )

        ui[
            hamster_y:hamster_y + hamster_height,
            hamster_x:hamster_x + hamster_width
        ] = hamster_panel

        if current_expression is not None:
            expression_text = (
                "EXPRESSION: "
                + current_expression.upper()
            )

            confidence_text = (
                "CONFIDENCE: "
                + str(round(current_confidence * 100))
                + "%"
            )

            draw_text(
                ui,
                expression_text,
                (365, 575),
                0.9,
                2
            )

            draw_text(
                ui,
                confidence_text,
                (470, 605),
                0.7,
                2
            )

        else:
            draw_text(
                ui,
                "EXPRESSION: --",
                (430, 575),
                0.9,
                2
            )

            draw_text(
                ui,
                "CONFIDENCE: --",
                (450, 605),
                0.7,
                2
            )

        if feedback_enabled and current_expression is not None:

            if feedback_mode == "confirm":
                draw_button(
                    ui,
                    YES_X1,
                    FEEDBACK_Y1,
                    YES_X2,
                    FEEDBACK_Y2,
                    "YES"
                )

                draw_button(
                    ui,
                    NO_X1,
                    FEEDBACK_Y1,
                    NO_X2,
                    FEEDBACK_Y2,
                    "NO"
                )

            else:
                button_width = 170
                gap = 5
                start_x = 55

                for i, class_name in enumerate(CLASSES):
                    x1 = start_x + i * (button_width + gap)
                    x2 = x1 + button_width

                    draw_button(
                        ui,
                        x1,
                        FEEDBACK_Y1,
                        x2,
                        FEEDBACK_Y2,
                        class_name
                    )

        if feedback_message != "":
            cv2.putText(
                ui,
                feedback_message,
                (420, 110),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (70, 70, 70),
                1,
                cv2.LINE_AA
            )

        session_total = adapter.get_total_samples()

        cv2.putText(
            ui,
            "Session samples: " + str(session_total),
            (900, 685),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (100, 100, 100),
            1,
            cv2.LINE_AA
        )

        cv2.putText(
            ui,
            "Press Q to quit",
            (510, 690),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (100, 100, 100),
            1,
            cv2.LINE_AA
        )

        ui = cv2.resize(
            ui,
            (SCREEN_WIDTH, SCREEN_HEIGHT),
            interpolation=cv2.INTER_LINEAR
        )

        cv2.imshow(WINDOW_NAME, ui)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

    camera.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()