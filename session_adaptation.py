class SessionAdapter:

    def __init__(
        self,
        original_centroids,
        classes,
        max_adaptation_weight=0.40,
        samples_for_max_weight=20
    ):
        self.classes = classes
        self.max_adaptation_weight = max_adaptation_weight
        self.samples_for_max_weight = samples_for_max_weight

        self.original_centroids = {
            class_name: list(original_centroids[class_name])
            for class_name in original_centroids
        }

        self.session_sums = {
            class_name: None
            for class_name in classes
        }

        self.session_counts = {
            class_name: 0
            for class_name in classes
        }

        self.effective_centroids = {
            class_name: list(self.original_centroids[class_name])
            for class_name in self.original_centroids
        }

    def add_sample(self, class_name, signature):
        if class_name not in self.original_centroids:
            return False

        if len(signature) == 0:
            return False

        if self.session_sums[class_name] is None:
            self.session_sums[class_name] = [0.0] * len(signature)

        for i in range(len(signature)):
            self.session_sums[class_name][i] += signature[i]

        self.session_counts[class_name] += 1
        self.update_centroid(class_name)

        return True

    def update_centroid(self, class_name):
        count = self.session_counts[class_name]

        if count == 0:
            self.effective_centroids[class_name] = list(
                self.original_centroids[class_name]
            )
            return

        session_sum = self.session_sums[class_name]
        original = self.original_centroids[class_name]

        session_centroid = [
            value / count
            for value in session_sum
        ]

        adaptation_weight = (
            count / self.samples_for_max_weight
        ) * self.max_adaptation_weight

        adaptation_weight = min(
            adaptation_weight,
            self.max_adaptation_weight
        )

        self.effective_centroids[class_name] = [
            (1.0 - adaptation_weight) * original[i]
            + adaptation_weight * session_centroid[i]
            for i in range(len(original))
        ]

    def get_centroids(self):
        return self.effective_centroids

    def get_counts(self):
        return dict(self.session_counts)

    def get_total_samples(self):
        return sum(
            self.session_counts[class_name]
            for class_name in self.classes
        )