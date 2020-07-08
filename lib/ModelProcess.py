from tensorflow.keras.models import load_model


class ModelProcess():
    def __init__(self, model_path):
        """
        Load the CNN model used to capture digits in an image.
        """
        self._model_path = model_path
        self._load_model()

    def _load_model(self):
        self.model = load_model(self._model_path)

    def get_model(self):
        return self.model
