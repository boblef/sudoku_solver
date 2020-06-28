from tensorflow.keras.models import load_model


class Model():
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

    def prediction(self, image):
        """
        Return a class which the given is being classified.
        """
        classes = self.model.predict_classes(image)

        if classes == [[0]]:
            return 0
        elif classes == [[1]]:
            return 1
        elif classes == [[2]]:
            return 2
        elif classes == [[3]]:
            return 3
        elif classes == [[4]]:
            return 4
        elif classes == [[5]]:
            return 5
        elif classes == [[6]]:
            return 6
        elif classes == [[7]]:
            return 7
        elif classes == [[8]]:
            return 8
        elif classes == [[9]]:
            return 9
