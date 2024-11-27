from sklearn.metrics import accuracy_score

class Evaluator:
    @staticmethod
    def calculate_accuracy(y_true, y_pred):
        return accuracy_score(y_true, y_pred)
