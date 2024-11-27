import pandas as pd
from sklearn import preprocessing

class DataProcessor:
    def __init__(self, file_path):
        self.file_path = file_path

    def load_data(self):
        data = pd.read_csv(self.file_path, delimiter=",")
        return data

    def preprocess_data(self, data):
        X = data[['Age', 'Sex', 'BP', 'Cholesterol', 'Na_to_K']].values
        y = data['Drug']

        # Encode categorical variables
        le_sex = preprocessing.LabelEncoder()
        le_sex.fit(['F', 'M'])
        X[:, 1] = le_sex.transform(X[:, 1])

        le_BP = preprocessing.LabelEncoder()
        le_BP.fit(['LOW', 'NORMAL', 'HIGH'])
        X[:, 2] = le_BP.transform(X[:, 2])

        le_Chol = preprocessing.LabelEncoder()
        le_Chol.fit(['NORMAL', 'HIGH'])
        X[:, 3] = le_Chol.transform(X[:, 3])

        return X, y
