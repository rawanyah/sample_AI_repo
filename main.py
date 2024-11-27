from sklearn.model_selection import train_test_split
from preprocessing.data_processing import DataProcessor
from models.decision_tree import DecisionTreeModel
from utils.evaluation import Evaluator

def main():
    # Step 1: Load and preprocess data
    data_processor = DataProcessor("data/drug200.csv")
    data = data_processor.load_data()
    X, y = data_processor.preprocess_data(data)

    # Step 2: Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=3)

    # Step 3: Create and train the model
    tree_model = DecisionTreeModel(
        criterion="entropy",
        max_depth=4,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features=None,
    )

    tree_model.train(X_train, y_train)

    # Step 4: Make predictions
    predictions = tree_model.predict(X_test)

    # Step 5: Evaluate the model
    accuracy = Evaluator.calculate_accuracy(y_test, predictions)
    print(f"Model Accuracy: {accuracy:.2f}")

if __name__ == "__main__":
    main()
