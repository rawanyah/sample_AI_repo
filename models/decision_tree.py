from sklearn.tree import DecisionTreeClassifier

class DecisionTreeModel:
    def __init__(self, criterion="gini", max_depth=None, min_samples_split=2, 
                min_samples_leaf=1, max_features=None, max_leaf_nodes=None, 
                min_impurity_decrease=0.0, splitter="best"):
        """
        Initialize the Decision Tree Model with customizable hyperparameters.
        """
        self.model = DecisionTreeClassifier(
        criterion=criterion,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        max_leaf_nodes=max_leaf_nodes,
        min_impurity_decrease=min_impurity_decrease,
        splitter=splitter
    )

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        return self.model.predict(X_test)

    def get_model(self):
        return self.model
