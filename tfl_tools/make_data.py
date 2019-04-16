from sklearn import model_selection


class MakeData:
    def __init__(self, X_data, Y_data, n_splits):
        self.X_data = X_data
        self.Y_data = Y_data
        self.n_splits = n_splits

    def gen(self):
        seed = 7
        for train_index, test_index in \
                model_selection.KFold(n_splits=self.n_splits,shuffle=True,random_state=seed).split(self.X_data):
            X_train, X_test = self.X_data[train_index], self.X_data[test_index]
            Y_train, Y_test = self.Y_data[train_index], self.Y_data[test_index]
            yield X_train,Y_train,X_test,Y_test