from sklearn import svm


class SConfig:
    def __init__(self) -> None:
        super().__init__()
        C = 1
        gamma = 'scale'
        # # kernel = 'rbf'
        # kernel = 'poly'
        kernel = 'linear'
        decision_function_shape = 'ovr'
        self.inputs = {
            'C': C,
            'kernel': kernel,
            'gamma': gamma,
            'decision_function_shape': decision_function_shape,
        }
        self.model_desc = f'svm_gamma_{gamma}'


def train_svm(X_train, y_train, X_test, y_test, C, kernel, gamma, decision_function_shape):
    clf = svm.SVC(C=C, kernel=kernel, gamma=gamma, decision_function_shape=decision_function_shape)

    clf.fit(X_train, y_train)

    outputs = clf.predict(X_test)
    targets = y_test
    return targets, outputs

