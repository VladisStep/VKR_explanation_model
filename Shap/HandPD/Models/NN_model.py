from joblib import dump
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import my_utils
from Shap.HandPD.data import load_hand_pd

path_to_project = my_utils.path_to_project


def create_NN(filename, X_train, Y_train):
    # model = keras.models.Sequential([
    #     keras.layers.GRU(128, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[0])),
    #     keras.layers.GRU(128),
    #     keras.layers.Dense(1, activation="sigmoid")
    # ])
    #
    # model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
    # model_checkpoint_callback = keras.callbacks.ModelCheckpoint(
    #     filepath=filename,
    #     save_weights_only=True,
    #     monitor='val_accuracy',
    #     mode='max',
    #     save_best_only=True)
    #
    # model.fit(X_train, Y_train, epochs=30, batch_size=32, validation_data=(X_test, Y_test),
    #           callbacks=[model_checkpoint_callback])
    model = make_pipeline(
        StandardScaler(),
        MLPRegressor(hidden_layer_sizes=(5,), activation='logistic', max_iter=10000, learning_rate='invscaling',
                     random_state=0)
    )
    model.fit(X_train.values, Y_train)

    dump(model, filename)


if __name__ == "__main__":
    test_size = 0.5
    X, Y = load_hand_pd()
    X_train, X_test, Y_train, Y_test = train_test_split(*[X, Y], test_size=test_size, random_state=0)
    create_NN(path_to_project + "Models/NN.joblib", X_train, Y_train)
