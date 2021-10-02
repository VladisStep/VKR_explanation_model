# GENERAL
# PATH PROCESS
import os
import os.path
from pathlib import Path
# IGNORING WARNINGS
from warnings import filterwarnings

# pip install opencv-python
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
# IMAGE PROCESS
import tensorflow.keras.optimizers
from catboost import CatBoostClassifier, CatBoostRegressor
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import BaggingRegressor
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import Lasso
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Ridge
# ACCURACY CONTROL
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
# SKLEARN & TRANSFORMATION
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
# MODEL LAYERS
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
# SKLEARN CLASSIFIER
from xgboost import XGBClassifier, XGBRegressor

filterwarnings("ignore", category=DeprecationWarning)
filterwarnings("ignore", category=FutureWarning)
filterwarnings("ignore", category=UserWarning)

# TRAIN-TEST PATH
Spiral_Train_Path = Path("../parkinsons-drawings/spiral/training")
Spiral_Test_Path = Path("../parkinsons-drawings/spiral/testing")

Spiral_Train_PNG_Path = list(Spiral_Train_Path.glob(r"*/*.png"))
Spiral_Test_PNG_Path = list(Spiral_Test_Path.glob(r"*/*.png"))

# LABELS
Spiral_Train_PNG_Labels = list(map(lambda x: os.path.split(os.path.split(x)[0])[1], Spiral_Train_PNG_Path))
Spiral_Test_PNG_Labels = list(map(lambda x: os.path.split(os.path.split(x)[0])[1], Spiral_Test_PNG_Path))

# TRANSFORMATION TO SERIES STRUCTURE
Spiral_Train_PNG_Path_Series = pd.Series(Spiral_Train_PNG_Path, name="PNG").astype(str)
Spiral_Train_PNG_Labels_Series = pd.Series(Spiral_Train_PNG_Labels, name="CATEGORY")

Spiral_Test_PNG_Path_Series = pd.Series(Spiral_Test_PNG_Path, name="PNG").astype(str)
Spiral_Test_PNG_Labels_Series = pd.Series(Spiral_Test_PNG_Labels, name="CATEGORY")

# TRANSFORMATION TO DATAFRAME STRUCTURE
Main_Spiral_Train_Data = pd.concat([Spiral_Train_PNG_Path_Series, Spiral_Train_PNG_Labels_Series], axis=1)
# print(Main_Spiral_Train_Data.head(-1))

Main_Spiral_Test_Data = pd.concat([Spiral_Test_PNG_Path_Series, Spiral_Test_PNG_Labels_Series], axis=1)
# print(Main_Spiral_Test_Data.head(-1))

# SHUFFLING
Main_Spiral_Train_Data = Main_Spiral_Train_Data.sample(frac=1).reset_index(drop=True)
Main_Spiral_Test_Data = Main_Spiral_Test_Data.sample(frac=1).reset_index(drop=True)
# print(Main_Spiral_Train_Data.head(-1))
# print("---"*20)
# print(Main_Spiral_Test_Data.head(-1))

# VISUALIZATION
plt.style.use("dark_background")
sns.countplot(Main_Spiral_Train_Data["CATEGORY"])
plt.show()

sns.countplot(Main_Spiral_Test_Data["CATEGORY"])
plt.show()

Main_Spiral_Train_Data['CATEGORY'].value_counts().plot.pie(figsize=(7, 7))
plt.show()

Main_Spiral_Test_Data['CATEGORY'].value_counts().plot.pie(figsize=(7, 7))
plt.show()

# fig, axes = plt.subplots(nrows=5,
#                         ncols=5,
#                         figsize=(10,10),
#                         subplot_kw={"xticks":[],"yticks":[]})
#
# for i,ax in enumerate(axes.flat):
#     ax.imshow(plt.imread(Main_Spiral_Train_Data["PNG"][i]))
#     ax.set_title(Main_Spiral_Train_Data["CATEGORY"][i])
# plt.tight_layout()
# plt.show()

# CLASSIFIERS
Spiral_New_JPG_Path = []
for i in range(0, 72):
    x = cv2.imread(Main_Spiral_Train_Data["PNG"][i])
    x = np.array(x).astype("float32")
    x = x.mean()
    Spiral_New_JPG_Path.append(x)

Spiral_New_JPG_Path_Series = pd.Series(Spiral_New_JPG_Path, name="PNG")
# print(Spiral_New_JPG_Path_Series)

encode = LabelEncoder()

Spiral_New_JPG_Labels = encode.fit_transform(Main_Spiral_Train_Data["CATEGORY"])

Spiral_New_JPG_Labels_Series = pd.Series(Spiral_New_JPG_Labels, name="CATEGORY")
# print(Spiral_New_JPG_Labels_Series)

Main_Spiral_New_Data = pd.concat([Spiral_New_JPG_Path_Series, Spiral_New_JPG_Labels_Series], axis=1)
# print(Main_Spiral_New_Data)

# TRAIN & TEST
x = Main_Spiral_New_Data[["PNG"]]
y = Main_Spiral_New_Data["CATEGORY"]
xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size=0.1, random_state=42)

# MODELS
lj = LogisticRegression(solver="liblinear").fit(xTrain, yTrain)
gnb = GaussianNB().fit(xTrain, yTrain)
knnc = KNeighborsClassifier().fit(xTrain, yTrain)
cartc = DecisionTreeClassifier(random_state=42).fit(xTrain, yTrain)
rfc = RandomForestClassifier(random_state=42, verbose=False).fit(xTrain, yTrain)
gbmc = GradientBoostingClassifier(verbose=False).fit(xTrain, yTrain)
xgbc = XGBClassifier().fit(xTrain, yTrain)
lgbmc = LGBMClassifier().fit(xTrain, yTrain)
catbc = CatBoostClassifier(verbose=False).fit(xTrain, yTrain)

modelsc = [lj, gnb, knnc, cartc, rfc, gbmc, xgbc, lgbmc, catbc]

for model in modelsc:
    name = model.__class__.__name__
    predict = model.predict(xTest)
    R2CV = cross_val_score(model, xTest, yTest, verbose=False).mean()
    error = -cross_val_score(model, xTest, yTest, scoring="neg_mean_squared_error", verbose=False).mean()
    print(name + ": ")
    print("-" * 10)
    print("ACC-->", accuracy_score(yTest, predict))
    print("R2CV-->", R2CV * 100)
    print("MEAN SQUARED ERROR-->", np.sqrt(error))
    print("-" * 30)

lm = LinearRegression().fit(xTrain, yTrain)
pls = PLSRegression().fit(xTrain, yTrain)
ridge = Ridge().fit(xTrain, yTrain)
lasso = Lasso().fit(xTrain, yTrain)
elasticnet = ElasticNet().fit(xTrain, yTrain)
knnr = KNeighborsRegressor().fit(xTrain, yTrain)
cartr = DecisionTreeRegressor(random_state=42).fit(xTrain, yTrain)
baggr = BaggingRegressor(random_state=42, bootstrap_features=True, verbose=False).fit(xTrain, yTrain)
rfr = RandomForestRegressor(random_state=42, verbose=False).fit(xTrain, yTrain)
gbmr = GradientBoostingRegressor(verbose=False).fit(xTrain, yTrain)
xgbr = XGBRegressor().fit(xTrain, yTrain)
lgbmr = LGBMRegressor().fit(xTrain, yTrain)
catbr = CatBoostRegressor(verbose=False).fit(xTrain, yTrain)
models = [lm, pls, ridge, lasso, elasticnet, knnr,
          cartr, baggr, rfr, gbmr, xgbr, lgbmr, catbr]

for model in models:
    name = model.__class__.__name__
    R2CV = cross_val_score(model, xTest, yTest, scoring="r2").mean()
    error = -cross_val_score(model, xTest, yTest, scoring="neg_mean_squared_error").mean()
    print(name + ": ")
    print("-" * 10)
    print(R2CV)
    print(np.sqrt(error))
    print("-" * 30)

# IMAGE GENERATOR PROCESS
# APPLYING GENERATOR
Train_Generator = ImageDataGenerator(rescale=1. / 255,
                                     zoom_range=0.7,
                                     shear_range=0.7,
                                     rotation_range=50,
                                     horizontal_flip=True,
                                     brightness_range=[0.2, 0.9],
                                     vertical_flip=True,
                                     validation_split=0.1)
Train_Spiral_Set = Train_Generator.flow_from_dataframe(dataframe=Main_Spiral_Train_Data,
                                                       x_col="PNG",
                                                       y_col="CATEGORY",
                                                       color_mode="grayscale",
                                                       class_mode="categorical",
                                                       subset="training")
Validation_Spiral_Set = Train_Generator.flow_from_dataframe(dataframe=Main_Spiral_Train_Data,
                                                            x_col="PNG",
                                                            y_col="CATEGORY",
                                                            color_mode="grayscale",
                                                            class_mode="categorical",
                                                            subset="validation")
Test_Spiral_Set = Train_Generator.flow_from_dataframe(dataframe=Main_Spiral_Test_Data,
                                                      x_col="PNG",
                                                      y_col="CATEGORY",
                                                      color_mode="grayscale",
                                                      class_mode="categorical")
print("TRAIN: ")
print(Train_Spiral_Set.class_indices)
print(Train_Spiral_Set.classes[0:5])
print(Train_Spiral_Set.image_shape)
print("---" * 20)
print("VALIDATION: ")
print(Validation_Spiral_Set.class_indices)
print(Validation_Spiral_Set.classes[0:5])
print(Validation_Spiral_Set.image_shape)
print("---" * 20)
print("TEST: ")
print(Test_Spiral_Set.class_indices)
print(Test_Spiral_Set.classes[0:5])
print(Test_Spiral_Set.image_shape)

# MODEL
# CNN
Call_Back_Early = tf.keras.callbacks.EarlyStopping(monitor="val_accuracy",
                                                   patience=7,
                                                   mode="max")
Call_Back_Check = tf.keras.callbacks.ModelCheckpoint(monitor="val_accuracy",
                                                     save_best_only=True,
                                                     filepath="./my_model")
Model_One = Sequential()

Model_One.add(Conv2D(2, (15, 15), activation="relu",
                     input_shape=(256, 256, 1)))
Model_One.add(MaxPooling2D((2, 2)))
Model_One.add(Dropout(0.2))

Model_One.add(Conv2D(4, (10, 10), activation="relu",
                     strides=(2, 2)))
Model_One.add(MaxPooling2D((2, 2)))
Model_One.add(Dropout(0.2))

Model_One.add(Flatten())
Model_One.add(Dropout(0.5))
Model_One.add(Dense(512, activation="relu"))
Model_One.add(Dense(2, activation="softmax"))
Model_One.compile(optimizer=tensorflow.keras.optimizers.RMSprop(lr=0.001), loss="categorical_crossentropy",
                  metrics=["accuracy"])
CNN_Model_One = Model_One.fit(Train_Spiral_Set,
                              validation_data=Validation_Spiral_Set,
                              callbacks=[Call_Back_Check, Call_Back_Early],
                              epochs=50)

Model_Results = Model_One.evaluate(Test_Spiral_Set)
print("LOSS:  " + "%.4f" % Model_Results[0])
print("ACCURACY:  " + "%.2f" % Model_Results[1])

# ANN
Model_Two = tf.keras.models.Sequential([
    # inputs
    tf.keras.layers.experimental.preprocessing.Rescaling(1. / 255),
    tf.keras.layers.Flatten(input_shape=(113,)),
    # hiddens layers
    tf.keras.layers.Dense(10, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    # output layer
    tf.keras.layers.Dense(2, activation="softmax")
])

lossfunc = tf.keras.losses.CategoricalCrossentropy(from_logits=True)

Model_Two.compile(optimizer='adam', loss=lossfunc, metrics=['accuracy'])

ANN_Model = Model_Two.fit(Train_Spiral_Set,
                          validation_data=Validation_Spiral_Set,
                          epochs=50, batch_size=10)

Model_Results_Two = Model_Two.evaluate(Test_Spiral_Set)
print("LOSS:  " + "%.4f" % Model_Results[0])
print("ACCURACY:  " + "%.2f" % Model_Results[1])


# PREDICTION PROCESS
def predict_prob(number):
    return [number[0], 1 - number[0]]


# CNN
Prediction_One = Model_One.predict(Test_Spiral_Set)
Prediction_One = Prediction_One.argmax(axis=-1)
Predict_Proba_One = Model_One.predict_prob(Test_Spiral_Set)
# ANN
Prediction_Two = Model_Two.predict(Test_Spiral_Set)
Prediction_Two = Prediction_Two.argmax(axis=-1)
Predict_Proba_Two = Model_Two.predict_prob(Test_Spiral_Set)
# COMMUNITY
Main_Predict = 0.5 * (Predict_Proba_One + Predict_Proba_Two)
print(Main_Predict)

fig, axes = plt.subplots(nrows=5,
                         ncols=5,
                         figsize=(20, 20),
                         subplot_kw={'xticks': [], 'yticks': []})

for i, ax in enumerate(axes.flat):
    ax.imshow(plt.imread(Main_Spiral_Test_Data["PNG"].iloc[i]))
    ax.set_title(f"PREDICTION:{Main_Predict[i]}")
plt.tight_layout()
plt.show()
