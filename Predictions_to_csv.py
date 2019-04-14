import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn import preprocessing
import numpy as np

#importarea datelor
train_samples = pd.read_csv("../../data/train_samples.csv", dtype = "double", header=None)
train_labels = pd.read_csv("../../data/train_labels.csv", dtype = "double", header=None)
test_samples = pd.read_csv("../../data/test_samples.csv", dtype = "double", header=None)

#preprocesarea datelor / normalizare
def normalize_data(train_samples, test_samples):
    scaler = preprocessing.StandardScaler()
    scaler.fit(train_samples)
    scaled_train_samples = scaler.transform(train_samples)
    scaled_test_samples = scaler.transform(test_samples)
    return scaled_train_samples, scaled_test_samples

#declararea modelului de retea neuronala
mlp_classifier_model = MLPClassifier(hidden_layer_sizes = num_neurons_per_layer, activation='relu', solver='sgd', alpha=reg_coef, batch_size=b_size,
learning_rate='invscaling', learning_rate_init=lr_initial, power_t=lr_coef, max_iter=num_max_epoci, shuffle=True, random_state=None, tol=tol_coef,
momentum=0.9, early_stopping=False, validation_fraction=0.1, n_iter_no_change=num_epoci_fara_schimbare)

scaled_train_data, scaled_test_data = normalize_data(train_samples, test_samples)#normalizarea datelor de antrenare si de test

mlp_classifier_model.fit(scaled_train_data, train_labels) #antrenarea modelului datele normalizate

prediction = mlp_classifier_model.predict(test_samples) #prezicerea etichetelor pentru datele de test
id = np.arange(1,5001)
d={'Id':id,'Prediction':prediction}
data_frame = pd.DataFrame(data=d) #construim un data frame pentru a-l scrie in fisierul csv
data_frame.to_csv('test_labels.csv') #scriem predictiile