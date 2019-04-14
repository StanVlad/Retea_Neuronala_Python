import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn import preprocessing
import numpy as np
from sklearn.metrics import confusion_matrix

#importarea datelor full
train_samples_full = pd.read_csv("../../data/train_samples.csv", dtype = "double", header=None)
train_labels_full = pd.read_csv("../../data/train_labels.csv", dtype = "double", header=None)
#nu importam si datele de test, pt ca vom valida, pe rand, pe o parte din cele 15k date de training

#impart datele de training in 3 pentru procedura de cross-validation
pseudo_test = np.split(np.copy(train_samples_full), 3)
pseudo_labels = np.split(np.copy(train_labels_full), 3)

train_samples = np.zeros((3,10000,4096))
train_labels = np.zeros((3,10000))
train_samples[0] = np.concatenate((np.copy(pseudo_test[1]), np.copy(pseudo_test[2])), axis = 0)
train_samples[1] = np.concatenate((np.copy(pseudo_test[0]), np.copy(pseudo_test[2])), axis = 0)
train_samples[2] = np.concatenate((np.copy(pseudo_test[0]), np.copy(pseudo_test[1])), axis = 0)

train_labels[0] = (np.concatenate((np.copy(pseudo_labels[1]), np.copy(pseudo_labels[2])), axis = 0)).reshape((10000))
train_labels[1] = (np.concatenate((np.copy(pseudo_labels[0]), np.copy(pseudo_labels[2])), axis = 0)).reshape((10000))
train_labels[2] = (np.concatenate((np.copy(pseudo_labels[0]), np.copy(pseudo_labels[1])), axis = 0)).reshape((10000))

#functie de preprocesare a datelor / normalizare
def normalize_data(train_samples, test_samples):
    scaler = preprocessing.StandardScaler()
    scaler.fit(train_samples)
    scaled_train_samples = scaler.transform(train_samples)
    scaled_test_samples = scaler.transform(test_samples)
    return scaled_train_samples, scaled_test_samples

#declararea modelului de retea folosita; detalii in documentatie
mlp_classifier_model = MLPClassifier(hidden_layer_sizes = num_neurons_per_layer, activation='relu', solver='sgd', alpha=reg_coef, batch_size=b_size,
learning_rate='invscaling', learning_rate_init=lr_initial, power_t=lr_coef, max_iter=num_max_epoci, shuffle=True, random_state=None, tol=tol_coef,
momentum=0.9, early_stopping=False, validation_fraction=0.1, n_iter_no_change=num_epoci_fara_schimbare)

# 3-fold cross-validation
for i in range(3):
    scaled_train_data, scaled_test_data = normalize_data(train_samples[i], pseudo_test[i])#normalizarea datelor pe fiecare fold

    mlp_classifier_model.fit(scaled_train_data, train_labels[i]) #antrenarea modelului pe fiecare fold

    accuracy = mlp_classifier_model.score(scaled_test_data, pseudo_labels[i]) #calculam acuratetea pentru fiecare multime de testare
    print(accuracy)

    prediction = mlp_classifier_model.predict(scaled_test_data) #prezicerea etichetelor

    matrix = confusion_matrix(pseudo_labels[i], prediction) #calcularea matricei de confuzie
    print(matrix)