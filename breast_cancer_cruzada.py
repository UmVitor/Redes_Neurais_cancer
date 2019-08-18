import pandas as pd
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.wrappers.scikit_learn import KerasClassifier #Biblioteca para a validação cruzada
from sklearn.model_selection import cross_val_score

previsores = pd.read_csv('entradas-breast.csv')
classe = pd.read_csv('saidas-breast.csv')

def criarRede():
    classificador = Sequential()
    classificador.add(Dense(units = 16 , activation = 'relu' ,kernel_initializer = 'random_uniform',input_dim = 30)) #(entrada + saida)/2 - Primeira camada oculta
    classificador.add(Dropout(0.2))
    classificador.add(Dense(units = 16 , activation = 'relu' ,kernel_initializer = 'random_uniform')) #Segunda camada oculta
    classificador.add(Dropout(0.2))
    classificador.add(Dense(units = 1, activation = 'sigmoid')) #Camada de saida
    otimizador = keras.optimizers.Adam(lr = 0.001, decay = 0.0001, clipvalue = 0.5)
    classificador.compile(optimizer = otimizador, loss='binary_crossentropy', metrics = ['binary_accuracy']) #compilação com os parametros adicionais da rede
    
    return classificador

classificador = KerasClassifier(build_fn = criarRede , epochs = 100 , batch_size = 10)
resultados = cross_val_score(estimator = classificador , X = previsores , y = classe , cv = 10 , scoring = 'accuracy')

media = resultados.mean()
desvio = resultados.std()