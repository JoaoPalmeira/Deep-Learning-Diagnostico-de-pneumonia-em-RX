import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras import optimizers
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
K.set_image_dim_ordering('th') #pode ser 'th' ou 'tf'
from sklearn.utils import class_weight

# fixar random seed para se puder reproduzir os resultados
seed = 9
np.random.seed(seed)


import os


def get_weight(y):
    class_weight_current =  class_weight.compute_class_weight('balanced', np.unique(y), y)
    return class_weight_current

'''util para visulaização do historial de aprendizagem'''
def print_history_accuracy(history):
    print(history.history.keys())
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

def print_model(model,fich):
    from keras.utils import plot_model
    plot_model(model, to_file=fich, show_shapes=True, show_layer_names=True)


'''Leitura do dataset'''
'''AS IMAGENS FORAM COLOCADAS NAS PASTAS TRAIN E TEST
    CRIOU-SE A PASTA DE VALIDAÇÃO POIS PERMITE EVITAR O OVERFITTING DA REDE 
    E FAZER COM QUE CONVERJA MAIS RAPIDO
'''

def read_dataset(path):
    paths,dirs,filesNormal = next(os.walk(path+"/train/NORMAL"))
    paths,dirs,filesPneumonia = next(os.walk(path+"/train/PNEUMONIA"))
    #Lê os nomes dos ficheiros e concatena num array 
    files = np.concatenate((filesNormal, filesPneumonia), axis=0)
    numFilesTreino = len(files)
    
    paths,dirs,filesNormal = next(os.walk(path+"/test/NORMAL"))
    paths,dirs,filesPneumonia = next(os.walk(path+"/test/PNEUMONIA"))
    #Lê os nomes dos ficheiros e concatena num array 
    files = np.concatenate((filesNormal, filesPneumonia), axis=0)   
    numFilesTeste = len(files)
    
    paths,dirs,filesNormal = next(os.walk(path+"/validation/NORMAL"))
    paths,dirs,filesPneumonia = next(os.walk(path+"/validation/PNEUMONIA"))
    #Lê os nomes dos ficheiros e concatena num array 
    files = np.concatenate((filesNormal, filesPneumonia), axis=0)
    numFilesValidacao = len(files)

    return (numFilesTreino,numFilesTeste,numFilesValidacao)


 #imprime um grafico com os valores de teste e com as correspondentes tabela de previsões
def print_series_prediction(y_test,predic):
    diff=[]
    racio=[]
    for i in range(len(y_test)): #para imprimir tabela de previsoes
        racio.append( (y_test[i]/predic[i])-1)
    diff.append( abs(y_test[i]- predic[i]))
    print('valor: %f ---> Previsão: %f Diff: %f Racio: %f' % (y_test[i],predic[i], diff[i],
    racio[i]))
    plt.plot(y_test,color='blue', label='y_test')
    plt.plot(predic,color='red', label='prediction') #este deu uma linha em branco
    plt.plot(diff,color='green', label='diff')
    plt.plot(racio,color='yellow', label='racio')
    plt.legend(loc='upper left')
    plt.show()


'''Etapa 2 - Definir a topologia da rede (arquitectura do modelo) e compilar'''
def create_model():    
    
    model = Sequential()
    model.add(Conv2D(16, (3, 3), activation='relu', padding="same", input_shape=(3,150,150)))
    model.add(Conv2D(16, (3, 3), padding="same", activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(32, (3, 3), activation='relu', padding="same", input_shape=(3,150,150)))
    model.add(Conv2D(32, (3, 3), padding="same", activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (3, 3), activation='relu', padding="same"))
    model.add(Conv2D(64, (3, 3), padding="same", activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(96, (3, 3), dilation_rate=(2, 2), activation='relu', padding="same"))
    model.add(Conv2D(96, (3, 3), padding="valid", activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(128, (3, 3), dilation_rate=(2, 2), activation='relu', padding="same"))
    model.add(Conv2D(128, (3, 3), padding="valid", activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())

    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.4))
    model.add(Dense(2 , activation='softmax'))


    print(model.summary())
    
    return model



'''Ciclo da rede'''
def cycle():

    (num_casos_treino, num_casos_teste, num_casos_validacao) = read_dataset("RX_torax")
    batch_size = 163
    
    '''NOVOOOOOOOOOO'''
    model = create_model()
    optimizer=optimizers.Adam()
    loss='categorical_crossentropy'
    metrics=['accuracy']
    epochs = 40
    print_model(model,"model.png")

    model.compile(optimizer, loss=loss, metrics=metrics)
    rescale = 1./255
    target_size = (150, 150)
    batch_size = 34
    class_mode = "categorical"


    train_datagen = ImageDataGenerator(
        rescale=rescale,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)


    train_generator = train_datagen.flow_from_directory(
        'RX_torax/train',
        target_size=target_size,
        class_mode=class_mode,
        batch_size=batch_size,
        shuffle=True)


    validation_datagen = ImageDataGenerator(rescale=rescale)

    validation_generator = validation_datagen.flow_from_directory(
        'RX_torax/validation',
        target_size=target_size,
        class_mode=class_mode,
        batch_size=34,
        shuffle = False)


    test_datagen = ImageDataGenerator(rescale=rescale)

    test_generator = test_datagen.flow_from_directory(
            'RX_torax/test',
            target_size=target_size,
            class_mode=class_mode,
            batch_size=34,
            shuffle = False)
    class_weight = get_weight(train_generator.classes)
    
    history = model.fit_generator(
            train_generator,
            steps_per_epoch = num_casos_treino//batch_size,
            epochs=epochs,
            verbose=2,
            validation_data=validation_generator,
            validation_steps=num_casos_validacao//batch_size, 
            class_weight=class_weight)
    
    
    print_history_accuracy(history)
    
    print("results")
    result  = model.evaluate_generator(test_generator, steps=len(test_generator), verbose=1)

    print("%s%.2f  "% ("Loss     : ", result[0]))
    print("%s%.2f%s"% ("Accuracy : ", result[1]*100, "%"))


if __name__ == '__main__':
    cycle()