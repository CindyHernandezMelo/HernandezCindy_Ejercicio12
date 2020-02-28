import matplotlib.pyplot as plt
import sklearn.datasets as skdata
import numpy as np
import sklearn.metrics
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import precision_recall_curve

# lee los numeros
numeros = skdata.load_digits()

# lee los labels
target = numeros['target']

# lee las imagenes
imagenes = numeros['images']

# cuenta el numero de imagenes total
n_imagenes = len(target)

# para poder correr PCA debemos "aplanar las imagenes"
data = imagenes.reshape((n_imagenes, -1)) # para volver a tener los datos como imagen basta hacer data.reshape((n_imagenes, 8, 8))

# Split en train/test
x_train, x_test, y_train, y_test = train_test_split(data, target, train_size=0.5)

# todo lo que es diferente de 1 queda marcado como 0
y_train[y_train!=1]=0
y_test[y_test!=1]=0


# Reescalado de los datos
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)


# Encuentro los autovalores y autovectores de las imagenes marcadas como 1.
numero = 1
dd3 = np.zeros([y_train.shape[0], 3], dtype=bool)
dd3[:,0] = y_train==numero
dd3[:,1] = y_train!=numero
dd3[:,2] = np.ones(y_train.shape[0])==1
labels = ['Sobre 1','Sobre 0','Sobre todo']

plt.figure()

for i in range(3):
    dd = dd3[:,i]
    cov = np.cov(x_train[dd].T)
    valores, vectores = np.linalg.eig(cov)
    
    # pueden ser complejos por baja precision numerica, asi que los paso a reales
    valores = np.real(valores)
    vectores = np.real(vectores)
    
    # reordeno de mayor a menor
    ii = np.argsort(-valores)
    valores = valores[ii]
    vectores = vectores[:,ii]
    
    # encuentro las imagenes en el espacio de los autovectores
    x_test_transform = x_test @ vectores[:,:10]
    x_train_transform = x_train @ vectores[:,:10]
    
    # inicializo el clasificador
    linear = LinearDiscriminantAnalysis()
    
    linear = LinearDiscriminantAnalysis()
    linear.fit(x_train_transform, y_train)
    y_score = linear.predict_proba(x_test_transform)
    
    precision, recall, thresholds  = precision_recall_curve(y_test,y_score[:,1])
    
    punto = np.argmax(( 2 * (precision * recall) / (precision + recall))[:-1])
    
    
    plt.subplot(121)
    plt.plot(precision, recall, label = labels[i])
    plt.scatter(precision[punto],recall[punto], c = 'r')
    plt.legend()
    plt.xlabel('Precision')
    plt.ylabel('Cobertura')
    
    plt.subplot(122)
    plt.plot(thresholds,( 2 * (precision * recall) / (precision + recall))[:-1])
    plt.scatter(thresholds[punto],( 2 * (precision * recall) / (precision + recall))[punto], c = 'r')
    
    plt.xlabel('Probabilidad')
    plt.ylabel('F1 score')

plt.savefig('F1_prec_recall.png')