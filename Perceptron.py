import tensorflow as tf
from tensorflow import keras

#загрузка данных для обучения (датасет)
(train_images, train_labels), (test_images, test_labels)=keras.datasets.mnist.load_data()

#нормализация значений датасета
train_images=train_images/255.0
test_images=test_images/255.0
train_labels=tf.one_hot(train_labels,10)
test_labels=tf.one_hot(test_labels,10)

#инициализация полносвязной нейронной сети
model=keras.models.Sequential([
    keras.layers.Flatten(input_shape=(28,28)),
    keras.layers.Dense(100, activation='relu'),
    keras.layers.Dense(10)
])
print(model.summary())

#инциализация функции ошибок (среднеквадратичная ошибка)
loss=keras.losses.MeanSquaredError()

#инициализация метода оптимизации (стохастический градиентный спуск)
opt=keras.optimizers.SGD(learning_rate=0.01)

#инициализация метрики качества обучения (доля верных распознований)
metrics=['accuracy']

#cборка модели (НС + метод оптимизации + функция ошибок + метрика)
model.compile(optimizer=opt, loss=loss, metrics=metrics)

#обучение нейронной сети
batch_size=10
epochs=30
model.fit(train_images, train_labels, epochs=epochs, batch_size=batch_size, verbose=2)

#оценка качества обучения
model.evaluate(test_images, test_labels, batch_size=batch_size, verbose=2)

input()
