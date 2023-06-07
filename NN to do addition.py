import numpy as np
import tensorflow as tf

X = np.random.randint(0,10,(100000,2))
y = np.sum(X,axis=1,keepdims=True).flatten()

model = tf.keras.models.Sequential([tf.keras.layers.Flatten(input_shape=(2,1)),
                                    tf.keras.layers.Dense(1024,activation=tf.nn.sigmoid),
                                    tf.keras.layers.Dense(512,activation=tf.nn.sigmoid),
                                    tf.keras.layers.Dense(128,activation=tf.nn.sigmoid),
                                    tf.keras.layers.Dense(19,activation=tf.nn.softmax),
                                    ])


model.compile(optimizer=tf.optimizers.SGD(learning_rate=0.01),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(X,y,epochs=100)
