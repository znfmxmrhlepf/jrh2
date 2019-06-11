import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tools
tf.enable_eager_execution()

model = tools.get_model2()

x_train, y_train, x_test, y_test = tools.load_data()

train_dataset = tf.data.Dataset.from_tensor_slices(
        (tf.cast(x_train/255, tf.float32), tf.cast(y_train, tf.int64)))
train_dataset = train_dataset.shuffle(27000).repeat().batch(30)

test_dataset = tf.data.Dataset.from_tensor_slices(
        (tf.cast(x_test/255, tf.float32), tf.cast(y_test, tf.int64)))
test_dataset = test_dataset.shuffle(27000).repeat().batch(30)


optimizer = tf.train.AdamOptimizer(learning_rate=0.0001)
loss_history = []
acc_history = []

for batch, (images, labels) in enumerate(train_dataset.take(9000)):
    if (batch+1) % 10 == 0:
        print(batch+1)
   
    if (batch) % 100 == 0:
        for (test_img, test_label) in test_dataset.take(1):
            predicts = model(tf.cast(test_img/255, tf.float32)).numpy()
            print(predicts)
            predicts = np.argmax(predicts, axis=1)
            print(predicts)
            print(test_label)
            acc = np.average(np.equal(test_label, predicts)) 
            print('acc : '+str(acc))
            acc_history.append(acc)

    with tf.GradientTape() as tape:
        logits = model(images, training=True)
        loss_value = tf.losses.sparse_softmax_cross_entropy(labels, logits)

    loss_history.append(loss_value.numpy())
    grads = tape.gradient(loss_value, model.trainable_variables)
    optimizer.apply_gradients(
            zip(grads, model.trainable_variables),
            global_step=tf.train.get_or_create_global_step()            
            )

plt.plot(loss_history)
plt.show()
plt.plot(acc_history)
plt.show()
