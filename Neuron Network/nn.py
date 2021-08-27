# Import các thư viện cần thiết
from __future__ import print_function
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# Import MNIST data
# TF có hỗ trợ giúp chúng ta đọc dữ liệu mnist
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)
import tensorflow as tf

# Khai báo các tham số của mô hình
learning_rate = 0.1 # tốc độ học
num_steps = 500 # tổng số lần học/huấn luyện
batch_size = 128 # Số điểm dữ liệu đưa vào học mỗi lần huấn luyến
display_step = 100 # Cứ sau 100 lần học, hiện thị các thay đổi thông số của mô hình
 
# Các tham số của mạng
n_hidden_1 = 256 # 1st layer number of neurons - số nơ ron của layer 1
n_hidden_2 = 256 # 2nd layer number of neurons - số nơ ron của layer 2
input_shape = 784 # MNIST data input (img shape: 28*28) kích thước của 1 input(vector 784 chiều).
num_classes = 10 # MNIST total classes (0-9 digits) - label vector dạng one-hot

# tf Graph input
X = tf.placeholder("float", [None, input_shape])
Y = tf.placeholder("float", [None, num_classes])

# Khai báo weights và bias cho từng layer
weights = {
    'h1': tf.Variable(tf.random_normal([input_shape, n_hidden_1])),
    'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_hidden_2, num_classes]))
}
biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'out': tf.Variable(tf.random_normal([num_classes]))
}

tf.matmul(X, weights['h1'])

layer_1 = tf.add(tf.matmul(X, weights['h1']), biases['b1'])

layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])

# Tạo model nối các layer
def neural_net(x):
    # Hidden fully connected layer with 256 neurons
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    # Hidden fully connected layer with 256 neurons
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    # Output fully connected layer with a neuron for each class
    out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
    return out_layer

logits = neural_net(X)
 
# Khởi tạo loss và optimizer
loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
    logits=logits, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss_op)
 
# Đánh giá model
correct_pred = tf.equal(tf.argmax(logits, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
 
# ops khởi tạo các biến của graph
init = tf.global_variables_initializer()

# Tạo model nối các layer
def neural_net(x):
    # Hidden fully connected layer with 256 neurons
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    # Hidden fully connected layer with 256 neurons
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    # Output fully connected layer with a neuron for each class
    out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
    return out_layer

logits = [1,2,3,4,-1]
tf.argmax(logits, 1) = 4

x = [1, 2, 3]
y = [1, 3, 3]
tf.equal(x, y) = [True, False, True]

with tf.Session() as sess:
 
    # Chạy op khởi tạo
    sess.run(init)
 
    for step in range(1, num_steps+1):
        batch_x, batch_y = mnist.train.next_batch(batch_size)
        # Run optimization op (backprop)
        sess.run(train_op, feed_dict={X: batch_x, Y: batch_y})
        if step % display_step == 0 or step == 1:
            # Calculate batch loss and accuracy
            loss, acc = sess.run([loss_op, accuracy], feed_dict={X: batch_x,
                                                                 Y: batch_y})
            print("Step " + str(step) + ", Minibatch Loss= " + \
                  "{:.4f}".format(loss) + ", Training Accuracy= " + \
                  "{:.3f}".format(acc))
 
    print("Optimization Finished!")
 
    # Tính toán độ chính xác của model dựa trên tập MNIST test
    print("Testing Accuracy:", \
        sess.run(accuracy, feed_dict={X: mnist.test.images,
                                      Y: mnist.test.labels}))
    