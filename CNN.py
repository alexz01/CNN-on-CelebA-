import tensorflow as tf
from LoadIMG import LoadCelebA

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

class CNN:
    img_dimx = 180
    img_dimy = 220
    def __init__(self,lrate=0.01, scale = 100):
        self.img_dimx = int(self.img_dimx *(scale/100))
        self.img_dimy = int(self.img_dimy *(scale/100))
        
        self.celeb = LoadCelebA()
        self.validationSet = self.celeb.validationSet()
        #self.mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

        self.x = tf.placeholder(tf.float32, shape=[None, 28*4*28*4])
        self.y_ = tf.placeholder(tf.float32, shape=[None, 2])

        self.W_conv1 = weight_variable([5, 5, 1, 32])
        self.b_conv1 = bias_variable([32])
        #print('size=',self.img_dimx*self.img_dimy)
        self.x_image = tf.reshape(self.x, [-1, 28*4, 28*4, 1])
        self.h_conv1 = tf.nn.relu(conv2d(self.x_image, self.W_conv1) + self.b_conv1)
        self.h_pool1 = max_pool_2x2(self.h_conv1)

        self.W_conv2 = weight_variable([5, 5, 32, 64])
        self.b_conv2 = bias_variable([64])
        self.h_conv2 = tf.nn.relu(conv2d(self.h_pool1, self.W_conv2) + self.b_conv2)
        self.h_pool2 = max_pool_2x2(self.h_conv2)

        self.W_fc1 = weight_variable([28*28*64, 1024])
        self.b_fc1 = bias_variable([1024])
        self.h_pool2_flat = tf.reshape(self.h_pool2, [-1, 28*28*64])
        self.h_fc1 = tf.nn.relu(tf.matmul(self.h_pool2_flat, self.W_fc1) + self.b_fc1)

        self.keep_prob = tf.placeholder(tf.float32)
        self.h_fc1_drop = tf.nn.dropout(self.h_fc1, self.keep_prob)

        self.W_fc2 = weight_variable([1024, 2])
        self.b_fc2 = bias_variable([2])
        self.y_conv = tf.matmul(self.h_fc1_drop, self.W_fc2) + self.b_fc2

        self.cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.y_, logits=self.y_conv))
        self.train_step = tf.train.AdamOptimizer(lrate).minimize(self.cross_entropy)
        self.correct_prediction = tf.equal(tf.argmax(self.y_conv, 1), tf.argmax(self.y_, 1))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))
        

    def train(self,iterations = 1500,batch_size = 100):
        with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)) as sess:
            sess.run(tf.global_variables_initializer())
            batchsize = tf.shape(self.x)[0]
            print('batchsize: ', sess.run(batchsize, {self.x:self.validationSet[0]}))
            for i in range(iterations):
                batch = self.celeb.next_batch(batch_size)
                #print(batch[0].T[0].size,batch[1].size )
                
                self.train_step.run(feed_dict={self.x: batch[0], self.y_: batch[1], self.keep_prob:0.5})
                if i % 100 == 0:
                    train_accuracy = self.accuracy.eval(feed_dict={self.x: self.validationSet[0], self.y_: self.validationSet[1], self.keep_prob: 1.0})
                    print('step %d, training accuracy %g' % (i, train_accuracy))
            
    
            
            x_celeb, y_celeb = self.celeb.testSet()
            print('accuracy of CelebA recognition: ', self.accuracy.eval(feed_dict={self.x: x_celeb, self.y_: y_celeb, self.keep_prob: 1.0}))
