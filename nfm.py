#my implementation of nfm 
import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_svmlight_file

def full_layer(input, size):
	in_size = int(input.get_shape()[1])
	W = weight_variable([in_size, size])
	b = bias_variable([size])
	return tf.matmul(input, W) + b
	
learning_rate = 0.1
training_epoches = 500
batch_size =100
display_step = 1

X_total,Y_total=load_svmlight_file("libsvmfinal_8_2")
shape=X_total.shape

Y_total=Y_total.reshape(-1, 1)
X_train, X_test, y_train, y_test = train_test_split(X_total, Y_total, test_size = 0.1, random_state = 42)

n_samples = X_train.shape[0]
n_features =shape[1]
k=2

#placeholder
#x_indices=tf.placeholder(tf.int64,[None,2])
#x_values=tf.placeholder(tf.float32,[None])
#x_shape=tf.placeholder(tf.int64,[2])
x = tf.sparse_placeholder(tf.float64)
y = tf.placeholder(tf.float64, [None, 1])

#parametor to train
V = tf.Variable(tf.zeros([n_features, k],dtype=tf.float64),dtype=tf.float64,name="vk")
b = tf.Variable(tf.zeros([1],dtype=tf.float64),dtype=tf.float64,name="b")
w = tf.Variable(tf.random_normal((n_features,1),dtype=tf.float64),dtype=tf.float64,name="w")
#model logic
#x=tf.SparseTensor(x_indices,x_values,x_shape)
vx=tf.sparse_tensor_dense_matmul(x,V)
vx_sq=tf.multiply(vx,vx)
xx=tf.square(x)
vsq_xsq=tf.sparse_tensor_dense_matmul(xx,V*V)
biterm=vx_sq-vsq_xsq
# here i just add one k to 1 full_layer,you can add as many layers as you like, 
# the only difference with my fm implementation is this line of code 
preds=tf.nn.sigmoid(tf.sparse_tensor_dense_matmul(x,w)+0.5*full_layer(biterm,1)+b)
cost=tf.reduce_mean(-y*tf.log(tf.clip_by_value(preds,1e-10,1.0))-(1-y)*tf.log(tf.clip_by_value(1-preds,1e-10,1.0)))
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)
threshold=tf.constant(0.5,dtype=tf.float64)
plabel=tf.cast(threshold<y,tf.float64)
accuracy=tf.metrics.accuracy(y,plabel)
#auc=tf.metrics.auc(y,preds)
#accuracy=tf.reduce_mean(tf.cast(tf.equal(plabel,y),tf.float32))
init = tf.global_variables_initializer()
saver=tf.train.Saver()

with tf.Session() as sess:
    #sess.run(init)
    sess.run(tf.initialize_local_variables())
    saver.restore(sess,"my_model3.ckpt198")
    for epoch in range(training_epoches):
        avg_cost = 0
        total_batch = int(n_samples / batch_size)
        batch=0
        for i in range(total_batch):
            xcsr=X_train[i*batch_size:(i+1)*batch_size]
            coo=xcsr.tocoo() 
            indices=np.mat([coo.row,coo.col]).transpose()           
            _, c = sess.run([optimizer, cost], 
                            feed_dict={x:tf.SparseTensorValue(indices,coo.data,coo.shape) 
                                     ,y: y_train[i * batch_size : (i+1) * batch_size]})
            avg_cost += c / total_batch
            batch +=1
            if batch%2000 ==1999:
                print "epoch",epoch,"batch",batch
        print "b", b.eval()
        if epoch%100 ==99:
            save_path=saver.save(sess,"./my_model3.ckpt"+str(epoch))
        if (epoch+1) % display_step == 0:

            print("Epoch:", "%04d" % (epoch+1), "cost=", avg_cost)          
            xcsr=X_test
            coo=xcsr.tocoo() 
            indices=np.mat([coo.row,coo.col]).transpose()
            act=sess.run([accuracy],feed_dict={x:tf.SparseTensorValue(indices,coo.data,coo.shape),y: y_test})
            print "accuracy" ,act
            
    print("Optimization Finished!")
        
            
            



    
