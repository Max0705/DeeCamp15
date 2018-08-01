import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.image as mpimg
import random

y_size = 1
img_height = 128
img_width = 128
img_size = img_height*img_width
img_cond_size = img_size+y_size
z_size = 256
z_cond_size = z_size+y_size
h1_size = 1024
h2_size = 1024
batch_size = 128
keep_prob = 0.5
EPOCH = 1000000

def halfSize(tmp):
    item = []
    for i in range(128):
        for j in range(128):
            item.append((tmp[i*2][j*2]+tmp[i*2+1][j*2]+tmp[i*2][j*2+1]+tmp[i*2+1][j*2+1])/4)
    return np.array(item)

lab = []
for i in range(y_size):
    lab.append([])
    for j in range(y_size):
        if i==j:
            lab[i].append(1.0)
        else:
            lab[i].append(0.0)

fileList = ['aircraft carrier','dog']
train = [[],0]
for i in range(y_size):
    for j in range(10000):
        try:
            img = mpimg.imread('./'+fileList[i]+'/%d.jpg'%(j))
            tmp = img/255.0
            item = halfSize(tmp)
            train[0].append([item,lab[i]])
        except:
            pass
        if j%1000==0:
            print('%d/%d: %d/10000 images have been read.'%(i+1,y_size,j))
random.shuffle(train[0])

'''
t = train[0][1][0].reshape([128,128])
plt.imshow(t) # 显示图片
plt.axis('off') # 不显示坐标轴
plt.show()
'''
x = tf.placeholder(tf.float32,shape=[None,img_size])
y = tf.placeholder(tf.float32,shape=[None,y_size])
z = tf.placeholder(tf.float32,shape=[None,z_size])

def xavier_init(shape):
    in_dim = shape[0]
    stddev = 1.0/tf.sqrt(in_dim/2.0)
    return tf.random_normal(shape=shape,stddev=stddev)

def get_z(shape):
    return np.random.uniform(-1.0,1.0,size=shape).astype(np.float32)

def generator(z,y):
    z_cond = tf.concat([z,y],axis=1)
    #L1
    w1 = tf.Variable(xavier_init([z_cond_size,h1_size]))
    b1 = tf.Variable(tf.zeros([h1_size]),dtype=tf.float32)
    h1 = tf.nn.relu(tf.matmul(z_cond,w1)+b1)
    #L2
    w2 = tf.Variable(xavier_init([h1_size,h2_size]))
    b2 = tf.Variable(tf.zeros([h2_size]),dtype=tf.float32)
    h2 = tf.nn.relu(tf.matmul(h1,w2)+b2)
    #OUT
    w3 = tf.Variable(xavier_init([h2_size,img_size]))
    b3 = tf.Variable(tf.zeros([img_size]),dtype=tf.float32)
    x_generated = tf.nn.sigmoid(tf.matmul(h2,w3)+b3)
    
    params = [w1,b1,w2,b2,w3,b3]
    return x_generated, params

def discriminator(x,x_generated,keep_prob,y):
    x_cond = tf.concat([x,y],axis=1)
    x_generated_cond = tf.concat([x_generated,y],axis=1)
    #L1
    w1 = tf.Variable(xavier_init([img_cond_size,h1_size]))
    b1 = tf.Variable(tf.zeros([h1_size]),dtype=tf.float32)
    h1_x = tf.nn.dropout(tf.nn.relu(tf.matmul(x_cond,w1)+b1),keep_prob)
    h1_x_generated = tf.nn.dropout(tf.nn.relu(tf.matmul(x_generated_cond,w1)+b1),keep_prob)
    #L2
    w2 = tf.Variable(xavier_init([h1_size,h2_size]))
    b2 = tf.Variable(tf.zeros([h2_size]),dtype=tf.float32)
    h2_x = tf.nn.relu(tf.matmul(h1_x,w2)+b2)
    h2_x_generated = tf.nn.relu(tf.matmul(h1_x_generated,w2)+b2)
    #OUT
    w3 = tf.Variable(xavier_init([h2_size,1]))
    b3 = tf.Variable(tf.zeros([1]),dtype=tf.float32)
    d_prob_x = tf.nn.sigmoid(tf.matmul(h2_x,w3)+b3)
    d_prob_x_generated = tf.nn.sigmoid(tf.matmul(h2_x_generated,w3)+b3)

    params = [w1,b1,w2,b2,w3,b3]
    return d_prob_x,d_prob_x_generated,params

def next_batch(L,size):
    if L[1]+size<=len(L[0]):
        res = L[0][L[1]:L[1]+size]
    else:
        res = L[0][L[1]:]+L[0][0:size-len(L[0])+L[1]]
    L[1] = (L[1]+size)%len(L[0])
    tx = []
    ty = []
    for i in res:
        tx.append(i[0])
        ty.append(i[1])
    return tx,ty

def save(samples,index,shape):
    x,y = shape
    fig = plt.figure(figsize=[y,x])
    gs = gridspec.GridSpec(x,y)
    gs.update(wspace=0.05,hspace=0.05)
    for i,sample in enumerate(samples):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(sample.reshape(img_height,img_width),cmap='Greys_r')
    plt.savefig('./img/{}.png'.format(str(index).zfill(3)))
    plt.close(fig)

x_generated,g_params = generator(z,y)
d_prob_real,d_prob_fake,d_params = discriminator(x,x_generated,keep_prob,y)

g_loss = -tf.reduce_mean(tf.log(d_prob_fake+1e-30))
d_loss = -tf.reduce_mean(tf.log(d_prob_real+1e-30)+tf.log(1.0-d_prob_fake+1e-30))

g_solver = tf.train.AdamOptimizer(0.001).minimize(g_loss,var_list=g_params)
d_solver = tf.train.AdamOptimizer(0.001).minimize(d_loss,var_list=d_params)

sess =tf.Session()
sess.run(tf.global_variables_initializer())

for i in range(EPOCH):
    #save image
    if i%100==0:
        labels = [i for i in range(y_size) for _ in range(10)]  # 我要让他生成的数字，每行相同，每列从0到1递增
        cond_y = sess.run(tf.one_hot(np.array(labels),depth=y_size))  # 喂的字典不能是tensor，我run成np array
        samples = sess.run(x_generated, feed_dict = {z:get_z([y_size*10,z_size]),y:cond_y})
        index = int(i/100)  # 用来当保存图片的名字
        shape = [y_size,10]  # 维度和labels的宽高匹配
        save(samples, index, shape)  # 保存图片
    #train
    x_mb,y_mb = next_batch(train,batch_size)
    _,d_loss_curr = sess.run([d_solver,d_loss],feed_dict={x:x_mb,z:get_z([batch_size,z_size]),y:y_mb})
    _,g_loss_curr = sess.run([g_solver,g_loss],feed_dict={z:get_z([batch_size,z_size]),y:y_mb})
    if i%100==0:
        print('iter: %d, d_loss: %.3f, g_loss: %.3f\n' % (i,d_loss_curr,g_loss_curr))
