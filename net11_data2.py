from util import gen_map,kappa
import gendata as input_data
import tensorflow as tf
import numpy
from sklearn.metrics import accuracy_score,classification_report
import model
from scipy.spatial.distance import cdist
import numpy as np

data_type = 0
data_arguement = False
balance = True
opt = input_data.OPT(data_type)
dir1 = opt.data_dir
label_dir = opt.label_dir
mymnist = input_data.Gen_data(dir1,label_dir,balance)
# numpy.set_printoptions(precision=4)

# 定义网络超参数
learning_rate = 0.001
training_iters = 800000//8
batch_size = 16
test_batch_size = 10000
display_step = 100
display_plot = 20
display_test_info = 20
topk = 5
cate = [['Rapeseed', 'Bare soil', 'Potatoes', 'Beet', 'Wheat', 'Peas', 'Barely', 'Lucerne'],  # Flevoland
        ]
K = 5   # 样本组的大小？
cata =cate[data_type]
# ------------------------------------------
# 改为  -------------------------------------
score=[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]
best = score[data_type]
# result_dir = ''
dropout = 0.5
# 定义网络参数
n_input = mymnist.samwin
n_channl = mymnist.chanel
n_classes = mymnist.nclass
iheight = mymnist.height
iweight = mymnist.weight
iwindow = n_input
# 占位符输入
x = tf.placeholder(tf.float32, [None, n_input, n_input, n_channl])
x1 = tf.placeholder(tf.float32, [None, n_input, n_input, n_channl])
y = tf.placeholder(tf.float32, [None, n_classes])
y_ = tf.placeholder(tf.float32, [None])

keep_prob = tf.placeholder(tf.float32)


#                base bet
# 构建模型
build = model.Model(x, x1, y_,True,n_classes,keep_prob)
pred, feature = build.model
# 定义损失函数和学习步骤
prob = tf.argmax(pred,1)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred,labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# 测试网络
correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
# confuseMatrix=tf.confusion_matrix()

#                 sem bet
cost1 = build.loss
learning_rate1 = 0.001  # 0.0005
optimizer1 = tf.train.AdamOptimizer(learning_rate=learning_rate1).minimize(cost1)
correct_pred1 = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
accuracy1 =tf.reduce_mean(tf.cast(correct_pred1, tf.float32))

# 初始化所有的共享变量
init = tf.global_variables_initializer()

# start_time=time.time()
# 开启一个训练
with tf.Session() as sess:
    sess.run(init)
    #some paraments
    step = 1
    dls = 0

    best_confuse = 0

    # Keep training until reach max iterations
    while step * batch_size < training_iters:
        batch_xs, batch_ys = mymnist.next_batch(batch_size,data_arguement)
        # 获取批数据
        sess.run(optimizer, feed_dict={x: batch_xs, y: batch_ys,keep_prob: dropout})
        if step % display_step == 0:
            # 计算精度
            acc = sess.run(accuracy, feed_dict={x: batch_xs, y: batch_ys, keep_prob:1.0})
            loss = sess.run(cost, feed_dict={x: batch_xs, y: batch_ys, keep_prob: 1.0})
            # 计算损失值
            print("Iter " + str(step*batch_size) + ",Loss= " + "{:.6f}".format(loss) + ", Acc= " + "{:.5f}".format(acc))

        step += 1

        if mymnist._epochs_completed !=0 and mymnist._epochs_completed % display_test_info == 0:

            dls+=1
            mymnist._epochs_completed=0
            show_flag = False

    print("Optimization Finished!")
    print(" start sime net work...")

    step = 1
    dls = 0
    training_iters1 = 2000000
    batch_size = 16
    mymnist._index_in_epoch = 0

    while step * batch_size < training_iters1:

        batch_xs1, batch_ys1 = mymnist.next_batch2(batch_size,data_arguement)
        batch_xs2, batch_ys2 = mymnist.next_batch2(batch_size,data_arguement)
        batch_y_ = ((np.argmax(batch_ys1, axis=1)) == (np.argmax(batch_ys2,axis=1))).astype(numpy.float32)
        batch_y_ = batch_y_[range(0,batch_size*K,K)]

        # 获取批数据
        sess.run(optimizer1, feed_dict={x: batch_xs1, x1: batch_xs2, y_: batch_y_, keep_prob: dropout})
        if step % display_step == 0:
            # 计算精度
            # acc = sess.run(accuracy1, feed_dict={x: batch_xs, y: batch_ys, keep_prob:1.0})
            acc = mymnist._epochs_completed
            loss1 = sess.run(cost1, feed_dict={x: batch_xs1, x1: batch_xs2, y_: batch_y_, keep_prob: 1.0})
            # 计算损失值
            print("Iter " + str(step*batch_size) + ",Loss= " + "{:.6f}".format(loss1) + ", Acc= " + "{:.5f}".format(acc))

        step += 1
        display_test_info1 = display_test_info*10
        if mymnist._epochs_completed !=0 and mymnist._epochs_completed% display_test_info1 == 0:

            dls+=1
            mymnist._epochs_completed=0
            show_flag = True


    count = 0
    bz= test_batch_size
    vprob = numpy.array([])

    sup = sess.run(feature, feed_dict={x: mymnist._images, keep_prob: 1.0})
    tlabel = np.argmax(mymnist._labels, axis=1)
    for i in range(mymnist.gettestnum()//bz+1):
        embed = sess.run(feature, feed_dict={x: mymnist.next_test_batch(bz), keep_prob: 1.0})

        sim = cdist(embed, sup, metric="euclidean")
        ranks = numpy.argsort(sim, axis=1)[:,0:topk]
        vprobtmp = np.array([np.argmax(np.bincount(tlabel[ranks1])) for ranks1 in ranks])
        vprob = numpy.append(vprob, vprobtmp)

    y_pred, y_true,irow,icol = gen_map(vprob, iheight, iweight, iwindow, 1, mymnist.test_label,n_classes ,show_flag=show_flag)

    yt= y_true[y_true!=0]
    yp= y_pred[y_true!=0]+1

    test_acc = accuracy_score(yt,yp)
    print('test acc:',test_acc)
    kappa = kappa(yt, yp,1,n_classes)
    print('kappa:',kappa)
    # print(classification_report(yt, yp, target_names=cata, digits=4))
      
    print("Optimization Finished!")






