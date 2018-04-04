import tensorflow as tf
# 定义CNN网络实现的类
class TextCNN(object):
    '''
    CNN用于文本分类，使用嵌入层，然后是卷积，最大池和softmax层。
    '''
    def __init__(self, sequence_length, num_classes, embedding_size, filter_sizes, 
                 num_filters, l2_reg_lambda=0.0):  
        # Placeholders for input, output, dropout
        # input_x输入语料,待训练的内容,维度是[batch_size,sequence_length,embedding_size]
        self.input_x = tf.placeholder(tf.float32, [None, sequence_length, embedding_size],
                                                                             name = "input_x")  
        # input_y输入语料,待训练的内容标签,维度是num_classes,"体育||军事||经济||艺术||计算机"
        self.input_y = tf.placeholder(tf.float32, [None, num_classes], name = "input_y")
        # dropout_keep_prob dropout参数,防止过拟合,训练时用
        self.dropout_keep_prob = tf.placeholder(tf.float32, name = "dropout_keep_prob")
	    
	   # Keeping track of l2 regularization loss (optional)
        l2_loss = tf.constant(0.0)   # 先不用，写0
        self.embedded_chars_expended = tf.expand_dims(self.input_x, -1)
	   # Create a convolution + maxpool layer for each filter size
        pooled_outputs = []
        #每一层卷积层滤波器的尺寸分别为3,4,5
        #卷积->池化->卷积->池化->卷积->池化
        #第一层卷积层的变量并实现前向传播过程。输入为[batch_size,sequence_length,embedding_size,1]
        #的矩阵，
        #输出为((sequence_length-3)/1+1)*((embedding_size-embedding_size)/1+1)*128的矩阵。
        #第二层池化层的变量并实现前向传播过程。输入为((sequence_length-3)/1+1)*((embedding_size-3)/1+1)*128的矩阵，
        #输出为1*1*128的矩阵。
        #第三层卷积层的变量并实现前向传播过程。输入为(1)*(1)*128的矩阵，
        #输出为1*((embedding_size-3)/1+1)*128的矩阵。
        for filter_size in filter_sizes:
            with tf.name_scope("conv-maxpool-%s" % filter_size):
	            # 卷积层
                 # 4个参数分别为filter_size高h，embedding_size宽w，channel为1，filter个数
                 filter_shape = [filter_size, embedding_size, 1, num_filters]
                 W = tf.Variable(tf.truncated_normal(filter_shape,stddev=0.1), 
                                 name="W")
                 b=tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")
                 conv = tf.nn.conv2d(self.embedded_chars_expended,W,strides=[1,1,1,1],
                                      padding="VALID",name="conv")  # 未使用全零填充
		            #可以理解为,正面或者负面评价有一些标志词汇,这些词汇概率被增强，即一旦出现这些
                 #词汇,倾向性分类进正或负面评价,
                 #该激励函数可加快学习进度，增加稀疏性,因为让确定的事情更确定,噪声的影响就降到了最低。
                 relu = tf.nn.relu(tf.nn.bias_add(conv, b), name = "relu")
		            #池化
                 #卷积卷的是句子时间那个维度，而第三维embedding是相互独立的，并不做卷积
                 pooled = tf.nn.max_pool(relu,ksize=[1, sequence_length - filter_size + 1, 1, 1],
                                         strides=[1,1,1,1],padding="VALID",name="pool")
                 
                 pooled_outputs.append(pooled)
	     #Combine all the pooled features
        num_filters_total = num_filters * len(filter_sizes)
        self.h_pool = tf.concat(pooled_outputs,3)
        self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])  # 扁平化数据，跟全连接层相连

	     #drop层,防止过拟合,参数为dropout_keep_prob
        # 过拟合的本质是采样失真,噪声权重影响了判断，如果采样足够多,足够充分,噪声的影响可以被量化到趋近事
        #实,也就无从过拟合。即数据越大,drop和正则化就越不需要。
        with tf.name_scope("dropout"):
	        self.h_drop = tf.nn.dropout(self.h_pool_flat, self.dropout_keep_prob)
	    
	     # Final (unnomalized) scores and predictions
        with tf.name_scope("output"):
             W = tf.get_variable("W", shape = [num_filters_total, num_classes],   #前面连扁平化后的池化操作
                                 initializer = tf.contrib.layers.xavier_initializer())  # 定义初始化方式
             b = tf.Variable(tf.constant(0.1,shape=[num_classes]),name='b')
             l2_loss += tf.nn.l2_loss(W)+tf.nn.l2_loss(b)      # 损失函数导入
             #self.scores的shape为batch_size*num_classes
             self.scores = tf.nn.xw_plus_b(self.h_drop, W, b, name = "scores")  # 得分函数
             #对预测结果按行求最大值的索引，其结果就是类别
             self.predictions = tf.argmax(self.scores, axis=1, name = "predictions")   # 预测结果
        # 计算平均交叉熵
        with tf.name_scope("loss"):
             losses = tf.nn.softmax_cross_entropy_with_logits(logits = self.scores, labels = self.input_y)
             self.loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss
        #准确率，求和计算算数平均值
        with tf.name_scope("accuracy"):
             correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
             self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name = "accuracy")

	         