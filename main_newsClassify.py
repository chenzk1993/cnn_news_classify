'''
此程序训练过程需要15分钟
训练命令：python main_newsClsaaify.py
测试命令：python main_newsClsaaify.py --mode=test
'''
import tensorflow as tf
import numpy as np
import os,time,math,sys
import datetime
import data_helpers
import word2vec_helpers

# Session configuration
os.environ['CUDA_VISIBLE_DEVICES'] = '0'  #只使用0号GPU
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  

from cnn_classify import TextCNN
def time_transform(s):
    ret = ''
    if s >= 60 * 60:
        h = math.floor(s / (60 * 60))
        ret += '{}h'.format(h)
        s -= h * 60 * 60
    if s >= 60:
        m = math.floor(s / 60)
        ret += '{}m'.format(m)
        s -= m * 60
    if s >= 1:
        s = math.floor(s)
        ret += '{}s'.format(s)
    return ret
# Parameters
# =======================================================
tf.flags.DEFINE_string('mode', 'train', 'train/test')
#语料文件路径定义
#tf.flags.DEFINE_float(flag_name, default_value, docstring)flag_name为字符型，表示该标志的名称；
#default_value为float型,表示默认值; docstring表示该条消息的用途等
tf.flags.DEFINE_float("test_sample_percentage", 0.1, "Percentage of the training data to use for validation")
#tf.flags.DEFINE_string(flag_name, default_value, docstring)flag_name为字符型，表示该标志的名称；
#default_value为string型,表示默认值; docstring表示该条消息的用途等
tf.flags.DEFINE_string("word_embedding_file", 'cnews/newsblog_575746w_200d.vec', "Data source for word embeddings file")
tf.flags.DEFINE_string("sports_file", "cnews/体育.txt", "Data source for 体育 data.")
tf.flags.DEFINE_string("amusement_file", "cnews/娱乐.txt", "Data source for 娱乐 data.")
tf.flags.DEFINE_string("home_file", "cnews/家居.txt", "Data source for 家居 data.")
tf.flags.DEFINE_string("estate_file", "cnews/房产.txt", "Data source for 房产 data.")
tf.flags.DEFINE_string("education_file", "cnews/教育.txt", "Data source for 教育 data.")
tf.flags.DEFINE_string("fashion_file", "cnews/时尚.txt", "Data source for 时尚 data.")
tf.flags.DEFINE_string("politics_file", "cnews/时政.txt", "Data source for 时政 data.")
tf.flags.DEFINE_string("game_file", "cnews/游戏.txt", "Data source for 游戏 data.")
tf.flags.DEFINE_string("technology_file", "cnews/科技.txt", "Data source for 科技 data.")
tf.flags.DEFINE_string("finance_file", "cnews/财经.txt", "Data source for 财经 data.")
tf.app.flags.DEFINE_string('checkpoint_dir','checkpoints', 'checkpoints save path.')

#tf.flags.DEFINE_string(flag_name, default_value, docstring)flag_name为字符型，表示该标志的名称；
#default_value为int型,表示默认值; docstring表示该条消息的用途等
tf.flags.DEFINE_integer("num_labels", 10, "Number of labels for data.")
tf.flags.DEFINE_integer("max_seq_length", 100, "Max number of words for a sentence.)")

# 定义网络超参数
tf.flags.DEFINE_integer("embedding_dim", 200, "Dimensionality of character embedding (default: 200)")
tf.flags.DEFINE_string("filter_sizes", "3,4,5", "Comma-spearated filter sizes (default: '3,4,5')")
tf.flags.DEFINE_integer("num_filters", 128, "Number of filters per filter size (default: 128)")
tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability (default: 0.5)")
tf.flags.DEFINE_float("l2_reg_lambda", 0.0, "L2 regularization lambda (default: 0.0)")

#训练参数
#每个数据集30个数据,就是一批数据
tf.flags.DEFINE_integer('num_epochs', 30, 'Number of training epochs (default: 30)')
tf.flags.DEFINE_integer("batch_size", 50, "Batch Size")
#tf.flags.DEFINE_boolean(flag_name, default_value, docstring)flag_name为字符型，表示该标志的名称；
#default_value为bool型,表示默认值; docstring表示该条消息的用途等

tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

#打印相关初始参数
FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
print("Parameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print('{}={}'.format(attr.upper(),value))  #将属性名大写

# Data preprocess
# =======================================================

# Load data
print("Loading data...")
#x_text为二维列表，第一维以每一句话为元素构成的列表，第二维以该句话的每个词组成的列表构成
#['全国', '少年儿童', '游泳', '锦标赛', '开幕', '新华社', '广州', '月', '日电', '记者', '何惠飞', '年', '喜乐', '杯', '全国', '少年儿童', '游泳', '锦标赛', '昨天', '在', '游泳', '之', '乡', '广东省', '东莞市', '开幕', '参加', '这次', '比赛', '的', '有', '个', '省', '自治区', '直辖市', '的', '名', '男女', '选手', '比赛', '分为', '岁', '组和岁', '以下', '组', '参赛者', '都', '是', '近几年', '涌现', '的', '优秀', '小', '选手', '不少', '是', '本', '年龄组', '的', '全国纪录', '创造者', '这次', '比赛', '是', '对', '我国', '参加', '下', '两届', '奥运会', '游泳赛', '后备力量', '的', '一次', '检阅', '国家体委', '将', '通过', '这次', '比赛', '选拔', '优秀', '选手', '组队参加', '今年', '月', '在', '印度尼西亚', '举行', '的', '亚太区', '年龄组', '游泳', '比赛', '比赛', '将', '于', '日', '结束', '完']
x_text, y = data_helpers.load_data_files(FLAGS.sports_file, FLAGS.amusement_file,FLAGS.home_file,
                                         FLAGS.estate_file,FLAGS.education_file,FLAGS.fashion_file,
                                         FLAGS.politics_file,FLAGS.game_file,FLAGS.technology_file,
                                         FLAGS.finance_file)

# Get embedding vector
sentences, max_document_length = data_helpers.padding_sentences(x_text, 'PADDING',FLAGS.max_seq_length)
#此时的sentences为每一句话中的词构成的列表为元素而构成的二维列表，其中每一句话的长度都相同，因为都用
#'PADDING'将其补充到最大长度
#将返回的列表转化为数组
x_embedding=word2vec_helpers.embedding_sentences(sentences, 
                                                 embedding_size = FLAGS.embedding_dim, 
                                                 ext_emb_path =FLAGS.word_embedding_file)
x = np.array(x_embedding)
#x的三维分别表示句子总数，每个句子中的单词数(以最长的句子计)，词向量的维数
print("x.shape =",x.shape)
print("y.shape =",y.shape)
#Save params
training_params_file = 'train/training_params.pickle'
params = {'num_labels' : FLAGS.num_labels, 'max_document_length' : max_document_length}
data_helpers.saveDict(params, training_params_file)

# 数据混杂
np.random.seed(10)
shuffle_indices = np.random.permutation(range(len(y)))
x_shuffled = x[shuffle_indices]
y_shuffled = y[shuffle_indices]

#将数据按训练train集和测试test集
test_sample_index = -1 * int(FLAGS.test_sample_percentage * float(len(y)))
x_train, x_test = x_shuffled[:test_sample_index], x_shuffled[test_sample_index:]
y_train, y_test = y_shuffled[:test_sample_index], y_shuffled[test_sample_index:]
#输出训练样本数和测试样本数
print("Train/test split: %d/%d" %(len(y_train), len(y_test)))

#检查点目录。 
##如果该文件不存在，则重新创建
if not os.path.exists(FLAGS.checkpoint_dir):
    os.makedirs(FLAGS.checkpoint_dir)

# Training
# =======================================================

#tf.ConfigProto(log_device_placement=True,allow_soft_placement=True)allow_soft_placement表示
#如果你指定的设备不存在，是否允许TF自动分配设备;log_device_placement表示是否打印设备分配日志
session_conf = tf.ConfigProto(allow_soft_placement = FLAGS.allow_soft_placement,
	                                   log_device_placement = FLAGS.log_device_placement)
session_conf.gpu_options.allow_growth = True  #显存分配器将不会指定所有的GPU内存，而是按照需求增长
session_conf.gpu_options.per_process_gpu_memory_fraction = 0.4   # 占用GPU40%的显存 
with tf.Session(config = session_conf) as sess:
     #卷积池化网络导入
     cnn = TextCNN(
             sequence_length = x.shape[1], #句子最大的长度
             num_classes = y_train.shape[1],     #表示分几类
             embedding_size = FLAGS.embedding_dim,  #词向量的维数
             #将上面定义的filter_sizes拿过来，"3,4,5"按","分割，组成一个列表
             filter_sizes = list(map(int, FLAGS.filter_sizes.split(","))), 
             num_filters=FLAGS.num_filters,  #filter数目即输出卷积层的深度
             l2_reg_lambda = FLAGS.l2_reg_lambda  #l2_reg_lambda表示l2正则化项
             )  

     #Define Training procedure
     global_step = tf.Variable(0, name="global_step", trainable=False)
     optimizer = tf.train.AdamOptimizer(1e-3)     # 定义优化器
#.compute_gradients(loss, var_list=None,gate_gradients=GATE_OP,aggregation_method=None,
#colocate_gradients_with_ops=False,grad_loss=None)loss表示包含值最小化的张量;var_list可选参数，
#为最小化`loss`的`tf.Variable`的列表或元组;gate_gradients表示如何选择梯度的计算，可选参数为
#`GATE_NONE`, `GATE_OP`, or `GATE_GRAPH`
     grads_and_vars = optimizer.compute_gradients(cnn.loss)
#.apply_gradients(grads_and_vars, global_step=None, name=None)grads_and_vars表示从
#compute_gradients()返回的梯度或者变量对的列表;global_step表示变量更新时所增加的值，可选参数;
#name为返回操作的名称，可选参数。 默认为传递给“Optimizer”构造函数的名称。
     train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)
     #Keep track of gradient values and sparsity (optional)
     grad_summaries = []
     for g, v in grads_and_vars:
         if g is not None:
            name=v.name.replace(':','_')  #用_代替:
            grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(name), g)
            sparsity_summary = tf.summary.scalar("{}/grad/sparsity".format(name), tf.nn.zero_fraction(g))
            grad_summaries.append(grad_hist_summary)
            grad_summaries.append(sparsity_summary)
#tf.summary.merge(inputs, collections=None, name=None)inputs表示包含序列化summary协议缓冲区的
#`string``Tensor`对象列表;collections为可选项，默认将此summary添加到GraphKeys.SUMMARIES;name表示操作名称 
     grad_summaries_merged = tf.summary.merge(grad_summaries)
     #损失函数和准确率的参数保存
     loss_summary = tf.summary.scalar("loss", cnn.loss)
     acc_summary = tf.summary.scalar("accuracy", cnn.accuracy)

     #训练数据保存
     train_summary_op = tf.summary.merge([loss_summary, acc_summary, grad_summaries_merged])
     train_summary_dir = "train/summaries/train"
     train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)
   
     #前面定义好的参数num_checkpoints
     saver = tf.train.Saver(tf.global_variables())
     #初始化所有变量
     sess.run(tf.global_variables_initializer())
     #batchsize：批大小。在深度学习中，一般采用SGD训练，即每次训练在训练集中取batchsize个样本训练；
     #iteration：1个iteration等于使用batchsize个样本训练一次；
     #epoch：1个epoch等于使用训练集中的全部样本训练一次；
     #练集有1000个样本，batchsize=10，那么：
     #训练完整个样本集需要：100次iteration，1次epoch。
     # Generate batches
     if FLAGS.mode == 'train':
        start_epoch=1
        #若训练过程中，半途结束则接着训练
        checkpoint = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
        if checkpoint:
            saver.restore(sess, checkpoint)
            print("restore from the checkpoint {0}".format(checkpoint))
            start_epoch += int(checkpoint.split('-')[-1])
         # 开始训练
        print('start training...')
        metrics = '  '.join(['\r[{}]','{:.2f}%','{:s}','Epoch:{:d}','{}/{}','loss={:.2f}',
                                                                   'acc={:.2f}','{}/{}'])
        bars_max = 20
        i=0
        for epoch in range(start_epoch,FLAGS.num_epochs+1):
            time_start = time.time()
            num_batches=len(y_train)//FLAGS.batch_size
            batch_trained=0
            batches = data_helpers.batch_iter(list(zip(x_train, y_train)), FLAGS.batch_size)
            # Training loop. For each batch...
            for batch in batches:
                x_batch, y_batch = zip(*batch)   # 按batch把数据拿进来
                batch_trained+=FLAGS.batch_size
                """
                A single training step
                """
                feed_dict = {cnn.input_x: x_batch, cnn.input_y: y_batch,
                              cnn.dropout_keep_prob: FLAGS.dropout_keep_prob}
                _, step, summaries, loss, accuracy = sess.run([train_op, global_step, 
                                train_summary_op, cnn.loss, cnn.accuracy],feed_dict)
                timestamp = datetime.datetime.now().isoformat()  #取当前时间
                time_now = time.time()
                time_spend = time_now - time_start  #已花费时间
                time_estimate = time_spend / (batch_trained / (num_batches*FLAGS.batch_size))  #预估总计花费时间
                #求每轮训练已完成的百分比
                percent = min(100, batch_trained /  (num_batches*FLAGS.batch_size)) * 100
                bars = math.floor(percent / 100 * bars_max) 
                timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
                sys.stdout.write(
                          metrics.format('=' * bars + '-' * (bars_max - bars),
                          percent,timestamp,epoch,batch_trained, num_batches*FLAGS.batch_size,loss,accuracy,
                          time_transform(time_spend), time_transform(time_estimate)))
                sys.stdout.flush()    #刷新
                train_summary_writer.add_summary(summaries, step)
            if epoch % 5 == 0:  # 每checkpoint_every次执行一次保存模型
               Path=saver.save(sess, os.path.join(FLAGS.checkpoint_dir,'model.ckpt'), global_step=epoch)   # 定义模型保存路径
               print("Save model checkpoint to ",Path)
     #Testing
     elif FLAGS.mode == 'test':
          #初始化所有变量
          ckpt = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
          if ckpt: 
             saver.restore(sess, ckpt)
             print("Model restored.")

          #在训练结束后，在测试数据上检测神经网络模型的最终正确率    
          feed_dict = {cnn.input_x: x_test,cnn.input_y: y_test,cnn.dropout_keep_prob: 1.0}  # 神经元全部保留
          TRAINING_STEPS, loss, test_acc = sess.run([global_step, cnn.loss, cnn.accuracy],feed_dict)
          print(("After %d training step(s), test accuracy using average model is %.2f" 
                           %(TRAINING_STEPS, test_acc)))
            
