import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import numpy as np
import random
import cv2 as cv
import image_processing as improc




# hyper-parameters
DEV_SET_SIZE = 600
BATCH_SIZE = 1 # MUST NOT bigger than DEV_SET_SIZE
LEARN_RATE = 0.00008
IMAGE_SIZE = (512, 512)
KERNEL_SIZE = (3, 3)
index_list = list(range(0, DEV_SET_SIZE))
indx_from = 0
random.shuffle(index_list)

def get_input_size(output_size):
    """
    calculate input_size follows 'overlap-tile strategy' on output_size
    :param output_size: tuple/list
                    desired output_size, (height, width)
    :return: tuple
                    (in_height, in_width)
    """
    oheight, owidth = output_size
    iheight = ((((((((((((oheight+8)//2)+8)//2)+8)//2)+8)*2)+8)*2)+8)*2)+8
    iwidth = ((((((((((((owidth+8)//2)+8)//2)+8)//2)+8)*2)+8)*2)+8)*2)+8
    return iheight, iwidth


def get_batch(batchsize, out_size, img_path, label_path, wm_path):
    """
    read from disk to make training batch
    :param batchsize: int
                      batch size
    :param out_size: tuple/list
                    net output size (height, width)
    :param img_path: str
                     training set path + prefix
    :param label_path: str
                    label set path + prefix
    :param wm_path: str
                    weight map set path + prefix
    :return: 4-d tensor, 4-d tensor, 4-d tensor
             (img_batch, label_batch, wm_batch) ready-to-train batch, shape (batch, weight, width, channel=1)
    """
    global index_list, indx_from
    arrindx = 0 # index to update ndarray
    in_size = get_input_size(out_size)
    image_batch = np.zeros(shape=(batchsize, in_size[0], in_size[1])).astype(np.float32) # shape(batch, row, column)
    label_batch = np.zeros(shape=(batchsize, out_size[0], out_size[1], 2)).astype(np.float32) # shape(batch, row, column, #class)
    wm_batch = np.zeros(shape=(batchsize, out_size[0], out_size[1], 2)).astype(np.float32)
    if indx_from + batchsize <= len(index_list):
        head = batchsize
        tail = 0
    else:
        head = len(index_list) - indx_from
        tail = batchsize - head
    for i in range(indx_from, indx_from + head):
        img = improc.image_mirror_padding(cv.imread(img_path + str(index_list[i]) + '.png', 0).astype(np.float32), shape=in_size)
        lb = cv.imread(label_path + str(index_list[i]) + '.png', 0).astype(np.float32)
        wm = cv.imread(wm_path + str(index_list[i]) + '.png', 0).astype(np.float32)
        lb_cls1 = (lb>127).astype(int) * 1  # class1(black pix) set to 1
        lb_cls2 = (lb<127).astype(int) * 1  # class2(white pix) set to 1
        image_batch[arrindx] += img
        label_batch[arrindx, :, :, 0] = lb_cls1
        label_batch[arrindx, :, :, 1] = lb_cls2
        wm = wm / 25 # 25 is wm visualize scaler
        wm = np.where(wm<0.5, 0.5, wm) # up scale inner cell weight to speedup inner learning
        wm_batch[arrindx, :, :, 0] = wm
        wm_batch[arrindx, :, :, 1] = wm
        arrindx += 1
    indx_from += head
    if indx_from >= len(index_list):
        indx_from = 0
    for j in range(indx_from, indx_from + tail):
        img = improc.image_mirror_padding(cv.imread(img_path + str(index_list[j]) + '.png', 0).astype(np.float32), shape=in_size)
        lb = cv.imread(label_path + str(index_list[j]) + '.png', 0).astype(np.float32)
        wm = cv.imread(wm_path + str(index_list[j]) + '.png', 0).astype(np.float32)
        lb_cls1 = (lb>127).astype(int) * 1
        lb_cls2 = (lb<127).astype(int) * 1
        image_batch[arrindx] = img
        label_batch[arrindx, :, :, 0] = lb_cls1
        label_batch[arrindx, :, :, 1] = lb_cls2
        wm = wm / 25 # 25 is wm visualize scaler
        wm = np.where(wm<0.5, 0.5, wm) # up scale inner cell weight to speedup inner learning
        wm_batch[arrindx, :, :, 0] = wm
        wm_batch[arrindx, :, :, 1] = wm
        arrindx += 1

    indx_from += tail

    image_batch = image_batch[:, :, :, np.newaxis] # shape=(batch, row, column, channel)
    return image_batch, label_batch, wm_batch

# calculate input size (overlap-tile strategy)
in_sz = get_input_size(IMAGE_SIZE)
in_shp = (in_sz[0], in_sz[1]) # (in_height, in_width)

# u-net
img_batch = tf.placeholder(tf.float32, shape=(BATCH_SIZE, in_shp[0], in_shp[1], 1), name='Image_In') # shape(batch, in_height, in_width, channel)
lbl_batch = tf.placeholder(tf.float32, shape=(BATCH_SIZE, IMAGE_SIZE[0], IMAGE_SIZE[1], 2), name='Ground_Truth')
wmap_batch = tf.placeholder(tf.float32, shape=(BATCH_SIZE, IMAGE_SIZE[0], IMAGE_SIZE[1], 2), name='Weight_Map') #shape(batch, out_height, out_width, channel)

# down sampling stage 1
# conv: kernel=(3,3), stride=(1,1), dilation=1, channel=128, act.func = ReLU
conv11 = tf.layers.conv2d(
                         inputs=img_batch, filters=128, kernel_size=KERNEL_SIZE, strides=(1, 1),
                         padding='valid', data_format='channels_last', dilation_rate=1,
                         activation=tf.nn.relu, use_bias=True, kernel_initializer=tf.glorot_uniform_initializer(),
                         bias_initializer=tf.zeros_initializer(), name='conv11'
                         )
# conv: kernel=(3,3), stride=(1,1), dilation=3, channel=128, act.func = ReLU
conv12 = tf.layers.conv2d(
                         inputs=conv11, filters=128, kernel_size=KERNEL_SIZE, strides=(1, 1),
                         padding='valid', data_format='channels_last', dilation_rate=3,
                         activation=tf.nn.relu, use_bias=True, kernel_initializer=tf.glorot_uniform_initializer(),
                         bias_initializer=tf.zeros_initializer(), name='conv12'
                         )
# conv: kernel=(2,2), stride=(2,2), dilation=1, channel=128, act.func = ReLU
conv13 = tf.layers.conv2d(
                         inputs=conv12, filters=128, kernel_size=(2, 2), strides=(2, 2),
                         padding='valid', data_format='channels_last', dilation_rate=1,
                         activation=tf.nn.relu, use_bias=True, kernel_initializer=tf.glorot_uniform_initializer(),
                         bias_initializer=tf.zeros_initializer(), name='conv13'
                         )

# down sampling stage 2
# conv: kernel=(3,3), stride=(1,1), dilation=1, channel=256, act.func = ReLU
conv21 = tf.layers.conv2d(
                         inputs=conv13, filters=256, kernel_size=KERNEL_SIZE, strides=(1, 1),
                         padding='valid', data_format='channels_last', dilation_rate=1,
                         activation=tf.nn.relu, use_bias=True, kernel_initializer=tf.glorot_uniform_initializer(),
                         bias_initializer=tf.zeros_initializer(), name='conv21'
                         )
# conv: kernel=(3,3), stride=(1,1), dilation=3, channel=256, act.func = ReLU
conv22 = tf.layers.conv2d(
                         inputs=conv21, filters=256, kernel_size=KERNEL_SIZE, strides=(1, 1),
                         padding='valid', data_format='channels_last', dilation_rate=3,
                         activation=tf.nn.relu, use_bias=True, kernel_initializer=tf.glorot_uniform_initializer(),
                         bias_initializer=tf.zeros_initializer(), name='conv22'
                         )
# conv: kernel=(2,2), stride=(2,2), dilation=1, channel=256, act.func = ReLU
conv23 = tf.layers.conv2d(
                         inputs=conv22, filters=256, kernel_size=(2, 2), strides=(2, 2),
                         padding='valid', data_format='channels_last', dilation_rate=1,
                         activation=tf.nn.relu, use_bias=True, kernel_initializer=tf.glorot_uniform_initializer(),
                         bias_initializer=tf.zeros_initializer(), name='conv23'
                         )

# down sampling stage 3
# conv: kernel=(3,3), stride=(1,1), dilation=1, channel=512, act.func = ReLU
conv31 = tf.layers.conv2d(
                         inputs=conv23, filters=512, kernel_size=KERNEL_SIZE, strides=(1, 1),
                         padding='valid', data_format='channels_last', dilation_rate=1,
                         activation=tf.nn.relu, use_bias=True, kernel_initializer=tf.glorot_uniform_initializer(),
                         bias_initializer=tf.zeros_initializer(), name='conv31'
                         )
# conv: kernel=(3,3), stride=(1,1), dilation=3, channel=512, act.func = ReLU
conv32 = tf.layers.conv2d(
                         inputs=conv31, filters=512, kernel_size=KERNEL_SIZE, strides=(1, 1),
                         padding='valid', data_format='channels_last', dilation_rate=3,
                         activation=tf.nn.relu, use_bias=True, kernel_initializer=tf.glorot_uniform_initializer(),
                         bias_initializer=tf.zeros_initializer(), name='conv32'
                         )
# conv: kernel=(2,2), stride=(2,2), dilation=1, channel=512, act.func = ReLU
conv33 = tf.layers.conv2d(
                         inputs=conv32, filters=512, kernel_size=(2, 2), strides=(2, 2),
                         padding='valid', data_format='channels_last', dilation_rate=1,
                         activation=tf.nn.relu, use_bias=True, kernel_initializer=tf.glorot_uniform_initializer(),
                         bias_initializer=tf.zeros_initializer(), name='conv33'
                         )

# up sampling stage 1
# conv: kernel=(3,3), stride=(1,1), dilation=1, channel=1024, act.func = ReLU
conv41 = tf.layers.conv2d(
                         inputs=conv33, filters=1024, kernel_size=KERNEL_SIZE, strides=(1, 1),
                         padding='valid', data_format='channels_last', dilation_rate=1,
                         activation=tf.nn.relu, use_bias=True, kernel_initializer=tf.glorot_uniform_initializer(),
                         bias_initializer=tf.zeros_initializer(), name='conv41'
                         )
# conv: kernel=(3,3), stride=(1,1), dilation=3, channel=1024, act.func = ReLU
conv42 = tf.layers.conv2d(
                         inputs=conv41, filters=1024, kernel_size=KERNEL_SIZE, strides=(1, 1),
                         padding='valid', data_format='channels_last', dilation_rate=3,
                         activation=tf.nn.relu, use_bias=True, kernel_initializer=tf.glorot_uniform_initializer(),
                         bias_initializer=tf.zeros_initializer(), name='conv42'
                         )
# deconv: kernel=(2,2), stride=(2,2), dilation=1, channel=512, act.func = ReLU
filter43 = tf.get_variable(
                          name='deconv43_filter', shape=[2, 2, 512, 1024],
                          initializer=tf.glorot_uniform_initializer()
                          )
out_shape43 = tf.constant([BATCH_SIZE, 142, 142, 512])
deconv43 = tf.nn.conv2d_transpose(
                                 input=conv42, filter=filter43, output_shape=out_shape43, strides=2,
                                 padding='VALID', data_format='NHWC', name='deconv_43', dilations=1
                                 )
deconv43 = tf.nn.relu(deconv43)

# up sampling stage 2
# concatenated feature map conv: kernel=(3,3), stride=(1,1), dilation=1, channel=512, act.func = ReLU
crop51 = improc.feature_map_crop(conv32, (deconv43.shape[1], deconv43.shape[2]))
in_tensor51 = tf.concat([crop51, deconv43], axis=3, name='crop_copy_51')
conv51 = tf.layers.conv2d(
                         inputs=in_tensor51, filters=512, kernel_size=KERNEL_SIZE, strides=(1, 1),
                         padding='valid', data_format='channels_last', dilation_rate=1,
                         activation=tf.nn.relu, use_bias=True, kernel_initializer=tf.glorot_uniform_initializer(),
                         bias_initializer=tf.zeros_initializer(), name='conv51'
                         )
# conv: kernel=(3,3), stride=(1,1), dilation=3, channel=512, act.func = ReLU
conv52 = tf.layers.conv2d(
                         inputs=conv51, filters=512, kernel_size=KERNEL_SIZE, strides=(1, 1),
                         padding='valid', data_format='channels_last', dilation_rate=3,
                         activation=tf.nn.relu, use_bias=True, kernel_initializer=tf.glorot_uniform_initializer(),
                         bias_initializer=tf.zeros_initializer(), name='conv52'
                         )
# deconv: kernel=(2,2), stride=(2,2), dilation=1, channel=256, act.func = ReLU
filter53 = tf.get_variable(
                          name='deconv53_filter', shape=[2, 2, 256, 512],
                          initializer=tf.glorot_uniform_initializer()
                          )
out_shape53 = tf.constant([BATCH_SIZE, 268, 268, 256])
deconv53 = tf.nn.conv2d_transpose(
                                 input=conv52, filter=filter53, output_shape=out_shape53, strides=2,
                                 padding='VALID', data_format='NHWC', name='deconv_53', dilations=1
                                 )
deconv53 = tf.nn.relu(deconv53)

# up sampling stage 3
# concatenated feature map conv: kernel=(3,3), stride=(1,1), dilation=1, channel=256, act.func = ReLU
crop61 = improc.feature_map_crop(conv22, (deconv53.shape[1], deconv53.shape[2]))
in_tensor61 = tf.concat([crop61, deconv53], axis=3, name='crop_copy_61')
conv61 = tf.layers.conv2d(
                         inputs=in_tensor61, filters=256, kernel_size=KERNEL_SIZE, strides=(1, 1),
                         padding='valid', data_format='channels_last', dilation_rate=1,
                         activation=tf.nn.relu, use_bias=True, kernel_initializer=tf.glorot_uniform_initializer(),
                         bias_initializer=tf.zeros_initializer(), name='conv61'
                         )
# conv: kernel=(3,3), stride=(1,1), dilation=3, channel=256, act.func = ReLU
conv62 = tf.layers.conv2d(
                         inputs=conv61, filters=256, kernel_size=KERNEL_SIZE, strides=(1, 1),
                         padding='valid', data_format='channels_last', dilation_rate=3,
                         activation=tf.nn.relu, use_bias=True, kernel_initializer=tf.glorot_uniform_initializer(),
                         bias_initializer=tf.zeros_initializer(), name='conv62'
                         )
# deconv: kernel=(2,2), stride=(2,2), dilation=1, channel=128, act.func = ReLU
filter63 = tf.get_variable(
                          name='deconv63_filter', shape=[2, 2, 128, 256],
                          initializer=tf.glorot_uniform_initializer()
                          )
out_shape63 = tf.constant([BATCH_SIZE, 520, 520, 128])
deconv63 = tf.nn.conv2d_transpose(
                                 input=conv62, filter=filter63, output_shape=out_shape63, strides=2,
                                 padding='VALID', data_format='NHWC', name='deconv_63', dilations=1
                                 )
deconv63 = tf.nn.relu(deconv63)

# finial stage
# concatenated feature map conv: kernel=(3,3), stride=(1,1), dilation=1, channel=128, act.func = ReLU
crop71 = improc.feature_map_crop(conv12, (deconv63.shape[1], deconv63.shape[2]))
in_tensor71 = tf.concat([crop71, deconv63], axis=3, name='crop_copy_71')
conv71 = tf.layers.conv2d(
                         inputs=in_tensor71, filters=128, kernel_size=KERNEL_SIZE, strides=(1, 1),
                         padding='valid', data_format='channels_last', dilation_rate=1,
                         activation=tf.nn.relu, use_bias=True, kernel_initializer=tf.glorot_uniform_initializer(),
                         bias_initializer=tf.zeros_initializer(), name='conv71'
                         )
# conv: kernel=(3,3), stride=(1,1), dilation=3, channel=128, act.func = ReLU
conv72 = tf.layers.conv2d(
                         inputs=conv71, filters=128, kernel_size=KERNEL_SIZE, strides=(1, 1),
                         padding='valid', data_format='channels_last', dilation_rate=3,
                         activation=tf.nn.relu, use_bias=True, kernel_initializer=tf.glorot_uniform_initializer(),
                         bias_initializer=tf.zeros_initializer(), name='conv72'
                         )

# final output segmentation probability map
pred_masks = tf.layers.conv2d(
                         inputs=conv72, filters=2, kernel_size=(1, 1), strides=(1, 1),
                         padding='valid', data_format='channels_last', dilation_rate=1,
                         activation=None, use_bias=True, kernel_initializer=tf.glorot_uniform_initializer(),
                         bias_initializer=tf.zeros_initializer(), name='seg_map_output'
                         )

soft_max = tf.nn.softmax(pred_masks, axis=3, name='soft_max') #shape(batch, height, width, #class)
# loss = average cross entropy loss over each pixel
loss = -tf.reduce_sum(tf.multiply(wmap_batch, tf.multiply(lbl_batch, tf.log(soft_max)))) / (IMAGE_SIZE[0]*IMAGE_SIZE[1])
optimizer = tf.train.AdamOptimizer(LEARN_RATE).minimize(loss)
saver = tf.train.Saver()

def train(iteration, impath, lbpath, wmpath, svpath, file_name, cont=False):
    with tf.Session() as sess:
        if cont:
            if not os.path.exists(svpath):
                print('train err: check point path not found, exit.')
                return
            saver.restore(sess, svpath+file_name)
            print('from:\n' + svpath+file_name + '\nrestore model succeed, continue training...')
            print('iteration to go:', iteration)
        else:
            print('start new training...')
            print('total iteration:', iteration)
            sess.run(tf.global_variables_initializer())
            if not os.path.exists(svpath):
                os.mkdir(svpath)
        for i in range(iteration):
            imbatch, lbatch, wbatch = get_batch(BATCH_SIZE, IMAGE_SIZE, impath, lbpath, wmpath)
            feed_dict={img_batch:imbatch, lbl_batch:lbatch, wmap_batch: wbatch}
            xnloss, _ = sess.run([loss, optimizer], feed_dict)
            if i % 1 == 0:
                print('total iter:', iteration, '   iter:', i, '   loss:', xnloss)
            if i % 1 == 0:
                save_path = saver.save(sess, svpath+file_name)
                print('check point saved.')
        print('training completed!')
    return


def run(img_list, result_path, model_path, model_name):
    if not os.path.exists(result_path):
        os.mkdir(result_path)
    result_list = []
    with tf.Session() as sess:
        print('restoring model:\n'+ model_path + model_name)
        saver.restore(sess, model_path+model_name)
        print('restore model succeed.')

        for i, img in enumerate(img_list, 0):
            inheight, inwidth = get_input_size((img.shape[0],img.shape[1]))
            imgbatch = np.zeros((1, inheight, inwidth, 1)).astype(np.float32)
            imgbatch[0, :, :, 0] = improc.image_mirror_padding(img, (inheight, inwidth))
            smx = sess.run(soft_max, feed_dict={img_batch:imgbatch})
            result = np.where(smx[:, :, :, 0]<smx[:, :, :, 1], 0, 255).astype(np.uint8)[0]
            result_list.append(result)
            cv.imwrite(result_path+'/result'+str(i)+'.png', result, [int(cv.IMWRITE_PNG_COMPRESSION), 0])
            print('result No.'+ str(i), 'saved.')
    return result_list
