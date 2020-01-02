import numpy as np
import tensorflow as tf
from model import Model
from train_ops import TrainOps
from train_config import TrainConfig
from dataset_loader import DatasetLoader
from random import randint
import os
from io import StringIO
from tensorflow.python.lib.io import file_io
from architecture import Architecture as Arch
import time
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib
import cv2


def create_training_ops():

    # get trainers
    model = Model()
    train_d, train_g, loss_d, loss_g, generated_images, Dx, Dg = model.trainers()

    # initialize variables
    global_step_var = tf.Variable(0, name='global_step')
    epoch_var = tf.Variable(0, name='epoch')
    batch_var = tf.Variable(0, name='batch')

    # prepare summaries
    loss_d_summary_op = tf.summary.scalar('Discriminator_Loss', loss_d)
    loss_g_summary_op = tf.summary.scalar('Generator_Loss', loss_g)
    images_summary_op = tf.summary.image('Generated_Image', generated_images, max_outputs=1)
    summary_op = tf.summary.merge_all()

def one_hot(labels):
    num_cat = Arch.num_cat
    one_hot_labels = np.eye(num_cat)[labels]
    one_hot_labels = np.reshape(one_hot_labels, [-1, 1, 1, num_cat])
    return one_hot_labels

def expand_labels(labels):
    one_hot_labels = one_hot(labels)
    # print ('one_hot_labels.shape[ex]: ', one_hot_labels.shape)
    # print ('one_hot_labels[ex]: ', one_hot_labels)
    M = one_hot_labels.shape[0]
    # print ('M[ex]: ', M)
    img_size = Arch.img_size    
    expanded_labels = one_hot_labels * np.ones([M, img_size, img_size, Arch.num_cat])
    # print ('expanded_labels.shape[ex]: ', expanded_labels.shape)
    # print ('expanded_labels[ex]: ', expanded_labels)
    # print ('one_hot_labels.shape[ex]: ', one_hot_labels.shape)
    # print ('one_hot_labels[ex]: ', one_hot_labels)
    return (one_hot_labels, expanded_labels)

def generate_z(M):
    return np.random.normal(0.0, 1.0, size=[M, 1, 1, Arch.z_size])

def random_codes(M):
    z = generate_z(M)
    labels = [randint(1, 26) for i in range(M)]
    y, y_expanded = expand_labels(labels)
    return y, y_expanded, z

def increment(variable, sess):
    sess.run(tf.assign_add(variable, 1))
    new_val = sess.run(variable)
    return new_val

def checkpoint_model(checkpoint_dir, session, step, saver):
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    model_name = checkpoint_dir + '/model-' + str(step) + '.ckpt'
    saver.save(session, model_name, global_step=step)
    print("saved checkpoint!")

def sample_category(sess, ops, config, category, num_samples, sub_dir):
    
    # prepare for calling generator
    labels = [category] * num_samples
    one_hot_labels = one_hot(labels)
    z = generate_z(num_samples)
    feed_dict = {
        'z_holder:0': z,
        'y_holder:0': one_hot_labels
    }

    # get images
    images = sess.run(ops.generated_images, feed_dict=feed_dict)
    images = images + 1.
    images = images * -128.


    # write to disk
    for i in range(images.shape[0]):
        image = images[i]
        img_tensor = tf.image.encode_png(image)
        folder = config.sample_dir + '/' + sub_dir + '/' + str(category)
        if not os.path.exists(folder):
            os.makedirs(folder)
        img_name = folder + '/' + 'sample_' + str(i) + '.png'
        output_data = sess.run(img_tensor)
        with file_io.FileIO(img_name, 'w+') as f:
            f.write(output_data)
            f.close

def sample_all_categories(sess, ops, config, num_samples, sub_dir):
    categories = [i for i in range(Arch.num_cat)]
    for category in categories:
        sample_category(sess, ops, config, category, num_samples, sub_dir)

def load_session(config):
    sess = tf.Session()

    # load stored graph into current graph
    graph_filename = str(tf.train.latest_checkpoint(config.checkpoint_dir)) + '.meta'
##    graph_filename = 'MNIST-cDCGAN-model-1\model-12175.ckpt-12175.meta'
    saver = tf.compat.v1.train.import_meta_graph(graph_filename)
    
    # restore variables into graph
    saver.restore(sess, tf.train.latest_checkpoint(config.checkpoint_dir))
        
    # load operations 
    ops = TrainOps()
    ops.populate(sess)
    return sess, ops

def sample(config):
    sess, ops = load_session(config)
    num_samples = int(config.sample)
    sample_all_categories(sess, ops, config, num_samples, 'all_samples')

def load_data(file_dir):
    # returns: train_images , train_labels 
    data = np.load(file_dir)
    return data['arr_0'], data['arr_1']

def _plot_loss(history):
    hist = pd.DataFrame(history)
    plt.figure(figsize=(20,5))
    for colnm in hist.columns:
        plt.plot(hist[colnm],label=colnm)
    plt.legend()
    plt.ylabel("loss")
    plt.xlabel("epochs")
    plt.show()

def train(sess, ops, config):
    
    start_time = time.time()
    writer = tf.summary.FileWriter(config.summary_dir, graph=tf.get_default_graph())
    saver = tf.train.Saver()

    # prepare data
    loader = DatasetLoader()
    dataset, num_batches = loader.load_dataset(config)
    iterator = dataset.make_initializable_iterator()
    next_batch = iterator.get_next()

    # counters
    epoch = sess.run(ops.epoch_var)
    batch = sess.run(ops.batch_var)
    global_step = sess.run(ops.global_step_var)

    history = []

    # loop over epochs
    while epoch < config.num_epochs:

        # draw samples
        sample_all_categories(sess, ops, config, 20, 'epoch_' + str(epoch))

        sess.run(iterator.initializer)

        # loop over batches
        while batch < num_batches:

            images, labels = sess.run(next_batch)
            _, expanded_labels = expand_labels(labels)
            M = images.shape[0]
            y, y_expanded, z = random_codes(M)

            # run session
            feed_dict = {
                'images_holder:0': images, 
                'labels_holder:0': expanded_labels,
                'y_expanded_holder:0': y_expanded,
                'z_holder:0': z,
                'y_holder:0': y
            }
            sess.run(ops.train_d, feed_dict=feed_dict)
            sess.run(ops.train_g, feed_dict=feed_dict)

            # logging
            if global_step % config.log_freq == 0:
                summary = sess.run(ops.summary_op, feed_dict=feed_dict)
                writer.add_summary(summary, global_step=global_step)

                loss_d_val = sess.run(ops.loss_d, feed_dict=feed_dict)
                loss_g_val = sess.run(ops.loss_g, feed_dict=feed_dict)
                history.append({"D":loss_d_val,"G":loss_g_val})
                
                print("epoch: " + str(epoch) + ", batch " + str(batch))
                print("G loss: " + str(loss_g_val))
                print("D loss: " + str(loss_d_val))
                
            # saving

            if global_step % config.checkpoint_freq == 0:
                checkpoint_model(config.checkpoint_dir, sess, global_step, saver)

            global_step = increment(ops.global_step_var, sess)
            batch = increment(ops.batch_var, sess)

        epoch = increment(ops.epoch_var, sess)
        sess.run(tf.assign(ops.batch_var, 0))
        batch = sess.run(ops.batch_var)
        
    _plot_loss(history)
    end_time = time.time()
    print('Time used(minutes): ', (end_time-start_time)/60)
    sess.close()

def begin_training(config):
    create_training_ops()
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    ops = TrainOps()
    ops.populate(sess)
    train(sess, ops, config)

def continue_training(config):
    sess, ops = load_session(config)
    train(sess, ops, config)

def create_output(inp, sess, ops):
    num_samples = 5
    # prepare for calling generator
    labels = [inp] * num_samples
    one_hot_labels = one_hot(labels)
    z = generate_z(num_samples)
    feed_dict = {
        'z_holder:0': z,
        'y_holder:0': one_hot_labels
        }

    # get images
    images = sess.run(ops.generated_images, feed_dict=feed_dict)
    images = images + 1.
    images = images * 128.
    return images

def letter(sess, ops, inp):
    alphabets_map = {
        'a':1,
        'b':2,
        'c':3,
        'd':4,
        'e':5,
        'f':6,
        'g':7,
        'h':8,
        'i':9,
        'j':10,
        'k':11,
        'l':12,
        'm':13,
        'n':14,
        'o':15,
        'p':16,
        'q':17,
        'r':18,
        's':19,
        't':20,
        'u':21,
        'v':22,
        'w':23,
        'x':24,
        'y':25,
        'z':26}
    samples = []
    samples = create_output(alphabets_map[inp],sess,ops)
    return samples

def letters2word(sess, ops, word):
    letters_samples = []
    for i in range(len(word)):
        letters_samples.append(letter(sess, ops, word[i]))
    genrate_word(letters_samples)

def trans(arr):
    temp_arr = []
    for i in range(len(arr[0])): # columns
        row = []
        for item in arr:
            row.append(item[i])
        temp_arr.append(row)
    return temp_arr

def clean_up(image): # clean up the black-background
    img = image
    for i in range(28):
        for k in range(28):
            if (image[i][k] == 0):
                img[i][k] = 255
    return img

def genrate_word(s):
    samples = trans(s)
    col = len(samples[0])
    row = len(samples)
    fig = plt.figure(figsize = (col,row))
    gs = gridspec.GridSpec(row,col)
    gs.update(wspace = 0.,hspace = 0.05)

    ori_im = samples[0][0].reshape(28,28)
    new_croped_im = crop_roi(ori_im)

    new_resize_img = cv2.resize(new_croped_im,dsize=(28,27),interpolation=cv2.INTER_CUBIC)
    
    matplotlib.image.imsave('original_image.png',ori_im, cmap='gray')
    matplotlib.image.imsave('new_croped_im.png', new_croped_im, cmap='gray')
    matplotlib.image.imsave('new_resize_img.png', new_resize_img, cmap='gray')
    
    
    
##    for c in range(len(samples[0])): # rows
##        for r in range(len(samples)): # columns
##            ax = plt.subplot(gs[r*col+c])
##            plt.axis('off')
##            ax.set_xticks([])
##            ax.set_yticks([])
##            ax.set_aspect('equal')
####            plt.imshow(*-1., cmap='gray')
##            plt.imshow(crop_roi(samples[r][c].reshape(28,28))*-1., cmap='gray')
##            if(r == 0):
##                word[count].append(samples[r][c].reshape(28,28)*-1.)
##            else:
##                word[count].append(np.concatenate((word[c], samples[r][c].reshape(28,28)*-1.), axis=1))
##
##
##                
##        count += 1

##    print(np.array(samples,dtype=np.float32).shape)
    
    
##    for r in range(len(samples)): # rows
##        temp = []
##        for c in range(len(samples[0])): # columns
##            if(c == 0):
##                w = (samples[r][c].reshape(28,28)*-1.)
##            else:
##                w = np.concatenate((w, samples[r][c].reshape(28,28)*-1.), axis=1)
##
##        if(r == 0):
##            word = w
##        else:
##            word = np.concatenate((word,w),axis=0)
##    matplotlib.image.imsave(str(r)+'.png', word, cmap='gray')
 
##    form_word = np.array(word,dtype=np.float32)
##    matplotlib.image.imsave('word.png', w, cmap='gray')

    for r in range(len(samples)): # rows
        temp = []
        for c in range(len(samples[0])): # columns
            temp.append(samples[r][c].reshape(28,28))
        
##        if(r == 0):
##            word = refine_image(temp)
##        else:
##            word = np.concatenate((word,refine_image(temp)),axis=0)
        matplotlib.image.imsave(str(r)+'.png', refine_image(temp)*-1., cmap='gray')
                
##    matplotlib.image.imsave('final.png', word, cmap='gray')

def refine_image(images): # refine word by crop letter and average resize
    avg_x = 0
    avg_y = 0
    for img in images:
        croped_image = crop_roi(img)
        avg_x += croped_image.shape[0]
        avg_y += croped_image.shape[1]
    avg_x = int(np.ceil((avg_x/len(images))))
    avg_y = int(np.ceil((avg_y/len(images))))
    print('avg_x: ',avg_x)
    print('avg_y: ',avg_y)
    

    for i in range(len(images)):
        if (i == 0):
            refined_images = cv2.resize(crop_roi(images[i]),dsize=(avg_x,avg_y),interpolation=cv2.INTER_CUBIC)
        else:
            refined_images = np.concatenate((refined_images, cv2.resize(crop_roi(images[i]),dsize=(avg_x,avg_y),interpolation=cv2.INTER_CUBIC)), axis=1)
    return refined_images
            

def crop_roi(img): # crop ROI (Region Of Interest)
    original_image = img 
    n = img.shape[0]-1
    x1 = n
    y1 = 0
    x2 = 0
    y2 = n
    found_y1 = False
    found_y2 = False

    # Find x1 and y1
    for y in range(img.shape[0]):
        for x in range(img.shape[1]):
            if(img[y][x] > 20): # found object
                if(y == 0 and not found_y1): # at the edge
                    y1 = y
                    found_y1 = True
                elif(not found_y1):
                    y1 = y-1
                    found_y1 = True
                if(x < x1):
                    if(x == 0):
                        x1 = x
                    else:
                        x1 = x-1
    # Find x2 and y2
    for y in range(n,0,-1):
        for x in range(n,0,-1):
            if(img[y][x] > 20): # found object
                if(y == n and not found_y2):
                    y2 = y
                    found_y2 = True
                elif(not found_y2):
                    y2 = y+1
                    found_y2 = True
                if(x > x2):
                    if(x == n):
                        x2 = x
                    else:
                        x2 = x+1    
    return original_image[y1:y2, x1:x2]

# Run
if __name__ == '__main__':
##    config = TrainConfig()
##    if config.sample > 0:
##        sample(config)
##    elif config.should_continue:
##        continue_training(config)
##    else:
##        begin_training(config)
    # sess, ops = load_session(config)

    # - get results -
    config = TrainConfig()
    sess, ops = load_session(config)
    letters2word(sess,ops,'hello')


