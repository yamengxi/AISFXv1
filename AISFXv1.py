import tensorflow as tf
import numpy as np
import random, os, cv2


class G:
    depth = 32 # 视频帧数为32
    epochs = 100000
    batch_size = 1 # batch大小
    train = 1 # 当前是否正在训练
    alpha = 0.2 # leaky relu coeff
    lambda1 = 10 # 梯度惩罚权重
    learning_rate = 0.001 # 学习速率
    beta1 = 0.5
    beta2 = 0.9
    g_model = 0
    d_loss = 0
    g_loss = 0
    temp = 0


def model_inputs(D, H, W, C):
    input_img = tf.placeholder(tf.float32, (None, H, W, C), name = 'input_img')

    input_z = tf.placeholder(tf.float32, (None, H, W, C), name = 'input_z')

    input_video = tf.placeholder(tf.float32, (None, D, H, W, C), name = 'input_video')

    return input_img, input_z, input_video


#data format of x0 is NHWC
#size of x0 is 2 * 256 * 256 * 3
def get_fm(x0):
    
    # 2 * 256 * 256 * 3
    x1 = tf.layers.conv3d(x0, 8, [2, 4, 4], [1, 1, 1], 'same')
    relu1 = tf.maximum(G.alpha * x1, x1)
    #print(x1.get_shape().as_list())

    # 2 * 256 * 256 * 8
    x2 = tf.layers.conv3d_transpose(relu1, 16, [2, 4, 4], [2, 1, 1], 'same')
    relu2 = tf.maximum(G.alpha * x2, x2)
    
    # 4 * 256 * 256 * 16
    x3 = tf.layers.conv3d_transpose(relu2, 32, [2, 4, 4], [2, 1, 1], 'same')
    relu3 = tf.maximum(G.alpha * x3, x3)

    # 8 * 256 * 256 * 32
    x4 = tf.layers.conv3d_transpose(relu3, 32, [4, 4, 4], [2, 1, 1], 'same')
    relu4 = tf.maximum(G.alpha * x4, x4)

    # 16 * 256 * 256 * 32
    f = tf.layers.conv3d_transpose(relu4, 3, [4, 4, 4], [2, 1, 1], 'same')
    f = tf.nn.tanh(f)
    # f: 32 * 256 * 256 * 3
    m = tf.layers.conv3d_transpose(relu4, 1, [4, 4, 4], [2, 1, 1], 'same')
    m = tf.nn.sigmoid(m)
    # m: 32 * 256 * 256 * 1

    return f[:,0:G.depth-1,:,:,:], m[:,0:G.depth-1,:,:,:]


#data format of x0 is NHWC
#size of x0 is 2 * 256 * 256 * 3
def get_b(x0):
    x1 = tf.layers.conv3d(x0, 8, [2, 4, 4], [2, 1, 1], padding = 'same')
    relu1 = tf.maximum(G.alpha * x1, x1)

    # 1 * 256 * 256 * 8
    relu1 = tf.reshape(relu1, [-1, 256, 256, 8])
    
    # 256 * 256 * 8
    x2 = tf.layers.conv2d(relu1, 16, 4, 1, padding = 'same')
    relu2 = tf.maximum(G.alpha * x2, x2)

    # 256 * 256 * 16
    x3 = tf.layers.conv2d(relu2, 32, 4, 1, padding = 'same')
    relu3 = tf.maximum(G.alpha * x3, x3)

    # 256 * 256 * 32
    x4 = tf.layers.conv2d(relu3, 3, 4, 1, padding = 'same')
    x4 = tf.nn.tanh(x4)

    # 256 * 256 * 3

    return x4



#data format of img and z is NHWC
#size of img and z is 256 * 256 * 3
def generator(img, z, reuse):

    with tf.variable_scope('generator', reuse = reuse):
        img = tf.reshape(img, [-1, 1, 256, 256, 3])
        z = tf.reshape(img, [-1, 1, 256, 256, 3])
        x0 = tf.concat([img, z], 1)

        f, m = get_fm(x0)
        b = get_b(x0)
        #print(b.get_shape().as_list())
        #f: 31 * 256 * 256 * 3
        #m: 31 * 256 * 256 * 1
        #b: 256 * 256 * 3

        #利用tensorflow的Broadcasting机制
        #res: 31 * 256 * 256 * 3
        res = m*f + (1-m)*b
        #print(res.get_shape().as_list())
        img = tf.reshape(img, [-1, 1, 256, 256, 3])
        res = tf.concat([img, res], 1)
        #res: 32 * 256 * 256 * 3
        #print(res.get_shape().as_list())
        return res

'''
def final1(video):
    # 32 * 256 * 256 * 3
    x0 = tf.layers.conv3d(video, 16, 4, 2, padding = 'same')
    relu0 = tf.maximum(G.alpha * x0, x0)

    # 16 * 128 * 128 * 16
    x1 = tf.layers.conv3d(relu0, 32, 4, 2, padding = 'same')
    relu1 = tf.maximum(G.alpha * x1, x1)

    # 8 * 64 * 64 * 32
    x2 = tf.layers.conv3d(relu1, 64, 4, 2, padding = 'same')
    relu2 = tf.maximum(G.alpha * x2, x2)

    # 4 * 32 * 32 * 64
    x3 = tf.layers.conv3d(relu2, 128, 4, 1, padding = 'valid')
    relu3 = tf.maximum(G.alpha * x3, x3)

    # 1 * 13 * 13 * 128
    # Flatten
    flatten = tf.reshape(relu3, (-1, 1 * 13 * 13 * 128))
    final = tf.layers.dense(flatten, 1)

    return final


def final2(video):
    # 31 * 128 * 128 * 3
    x0 = tf.layers.conv3d(video, 16, 4, 2, padding = 'same')
    relu0 = tf.maximum(G.alpha * x0, x0)

    # 16 * 64 * 64 * 16
    x1 = tf.layers.conv3d(relu0, 32, 4, 2, padding = 'same')
    relu1 = tf.maximum(G.alpha * x1, x1)

    # 8 * 32 * 32 * 32
    x2 = tf.layers.conv3d(relu1, 64, 4, 2, padding = 'same')
    relu2 = tf.maximum(G.alpha * x2, x2)

    # 4 * 16 * 16 * 64
    x3 = tf.layers.conv3d(relu2, 128, 4, 1, padding = 'valid')
    relu3 = tf.maximum(G.alpha * x3, x3)

    # 1 * 13 * 13 * 128
    # Flatten
    flatten = tf.reshape(relu3, (-1, 1 * 13 * 13 * 128))
    final = tf.layers.dense(flatten, 1)

    return final
'''


def discriminator(video, reuse):

    with tf.variable_scope('discriminator', reuse = reuse):
        #final = 0.3 * final1(video) + 0.7 * final2(video[:,1:G.depth,:,:,:] - video[:,0:G.depth-1,:,:,:])
        #return final

        
        video1 = video
        video2 = video[:,1:G.depth,:,:,:] - video[:,0:G.depth-1,:,:,:]
        video3 = video[:,2:G.depth,:,:,:] - video[:,0:G.depth-2,:,:,:]
        video4 = video[:,4:G.depth,:,:,:] - video[:,0:G.depth-4,:,:,:]
        video5 = video[:,8:G.depth,:,:,:] - video[:,0:G.depth-8,:,:,:]

        # new_depth = D + (D-1) + (D-2) + (D-4) + (D-8)
        # new_depth = 5*D-15
        # 145 * 256 * 256 * 3
        super_video = tf.concat([video1, video2, video3, video4, video5], 1)
        
        # 145 * 256 * 256 * 3
        x0 = tf.layers.conv3d(super_video, 8, 4, [2, 2, 2], padding = 'same')
        relu0 = tf.maximum(G.alpha * x0, x0)

        # 73 * 128 * 128 * 8
        x1 = tf.layers.conv3d(relu0, 16, 4, [2, 2, 2], padding = 'same')
        relu1 = tf.maximum(G.alpha * x1, x1)

        # 37 * 64 * 64 * 16
        x2 = tf.layers.conv3d(relu1, 32, 4, [2, 2, 2], padding = 'same')
        relu2 = tf.maximum(G.alpha * x2, x2)

        # 19 * 32 * 32 * 32
        x3 = tf.layers.conv3d(relu2, 64, 4, [2, 2, 2], padding = 'same')
        relu3 = tf.maximum(G.alpha * x3, x3)

        # 10 * 16 * 16 * 64
        x4 = tf.layers.conv3d(relu3, 128, 4, [2, 2, 2], padding = 'same')
        relu4 = tf.maximum(G.alpha * x4, x4)

        # 5 * 8 * 8 * 128
        x5 = tf.layers.conv3d(relu4, 256, [5, 4, 4], 1, padding = 'valid')
        relu5 = tf.maximum(G.alpha * x5, x5)

        # 1 * 5 * 5 * 256
        # Flatten
        flatten = tf.reshape(relu5, (-1, 1 * 5 * 5 * 256))
        final = tf.layers.dense(flatten, 1)

        return final


def model_opt(img, z, video):
    g_model = generator(img, z, reuse = False)

    d_model_real = discriminator(video, reuse = False)

    d_model_fake = discriminator(g_model, reuse = True)

    d_loss = tf.reduce_mean(d_model_fake) - tf.reduce_mean(d_model_real)

    g_loss = -tf.reduce_mean(d_model_fake)

    all_vars = tf.trainable_variables()
    d_vars = [var for var in all_vars if var.name.startswith('discriminator')]
    g_vars = [var for var in all_vars if var.name.startswith('generator')]

    alpha = tf.random_uniform(shape=[G.batch_size, 1, 1, 1, 1], minval=0., maxval=1. )
    video_ = alpha*video + (1-alpha)*g_model
    grads = tf.gradients(discriminator(video_, reuse = True), [video_])[0]
    slops = tf.sqrt(tf.reduce_sum(tf.square(tf.reshape(grads, [G.batch_size, -1]))))
    d_loss += G.lambda1 * tf.reduce_mean((slops - 1.0) ** 2)

    train_op_d = tf.train.AdamOptimizer(G.learning_rate, beta1 = G.beta1, beta2 = G.beta2).minimize(d_loss, var_list = d_vars)
    train_op_g = tf.train.AdamOptimizer(G.learning_rate, beta1 = G.beta1, beta2 = G.beta2).minimize(g_loss, var_list = g_vars)

    G.g_model = g_model
    G.d_loss = d_loss
    G.g_loss = g_loss
    
    return train_op_d, train_op_g


def output(epoch_i, real_video, fake_video):
    real_video = np.uint8(real_video*127.5 + 127.5)
    fake_video = np.uint8(fake_video*127.5 + 127.5)


    if not os.path.isdir('output\\'+str(epoch_i)):
        os.mkdir('output\\'+str(epoch_i))
    
    for i in range(G.batch_size):
        if not os.path.isdir('output\\'+str(epoch_i)+'\\'+str(i)):
            os.mkdir('output\\'+str(epoch_i)+'\\'+str(i))
        
        video_writer = cv2.VideoWriter('output\\' + str(epoch_i) + '\\'+str(i)  + '\\real_video.avi', cv2.VideoWriter_fourcc(*'XVID'), 10, (256, 256))# size = (W, H)

        for j in range(G.depth):
            img_now = real_video[i,j]
            cv2.imwrite('output\\'+str(epoch_i)+'\\'+str(i)+'\\real'+str(j)+'.jpg', img_now)
            video_writer.write(img_now)

    for i in range(G.batch_size):
        if not os.path.isdir('output\\'+str(epoch_i)+'\\'+str(i)):
            os.mkdir('output\\'+str(epoch_i)+'\\'+str(i))

        video_writer = cv2.VideoWriter('output\\' + str(epoch_i) + '\\'+str(i) + '\\fake_video.avi', cv2.VideoWriter_fourcc(*'XVID'), 10, (256, 256))

        for j in range(G.depth):
            img_now = fake_video[i,j]
            cv2.imwrite('output\\'+str(epoch_i)+'\\'+str(i)+'\\fake'+str(j)+'.jpg', img_now)
            video_writer.write(img_now)


'''
def get_data():
    red = random.randint(0, 255)
    green = random.randint(0, 255)
    blue = random.randint(0, 255)
    img = np.ones((G.batch_size, 256, 256, 3))
    img[:,:,:,0] *= red
    img[:,:,:,1] *= green
    img[:,:,:,2] *= blue

    z = np.random.normal(loc=0.0, scale=1.0, size = (G.batch_size, 256, 256, 3))

    video = np. ones((G.batch_size, G.depth, 256, 256, 3))
    video[:,:,:,:,0] *= red
    video[:,:,:,:,1] *= green
    video[:,:,:,:,2] *= blue

    return img/255.0, z/255.0, video/255.0
'''


from PIL import Image



# 标准化每一帧，默认H==W
def ISO(img, H, W, C):
    [h, w, c] = img.shape
    
    if h>w:
        img = cv2.resize(img, (W, h*W // w), interpolation=cv2.INTER_CUBIC)#这个resize函数的参数H和W是反的！！！！！！！！！！！！
    else:
        img = cv2.resize(img, (w*H // h, H), interpolation=cv2.INTER_CUBIC)#fuck the shit!!!!!


    img = img[0:H,0:W,:]
    #img = Image.fromarray(img)
    #img = np.array(img.rotate(-90))
    
    return img


class train_data:
    # num:视频个数
    # path:视频目录
    # H:标准化后的帧高度
    # W:标准化后的帧宽度
    # C:标准化后的帧通道数
    def __init__(self, num, path, H, W, C):
        '''
        self.num = num
        self.video_set = []# 特别注意：video_set绝对不能转化为np.array，因为它的元素们的形状不同
        for i in range(self.num):
            print('Loading the video %d ...' %i)
            video_reader = cv2.VideoCapture(path + '\\' + str(i) + '.mpg')
            full_video = []
            flag, frame = video_reader.read()
            while flag:
                frame = ISO(frame, H, W, C)
                full_video.append(frame)
                flag, frame = video_reader.read()

            full_video = np.array(full_video)
            self.video_set.append(full_video)
        '''
        #self.num = num
        self.video_set = []# 特别注意：video_set绝对不能转化为np.array，因为它的元素们的形状不同
        file_list = os.listdir(path)
        self.num = len(file_list)
        for file in file_list:
            print('Loading the video ' + file)
            video_reader = cv2.VideoCapture(path + '\\' + file)
            full_video = []
            flag, frame = video_reader.read()
            while flag:
                frame = ISO(frame, H, W, C)
                full_video.append(frame)
                flag, frame = video_reader.read()

            full_video = np.array(full_video)
            self.video_set.append(full_video)            


    def cut_video(self, depth):
        i = np.random.randint(0, self.num)
        length = self.video_set[i].shape[0]
        start = np.random.randint(0, length - depth + 1)
        return self.video_set[i][start:start+depth, :, :, :]
        

    def data_input(self, batch_size, depth):#图像标准化
        video = []
        for i in range(batch_size):
            video.append(self.cut_video(depth))
        video = np.array(video)
        video = (video-127.5) / 127.5
        
        z = abs(np.random.normal(loc=0.0, scale=2.0, size = (G.batch_size, 256, 256, 3)))
        z = (z-127.5) / 127.5
        
        return video[:,0,:,:,:], z, video



def main():
    input_img, input_z, input_video = model_inputs(G.depth, 256, 256, 3)
    d_opt, g_opt = model_opt(input_img, input_z, input_video)
    data = train_data(30, 'input', 256, 256, 3)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for epoch_i in range(G.epochs):
            img, z, video = data.data_input(G.batch_size, G.depth)

            if epoch_i % 10 == 0:

                train_loss_d = G.d_loss.eval({input_img: img, input_z: z, input_video: video})
                train_loss_g = G.g_loss.eval({input_img: img, input_z: z, input_video: video})
                    
                print("Epoch {}...".format(epoch_i+1),
                      "Discriminator Loss: {:.4f}...".format(train_loss_d),
                      "Generator Loss: {:.4f}".format(train_loss_g))

            if epoch_i % 100 == 0:
                output(epoch_i, video, G.g_model.eval({input_img: img, input_z: z, input_video: video}))
            
            #img, z, video = get_data()
            sess.run(d_opt, feed_dict = {input_img: img, input_z: z, input_video: video})
            sess.run(g_opt, feed_dict = {input_img: img, input_z: z, input_video: video})
            #sess.run(g_opt, feed_dict = {input_img: img, input_z: z, input_video: video})

            #img, z, video = data.data_input(G.batch_size, G.depth)
            #sess.run(g_opt, feed_dict = {input_img: img, input_z: z, input_video: video})
            #sess.run(g_opt, feed_dict = {input_img: img, input_z: z, input_video: video})



main()
