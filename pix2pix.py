import tensorflow as tf
import tensorflow_addons as tfa

import argparse

import os
import time
import datetime

import pandas as pd

from matplotlib import pyplot as plt
from IPython import display

import numpy as np
from PIL import Image

import math

PATH = "./"

# Check Sample
# sample_image = tf.io.read_file(PATH + 'clothing-co-parsing/photos/0002.jpg')
# sample_image = tf.io.decode_jpeg(sample_image)
# print(sample_image.shape)

# Show sample
# plt.figure()
# plt.imshow(sample_image)
# plt.show()



parser = argparse.ArgumentParser(description="ml pix2pix image segmentator")
parser.add_argument("--history", dest="history",)


# TRAINING SETTINGS

BUFFER_SIZE = 1000
BATCH_SIZE = 1

ORG_WIDTH = 320
ORG_HEIGHT = 320

IM_WIDTH = 64
IM_HEIGHT = 64


LAMBDA = 100

OUTPUT_CHANNELS = 4

EPOCHS = 150

log_dir="logs/"



def load(name,name2):
    image = tf.image.decode_jpeg(tf.io.read_file(name))
    truth = tf.strings.to_number(tf.strings.split(tf.strings.split(tf.strings.strip(tf.io.read_file(name2)), "\n"), ","), tf.int32).to_tensor()
    #truth = tf.io.decode_csv(truth, [0] * (550 * OUTPUT_CHANNELS))
    truth = tf.reshape(truth , (-1, ORG_HEIGHT ,OUTPUT_CHANNELS))

    dim = tf.shape(image)
    image = tf.concat([image, tf.zeros([dim[0],dim[1],1], dtype=tf.uint8)], axis=2)
    # Convert to float32 tensors
    return tf.cast(image, tf.float32), tf.cast(truth, tf.float32)


# Test the load fnct
# im, tr = load(1)

# plt.figure()
# plt.imshow(im/255.0)
# plt.figure()
# plt.imshow(tr/255.0)
# plt.show()


summary_writer = tf.summary.create_file_writer(log_dir + "fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

def random_crop(image, truth):
    #print(tf.size(image))
    # print(tf.shape(image)[0])
    # print(tf.shape(truth)[0])
    # print(ORG_WIDTH//2 - 6 - IM_WIDTH // 2, " ", ORG_WIDTH//2 + 6 - IM_WIDTH // 2)
    # print(ORG_HEIGHT//2 - 2 - IM_HEIGHT // 2," ", ORG_HEIGHT//2 + 2 - IM_HEIGHT // 2)
    cunt = tf.shape(image)
    width = tf.minimum(cunt[1], 256)
    height = tf.minimum(cunt[0], 256)
    o_w =tf.maximum (tf.minimum(tf.random.uniform([], minval=-15, maxval=15, dtype=tf.int32) + (ORG_WIDTH - width) // 2, cunt[1]-width), 0)
    o_h = tf.maximum(tf.minimum(tf.random.uniform([], minval=-15, maxval=15, dtype=tf.int32) + (ORG_HEIGHT - height) // 2, cunt[0]-height), 0)
    # print(tf.shape(image)[1]-IM_WIDTH)
    #print(tf.shape(image))
    # print(o_w, o_h)
    return tf.image.crop_to_bounding_box(image, o_h, o_w,  height,width), tf.image.crop_to_bounding_box(truth, o_h, o_w,  height,width)

deg10 = 40* math.pi / 180

def random_rotate(image, truth):
    theta = tf.random.uniform([], minval=-deg10, maxval=deg10, dtype=tf.float32)

    return tfa.image.rotate(image, theta), tfa.image.rotate(truth, theta)

def random_img_quality(image, truth):
    return image, truth

def normalize(image, truth):
    return (image/127.5)-1, truth

def downsize(image, truth):
    return tf.image.resize(image, [IM_WIDTH,IM_HEIGHT]), tf.image.resize(truth, [IM_WIDTH,IM_HEIGHT])

@tf.function()
def random_jitter(image, truth):
    image, truth = random_rotate(image, truth)
    image, truth = random_crop(image, truth)
    image, truth = downsize(image, truth)
    image, truth = random_img_quality(image, truth)

    if(tf.random.uniform(()) > 0.5):
        image = tf.image.flip_left_right(image)
        truth = tf.image.flip_left_right(truth)
    
    return image, truth



def load_image_train(image_file, b):
    input_image, real_image = load(image_file, b)
    input_image, real_image = random_jitter(input_image, real_image)
    input_image, real_image = normalize(input_image, real_image)

    return input_image, real_image



files = []
labels = []
for i in range(1,801):
    files.append(f"./clothing-co-parsing/cropped/{i:04d}.jpg")
    labels.append(f"./clothing-co-parsing/cropped-labels/{i:04d}.csv")

# print(labels)

dataset = tf.data.Dataset.from_tensor_slices((files, labels))
dataset = dataset.shuffle(BUFFER_SIZE)
dataset = dataset.map(load_image_train, num_parallel_calls=4)
dataset = dataset.batch(BATCH_SIZE)
dataset = dataset.apply(tf.data.experimental.ignore_errors(log_warning=True))
dataset = dataset.prefetch(1)

def downsample(filters, size, apply_batchnorm=True):
    intializer = tf.random_normal_initializer(0., 0.02)
    result = tf.keras.Sequential()
    result.add(
        tf.keras.layers.Conv2D(filters, size, strides=2, padding="same", kernel_initializer=intializer, use_bias=False)
    )
    if(apply_batchnorm):
        result.add(tf.keras.layers.BatchNormalization())
    result.add(tf.keras.layers.LeakyReLU())

    return result


def upsample(filters, size, apply_dropout=False):
    intializer = tf.random_normal_initializer(0., 0.02)

    result = tf.keras.Sequential()
    result.add(
        tf.keras.layers.Conv2DTranspose(filters, size, strides=2, padding="same", kernel_initializer=intializer, use_bias=False)
    )
    result.add(tf.keras.layers.BatchNormalization())
    if(apply_dropout):
        result.add(tf.keras.layers.Dropout(0.5))
    
    result.add(tf.keras.layers.ReLU())

    return result


def Generator():
    inputs = tf.keras.layers.Input(shape=[IM_WIDTH, IM_HEIGHT, 4])

    down_stack = [
        downsample(64, 4, apply_batchnorm=False),  # (batch_size, 128, 128, 64)
        downsample(128, 4),  # (batch_size, 64, 64, 128)
        downsample(256, 4),  # (batch_size, 32, 32, 256)
        # downsample(512, 4),  # (batch_size, 16, 16, 512)
        downsample(512, 4),  # (batch_size, 8, 8, 512)
        #downsample(512, 4),  # (batch_size, 4, 4, 512)
        downsample(512, 4),  # (batch_size, 2, 2, 512)
        downsample(512, 4),  # (batch_size, 1, 1, 512)
    ]

    up_stack = [
        upsample(512, 4, apply_dropout=True),  # (batch_size, 2, 2, 1024) 1
        # upsample(512, 4, apply_dropout=True),  # (batch_size, 4, 4, 1024) 2 
        #upsample(512, 4, apply_dropout=True),  # (batch_size, 8, 8, 1024) 3
        upsample(512, 4),  # (batch_size, 16, 16, 1024)                   4
        upsample(256, 4),  # (batch_size, 32, 32, 512)
        upsample(128, 4),  # (batch_size, 64, 64, 256)
        upsample(64, 4),  # (batch_size, 128, 128, 128)
    ]

    initializer = tf.random_normal_initializer(0., 0.02)
    last = tf.keras.layers.Conv2DTranspose(OUTPUT_CHANNELS, 4,
                                            strides=2,
                                            padding='same',
                                            kernel_initializer=initializer,
                                            activation='tanh')  # (batch_size, 256, 256, 3)

    x = inputs

    # Downsampling through the model
    skips = []
    for down in down_stack:
        x = down(x)
        skips.append(x)

    skips = reversed(skips[:-1])

    # Upsampling and establishing the skip connections
    for up, skip in zip(up_stack, skips):
        #print(x)
        #print(skip)
        x = up(x)
        x = tf.keras.layers.Concatenate()([x, skip])

    x = last(x)

    return tf.keras.Model(inputs=inputs, outputs=x)


def Discriminator():
    initializer = tf.random_normal_initializer(0., 0.02)

    inp = tf.keras.layers.Input(shape=[IM_WIDTH, IM_HEIGHT, 4], name='image')
    tar = tf.keras.layers.Input(shape=[IM_WIDTH, IM_HEIGHT, 4], name='truth')

    x = tf.keras.layers.concatenate([inp, tar])  # (batch_size, 256, 256, channels*2)

    down1 = downsample(64, 4, False)(x)  # (batch_size, 128, 128, 64)
    down2 = downsample(128, 4)(down1)  # (batch_size, 64, 64, 128)
    down3 = downsample(256, 4)(down2)  # (batch_size, 32, 32, 256)

    zero_pad1 = tf.keras.layers.ZeroPadding2D()(down3)  # (batch_size, 34, 34, 256)
    conv = tf.keras.layers.Conv2D(512, 4, strides=1,
                                    kernel_initializer=initializer,
                                    use_bias=False)(zero_pad1)  # (batch_size, 31, 31, 512)

    batchnorm1 = tf.keras.layers.BatchNormalization()(conv)

    leaky_relu = tf.keras.layers.LeakyReLU()(batchnorm1)

    zero_pad2 = tf.keras.layers.ZeroPadding2D()(leaky_relu)  # (batch_size, 33, 33, 512)

    last = tf.keras.layers.Conv2D(1, 4, strides=1,
                                    kernel_initializer=initializer)(zero_pad2)  # (batch_size, 30, 30, 1)

    return tf.keras.Model(inputs=[inp, tar], outputs=last)

generator = Generator()
discriminator = Discriminator()

loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def generator_loss(disc_generated_output, gen_output, target):
    gan_loss = loss_object(tf.ones_like(disc_generated_output), disc_generated_output)

    # Mean absolute error
    l1_loss = tf.reduce_mean(tf.abs(target - gen_output))

    total_gen_loss = gan_loss + (LAMBDA * l1_loss)

    return total_gen_loss, gan_loss, l1_loss

def discriminator_loss(disc_real_output, disc_generated_output):
    real_loss = loss_object(tf.ones_like(disc_real_output), disc_real_output)

    generated_loss = loss_object(tf.zeros_like(disc_generated_output), disc_generated_output)

    total_disc_loss = real_loss + generated_loss

    return total_disc_loss

generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
discriminator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

checkpoint_dir = './training_checkpoints/model4'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 generator=generator,
                                 discriminator=discriminator)

img_num = 0
def generate_images(model, test_input, tar):
    prediction = model(test_input, training=True)
    # print(np.array((prediction[0] + 1) *127.5, dtype=np.uint8))
    # b[np.arange(len(a)), np.argmax(a, 1)] = 1
    # b[np.arange(len(a)), np.argmax(a, 1)] = 1

    def construct_image(tensor):
        a= np.array((tensor[0]))
        pixels = []
        h = len(a)
        w = len(a[0])
        for c in range(h):
            row = []
            for r in range(w):
                b = np.zeros(4)
                b[np.argmax(a[c][r], 0)] = 1
                row.append(np.dot(b, np.array([[50, 168, 82],[237, 64, 17],[20, 103, 219],[255, 48, 224]])))
            pixels.append(row)
        return np.array(pixels, dtype=np.uint8)

    test_input = np.array((test_input[0] + 1) *127.5, dtype=np.uint8)[:,:,:3]
    global img_num
    img = Image.fromarray(np.concatenate((test_input, construct_image(tar), construct_image(prediction))))
    img.save(f"./history/{img_num:d}.png")
    img_num+=1



# for example_input, example_target in dataset.take(1):
#     generate_images(generator, example_input, example_target)




@tf.function
def train_step(input_image, target, epoch):
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        gen_output = generator(input_image, training=True)

        disc_real_output = discriminator([input_image, target], training=True)
        disc_generated_output = discriminator([input_image, gen_output], training=True)

        gen_total_loss, gen_gan_loss, gen_l1_loss = generator_loss(disc_generated_output, gen_output, target)
        disc_loss = discriminator_loss(disc_real_output, disc_generated_output)

    generator_gradients = gen_tape.gradient(gen_total_loss,
                                            generator.trainable_variables)
    discriminator_gradients = disc_tape.gradient(disc_loss,
                                                discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(generator_gradients,
                                            generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(discriminator_gradients,
                                                discriminator.trainable_variables))

    with summary_writer.as_default():
        tf.summary.scalar('gen_total_loss', gen_total_loss, step=epoch)
        tf.summary.scalar('gen_gan_loss', gen_gan_loss, step=epoch)
        tf.summary.scalar('gen_l1_loss', gen_l1_loss, step=epoch)
        tf.summary.scalar('disc_loss', disc_loss, step=epoch)

# for i in range(10):
#    for example_input, example_target in dataset.take(1):
#         # print(example_target)
#         generate_images(generator, example_input, example_target)
# e_i, e_t = load("./clothing-co-parsing/photos/0001.jpg","./clothing-co-parsing/labels/0001.csv")
# e_i, e_t = random_jitter(e_i,e_t)
# e_i, e_t =  load_image_train("./clothing-co-parsing/photos/0001.jpg","./clothing-co-parsing/labels/0001.csv")
# generate_images(generator, e_i, e_t)

def fit(train_ds, epochs):
    for epoch in range(1, epochs+1):
        start = time.time()

        display.clear_output(wait=True)

        for example_input, example_target in train_ds.take(1):
            generate_images(generator, example_input, example_target)
        print("Epoch: ", epoch)

        # Training step
        for n, (input_image, target) in train_ds.enumerate():
            print('.', end='')
            if (n+1) % 100 == 0:
                print(f" {n+1:-4d}")
            train_step(input_image, target, epoch)
        print()

        # Saving (checkpointing) the model every epoch
        checkpoint.save(file_prefix=checkpoint_prefix)

        print ('Time taken for epoch {} is {} sec\n'.format(epoch + 1,
                                                            time.time()-start))
        checkpoint.save(file_prefix=checkpoint_prefix)

fit(dataset, EPOCHS)

# discriminator = Discriminator()
# tf.keras.utils.plot_model(discriminator, show_shapes=True, dpi=64)
