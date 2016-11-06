from keras.layers import Input
from keras.models import Model
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator

import sys
sys.path.append("..")

import models
from loss import PSNRLoss, psnr

import os
import time
import numpy as np
from scipy.misc import imresize, imsave
from scipy.ndimage.filters import gaussian_filter

base_weights_path = "weights/"
base_val_images_path = "val_images/"
base_test_images = "test_images/"

set5_path = r"set5/"
set14_path = r"set14/"
bsd100_path = r"bsd100/"

if not os.path.exists(base_weights_path):
    os.makedirs(base_weights_path)

if not os.path.exists(base_val_images_path):
    os.makedirs(base_val_images_path)

if not os.path.exists(base_test_images):
    os.makedirs(base_test_images)

def test_set5(model : Model, img_width=32, img_height=32, batch_size=1):
    datagen = ImageDataGenerator(rescale=1. / 255)
    large_img_width = img_width * 4
    large_img_height = img_height * 4

    iteration = 0
    total_psnr = 0.0

    print("Testing model on Set 5 Validation images")
    total_psnr = _test_loop(set5_path, batch_size, datagen, img_height, img_width, iteration, large_img_height, large_img_width,
                            model, total_psnr, "set5", 5)

    print("Average PSNR of Set5 validation images : ", total_psnr / 5)
    print()


def test_set14(model : Model, img_width=32, img_height=32, batch_size=1):
    datagen = ImageDataGenerator(rescale=1. / 255)
    large_img_width = img_width * 4
    large_img_height = img_height * 4

    iteration = 0
    total_psnr = 0.0

    print("Testing model on Set 14 Validation images")
    total_psnr = _test_loop(set14_path, batch_size, datagen, img_height, img_width, iteration, large_img_height,
                            large_img_width, model, total_psnr, "set14", 14)

    print("Average PSNR of Set5 validation images : ", total_psnr / 14)
    print()

def test_bsd100(model : Model, img_width=32, img_height=32, batch_size=1):
    datagen = ImageDataGenerator(rescale=1. / 255)
    large_img_width = img_width * 4
    large_img_height = img_height * 4

    iteration = 0
    total_psnr = 0.0

    print("Testing model on BSD 100 Validation images")
    total_psnr = _test_loop(bsd100_path, batch_size, datagen, img_height, img_width, iteration, large_img_height, large_img_width,
                            model, total_psnr, "bsd100", 100)

    print("Average PSNR of BSD100 validation images : ", total_psnr / 100)
    print()


def _test_loop(path, batch_size, datagen, img_height, img_width, iteration, large_img_height, large_img_width, model,
               total_psnr, prefix, nb_images):
    for x in datagen.flow_from_directory(path, class_mode=None, batch_size=batch_size,
                                         target_size=(large_img_width, large_img_height)):
        t1 = time.time()

        # resize images
        x_temp = x.copy()
        x_temp = x_temp.transpose((0, 2, 3, 1))

        x_generator = np.empty((batch_size, img_width, img_height, 3))

        for j in range(batch_size):
            img = imresize(x_temp[j], (img_width, img_height))
            x_generator[j, :, :, :] = img

        x_generator = x_generator.transpose((0, 3, 1, 2))

        output_image_batch = model.predict_on_batch(x_generator)

        average_psnr = 0.0
        for x_i in range(batch_size):
            average_psnr += psnr(x[x_i], output_image_batch[x_i] / 255.)
            total_psnr += average_psnr

        average_psnr /= batch_size

        iteration += batch_size
        t2 = time.time()

        print("Time required : %0.2f. Average validation PSNR over %d samples = %0.2f" %
              (t2 - t1, batch_size, average_psnr))

        for x_i in range(batch_size):
            real_path = base_test_images + prefix + "_iteration_%d_num_%d_real_.png" % (iteration, x_i + 1)
            generated_path = base_test_images + prefix + "_iteration_%d_num_%d_generated.png" % (iteration, x_i + 1)

            val_x = x[x_i].copy() * 255.
            val_x = val_x.transpose((1, 2, 0))
            val_x = np.clip(val_x, 0, 255).astype('uint8')

            output_image = output_image_batch[x_i]
            output_image = output_image.transpose((1, 2, 0))
            output_image = np.clip(output_image, 0, 255).astype('uint8')

            imsave(real_path, val_x)
            imsave(generated_path, output_image)

        if iteration >= nb_images:
            break
    return total_psnr


class SRResNetTest:

    def __init__(self, img_width=96, img_height=96, batch_size=16):
        assert img_width >= 16, "Minimum image width must be at least 16"
        assert img_height >= 16, "Minimum image height must be at least 16"

        self.img_width = img_width
        self.img_height = img_height
        self.batch_size = batch_size

        self.model = None # type: Model
        self.weights_path = base_weights_path + "sr_resnet_weights.h5"

    def build_model(self, load_weights=False) -> Model:
        sr_resnet = models.GenerativeNetwork(self.img_width, self.img_height, self.batch_size)

        ip = Input(shape=(3, self.img_width, self.img_height), name='x_generator')
        output = sr_resnet.create_sr_model(ip)

        self.model = Model(ip, output)

        optimizer = Adam(lr=1e-4)
        self.model.compile(optimizer, loss='mse', metrics=[PSNRLoss])

        if load_weights:
            try:
                self.model.load_weights(self.weights_path)
                print("SR ResNet model weights loaded.")
            except Exception:
                print("Weight for SR ResNet model not found or are incorrect size. Cannot load weights.")

                response = input("Continue without loading weights? 'y' or 'n' ")
                if response == 'n':
                    exit()

        return self.model

    def train_model(self, image_dir, nb_images=50000, nb_epochs=1):
        datagen = ImageDataGenerator(rescale=1. / 255)
        img_width = self.img_width * 4
        img_height = self.img_height * 4

        early_stop = False
        iteration = 0
        prev_improvement = -1

        print("Training SR ResNet network")
        for i in range(nb_epochs):
            print()
            print("Epoch : %d" % (i + 1))

            for x in datagen.flow_from_directory(image_dir, class_mode=None, batch_size=self.batch_size,
                                                 target_size=(img_width, img_height)):

                try:
                    t1 = time.time()

                    # resize images
                    x_temp = x.copy()
                    x_temp = x_temp.transpose((0, 2, 3, 1))

                    x_generator = np.empty((self.batch_size, self.img_width, self.img_height, 3))

                    for j in range(self.batch_size):
                        img = gaussian_filter(x_temp[j], sigma=0.5)
                        img = imresize(img, (self.img_width, self.img_height))
                        x_generator[j, :, :, :] = img

                    x_generator = x_generator.transpose((0, 3, 1, 2))

                    if iteration % 50 == 0 and iteration != 0 :
                        print("Random Validation image..")
                        output_image_batch = self.model.predict_on_batch(x_generator)

                        print("Pred Max / Min: %0.2f / %0.2f" % (output_image_batch.max(),
                                                                 output_image_batch.min()))

                        average_psnr = 0.0
                        for x_i in range(self.batch_size):
                            average_psnr += psnr(x[x_i], output_image_batch[x_i] / 255.)

                        average_psnr /= self.batch_size

                        iteration += self.batch_size
                        t2 = time.time()

                        print("Time required : %0.2f. Average validation PSNR over %d samples = %0.2f" %
                              (t2 - t1, self.batch_size, average_psnr))

                        for x_i in range(self.batch_size):
                            real_path = base_val_images_path + "epoch_%d_iteration_%d_num_%d_real_.png" % \
                                                               (i + 1, iteration, x_i + 1)

                            generated_path = base_val_images_path + \
                                             "epoch_%d_iteration_%d_num_%d_generated.png" % (i + 1,
                                                                                            iteration,
                                                                                            x_i + 1)

                            val_x = x[x_i].copy() * 255.
                            val_x = val_x.transpose((1, 2, 0))
                            val_x = np.clip(val_x, 0, 255).astype('uint8')

                            output_image = output_image_batch[x_i]
                            output_image = output_image.transpose((1, 2, 0))
                            output_image = np.clip(output_image, 0, 255).astype('uint8')

                            imsave(real_path, val_x)
                            imsave(generated_path, output_image)

                        '''
                        Don't train of validation images for now.

                        Note that if nb_epochs > 1, there is a chance that
                        validation images may be used for training purposes as well.

                        In that case, this isn't strictly a validation measure, instead of
                        just a check to see what the network has learned.
                        '''
                        continue

                    hist = self.model.fit(x_generator, x * 255, batch_size=self.batch_size, nb_epoch=1, verbose=0)
                    psnr_loss_val = hist.history['PSNRLoss'][0]

                    if prev_improvement == -1:
                        prev_improvement = psnr_loss_val

                    improvement = (prev_improvement - psnr_loss_val) / prev_improvement * 100
                    prev_improvement = psnr_loss_val

                    iteration += self.batch_size
                    t2 = time.time()

                    print("Iter : %d / %d | Improvement : %0.2f percent | Time required : %0.2f seconds | "
                          "PSNR : %0.3f" % (iteration, nb_images, improvement, t2 - t1, psnr_loss_val))

                    if iteration % 1000 == 0 and iteration != 0:
                        print("Saving weights")
                        self.model.save_weights(self.weights_path, overwrite=True)

                    if iteration >= nb_images:
                        break

                except KeyboardInterrupt:
                    print("Keyboard interrupt detected. Stopping early.")
                    early_stop = True
                    break

            iteration = 0

            if early_stop:
                break

        print("Finished training SRGAN network. Saving model weights.")


if __name__ == "__main__":
    from keras.utils.visualize_util import plot

    coco_path = r"D:\Yue\Documents\Dataset\coco2014\train2014"

    img_width = img_height = 64

    sr_resnet_test = SRResNetTest(img_width=img_width, img_height=img_height, batch_size=1)
    sr_resnet_test.build_model(load_weights=False)
    #plot(sr_resnet_test.model, to_file='sr_resnet.png', show_shapes=True)

    sr_resnet_test.train_model(coco_path, nb_images=50000, nb_epochs=1)

    test_set5(sr_resnet_test.model, img_width=img_width, img_height=img_height)
    test_set14(sr_resnet_test.model, img_width=img_width, img_height=img_height)
    test_bsd100(sr_resnet_test.model, img_width=img_width, img_height=img_height)
