from __future__ import print_function, division
import scipy
from keras.datasets import mnist
from keras_contrib.layers.normalization.instancenormalization import InstanceNormalization
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, Concatenate
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam
import datetime
import matplotlib.pyplot as plt
import sys
from glob import glob
import numpy as np
import os

class DataLoader():
    def __init__(self, dataname, res=(128, 128)):
        self.dataset_name = dataname
        self.img_res = res

    def loadimages(self, domain, batch_size=1, is_testing=False):
        data_type = "train%s" % domain if not is_testing else "test%s" % domain
        path = glob('./datasets/%s/%s/*' % (self.dataset_name, data_type))

        batch_images = np.random.choice(path, size=batch_size)

        imgs = []
        for img_path in batch_images:
            img = self.imread(img_path)
            if not is_testing:
                img = scipy.misc.imresize(img, self.img_res)

                if np.random.random() > 0.5:
                    img = np.fliplr(img)
            else:
                img = scipy.misc.imresize(img, self.img_res)
            imgs.append(img)

        imgs = np.array(imgs)/127.5 - 1.

        return imgs

    def loadbatch(self, batch_size=1, is_testing=False):
        data_type = "train" if not is_testing else "val"
        path_A = glob('./datasets/%s/%sA/*' % (self.dataset_name, data_type))
        path_B = glob('./datasets/%s/%sB/*' % (self.dataset_name, data_type))

        self.n_batches = int(min(len(path_A), len(path_B)) / batch_size)
        total_samples = self.n_batches * batch_size

        # Sample n_batches * batch_size from each path list so that model sees all
        # samples from both domains
        path_A = np.random.choice(path_A, total_samples, replace=False)
        path_B = np.random.choice(path_B, total_samples, replace=False)

        for i in range(self.n_batches-1):
            batch_A = path_A[i*batch_size:(i+1)*batch_size]
            batch_B = path_B[i*batch_size:(i+1)*batch_size]
            imgs_A, imgs_B = [], []
            for img_A, img_B in zip(batch_A, batch_B):
                img_A = self.imread(img_A)
                img_B = self.imread(img_B)

                img_A = scipy.misc.imresize(img_A, self.img_res)
                img_B = scipy.misc.imresize(img_B, self.img_res)

                if not is_testing and np.random.random() > 0.5:
                        img_A = np.fliplr(img_A)
                        img_B = np.fliplr(img_B)

                imgs_A.append(img_A)
                imgs_B.append(img_B)

            imgs_A = np.array(imgs_A)/127.5 - 1.
            imgs_B = np.array(imgs_B)/127.5 - 1.

            yield imgs_A, imgs_B

    def loadimg(self, path):
        img = self.imread(path)
        img = scipy.misc.imresize(img, self.img_res)
        img = img/127.5 - 1.
        return img[np.newaxis, :, :, :]

    def imread(self, path):
        return scipy.misc.imread(path, mode='RGB').astype(np.float)


class CycleGAN():
	def __init__(self):
		self.irows = 128
		self.icols = 128
		self.chann = 3
		self.ishape = (self.irows, self.icols, self.chann)
		self.dataname = 'night2day'
		self.dataload = DataLoader(dataname=self.dataname, res=(self.irows, self.icols))
		patch = int(self.irows/2**4)
		self.disc = (patch, patch, 1)
		
		self.gf = 32
		self.df = 64

		self.lambdacycle = 10.0
		self.lambdaid = 0.1 * self.lambdacycle

		optimizer = Adam(0.0002, 0.5)

		#DISCRIMINATOR
		self.dA = self.startdiscriminator()
		self.dB = self.startdiscriminator()
		
		self.dA.compile(loss='mse', optimizer=optimizer, metrics=['accuracy'])

		self.dB.compile(loss='mse', optimizer=optimizer, metrics=['accuracy'])
		#END DISCRIMINATOR
		#GENERATOR
		self.gAB = self.startgen()
		self.gBA = self.startgen()
		
		imgsA = Input(shape=self.ishape)
		imgsB = Input(shape=self.ishape)

		fakeB = self.gAB(imgsA)
		fakeA = self.gBA(imgsB)

		reconA = self.gBA(fakeB)
		reconB = self.gAB(fakeA)
		
		imgsAid = self.gBA(imgsA)
		imgsBid = self.gAB(imgsB)
		#END GENERATOR

		self.dA.train = False
		self.dB.train = False

		validA = self.dA(fakeA)
		validB = self.dB(fakeB)
		
		self.combined = Model(inputs=[imgsA, imgsB],outputs=[validA, validB, reconA, reconB,imgsAid, imgsBid ])

		self.combined.compile(loss=['mse', 'mse','mae','mae','mae', 'mae'], loss_weights=[1, 1,self.lambdacycle, self.lambdacycle,self.lambdaid, self.lambdaid],optimizer=optimizer)

	'''def startgen(self):
		#UNET GENERATOR
		def conv(layerin, filters, fsize=4):
			lay = Conv2D(filters, kernal_size=fsize, strides=2, padding='same')(layerin)
			lay = LeakyReLu(alpha=0.2)(lay)
			lay = InstanceNormalization()(lay)
			return lay
		
		def deconv(layerin, skipin, filters, fsize=4, dropout=0):
			lay = UpSampling2D(size=2)(layerin)
			lay = Conv2D(filters, kernal_size=fsize, strides=1, padding='same', activation='relu')(lay)
			if dropout:
				lay = Dropout(dropout)(lay)
			lay = InstanceNormalization()(lay)
			lay = Concatenate()([lay,skipin])
			return lay
		
		imin = Input(shape=self.ishape)

		lay1 = conv(imin, self.gf)
		lay2 = conv(lay1, self.gf*2)
		lay3 = conv(lay2, self.gf*4)
		lay4 = conv(lay3, self.gf*8)

		up1 = deconv(lay4, lay3, self.gf*4)
		up2 = deconv(up1, lay2, self.gf*2)
		up3 = deconv(up2, lay1, self.gf)
	
		up4 = UpSampling2D(size=2)(up3)
		output = Conv2D(self.chann,kernal_size=4, strides=1, padding='same',activation='tanh')(up4)
		return Model(imin, output)'''

	def startgen(self):
		"""U-Net Generator"""

		def conv2d(layer_input, filters, f_size=4):
		    """Layers used during downsampling"""
		    d = Conv2D(filters, kernel_size=f_size, strides=2, padding='same')(layer_input)
		    d = LeakyReLU(alpha=0.2)(d)
		    d = InstanceNormalization()(d)
		    return d

		def deconv2d(layer_input, skip_input, filters, f_size=4, dropout_rate=0):
		    """Layers used during upsampling"""
		    u = UpSampling2D(size=2)(layer_input)
		    u = Conv2D(filters, kernel_size=f_size, strides=1, padding='same', activation='relu')(u)
		    if dropout_rate:
		        u = Dropout(dropout_rate)(u)
		    u = InstanceNormalization()(u)
		    u = Concatenate()([u, skip_input])
		    return u

		# Image input
		d0 = Input(shape=self.ishape)

		# Downsampling
		d1 = conv2d(d0, self.gf)
		d2 = conv2d(d1, self.gf*2)
		d3 = conv2d(d2, self.gf*4)
		d4 = conv2d(d3, self.gf*8)

		# Upsampling
		u1 = deconv2d(d4, d3, self.gf*4)
		u2 = deconv2d(u1, d2, self.gf*2)
		u3 = deconv2d(u2, d1, self.gf)

		u4 = UpSampling2D(size=2)(u3)
		output_img = Conv2D(self.chann, kernel_size=4, strides=1, padding='same', activation='tanh')(u4)

		return Model(d0, output_img)

	'''def startdiscriminator(self):
		
		def dlayer(layerin, filt, fsize=4, norm = True):
			lay = Conv2D(filt, kernal_size=fsize, strides = 2, padding = 'same')(layerin)
			lay = LeakyReLU(alpha=0.2)(lay)
			if norm:
				lay = InstanceNormalization()(lay)
			return lay

		img = Input(shape=self.ishape)
		
		lay1 = dlayer(img, self.df, norm = False)
		lay2 = dlayer(lay1, self.df*2)
		lay3 = dlayer(lay2, self.df*4)
		lay4 = dlayer(lay3, self.df*8)
		valid = Conv2D(1, kernal_size=4, strides=1, padding='same',activation='tanh')(lay4)

		return Model(img, valid)'''
	def startdiscriminator(self):
		def d_layer(layer_input, filters, f_size=4, normalization=True):
		    """Discriminator layer"""
		    d = Conv2D(filters, kernel_size=f_size, strides=2, padding='same')(layer_input)
		    d = LeakyReLU(alpha=0.2)(d)
		    if normalization:
		        d = InstanceNormalization()(d)
		    return d

		img = Input(shape=self.ishape)

		d1 = d_layer(img, self.df, normalization=False)
		d2 = d_layer(d1, self.df*2)
		d3 = d_layer(d2, self.df*4)
		d4 = d_layer(d3, self.df*8)

		validity = Conv2D(1, kernel_size=4, strides=1, padding='same')(d4)

		return Model(img, validity)

	def train(self, epochs, batch=1, interval=50):
		start = datetime.datetime.now()

		valid = np.ones((batch,) + self.disc)
		fake = np.zeros((batch,) + self.disc)

		for epoch in range(epochs):
			for batchi, (imgsA, imgsB) in enumerate(self.dataload.loadbatch(batch)):
				fakeB = self.gAB.predict(imgsA)
				fakeA = self.gBA.predict(imgsB)

				dAlossreal = self.dA.train_on_batch(imgsA, valid)
				dAlossfake = self.dB.train_on_batch(fakeA, fake)
				dAloss = 0.5 * np.add(dAlossreal, dAlossfake)
				dBlossreal = self.dB.train_on_batch(imgsB, valid)
				dBlossfake = self.dB.train_on_batch(fakeB, fake)
				dBloss = 0.5 * np.add(dBlossreal, dBlossfake)
				dloss = 0.5 * np.add(dAloss, dBloss)

				gloss = self.combined.train_on_batch([imgsA, imgsB], [valid, valid, imgsA, imgsB, imgsA, imgsB])

				timetaken = datetime.datetime.now() - start

				print ("[Epoch %d/%d] [Batch %d/%d] [D loss: %f, acc: %3d%%] [G loss: %05f, adv: %05f, recon: %05f, id: %05f] time: %s " % ( epoch, epochs,batchi, self.dataload.n_batches,dloss[0], 100*dloss[1],gloss[0],np.mean(gloss[1:3]),np.mean(gloss[3:5]),np.mean(gloss[5:6]),timetaken))
				if batchi % interval == 0:
					self.isample(epoch, batchi)

	def isample(self, epoch, batchi):
		try:
			os.makedirs('images/%s' % self.dataname)
		except OSError, e:
			if e.errno != os.errno.EEXIST:
				raise
			pass
		
		r, c = 2, 3	
		imgsA = self.dataload.loadimages(domain="A", batch_size = 1, is_testing = True)
		imgsB = self.dataload.loadimages(domain="B", batch_size = 1, is_testing = True)

		fakeB = self.gAB.predict(imgsA)
		fakeA = self.gBA.predict(imgsB)

		reconstrA = self.gBA.predict(fakeB)
		reconstrB = self.gAB.predict(fakeA)
		
		genimgs = np.concatenate([imgsA, fakeB, reconstrA, imgsB, fakeA, reconstrB])

		genimgs = 0.5 * genimgs + 0.5

		titles = ['Original', 'Translated', 'Reconstructed']

		fig, axis = plt.subplots(r, c)
		
		cnt = 0
		
		for i in range(r):
			for j in range(c):
				axis[i, j].imshow(genimgs[cnt])
				axis[i, j].axis('off')
				cnt += 1
		fig.savefig("images/%s/%d_%d.png" % (self.dataname, epoch, batchi))
		plt.close()
		
if __name__ == '__main__':
	gan = CycleGAN()
	gan.train(epochs=200, batch=1, interval=200)
