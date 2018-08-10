import os 
#os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
from keras.models import *
from keras.models import load_model
from keras.layers import Input, merge, Conv2D, MaxPooling2D, UpSampling2D, Dropout, Cropping2D
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping
from keras import backend as keras
from data import *


class myUnet(object):

	def __init__(self, img_rows = 128, img_cols = 128):

		self.img_rows = img_rows
		self.img_cols = img_cols

	def load_data(self):

		mydata = dataProcess(self.img_rows, self.img_cols)
		imgs_train, imgs_mask_train = mydata.load_train_data()
		imgs_test = mydata.load_test_data()
		return imgs_train, imgs_mask_train, imgs_test

	def get_unet(self):

		inputs = Input((self.img_rows, self.img_cols,1))
		
		
		conv1 = Conv2D(16, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
		print "conv1 shape:",conv1.shape
		conv1 = Conv2D(16, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
		print "conv1 shape:",conv1.shape
		pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
		print "pool1 shape:",pool1.shape

		conv2 = Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
		print "conv2 shape:",conv2.shape
		conv2 = Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
		print "conv2 shape:",conv2.shape
		pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
		print "pool2 shape:",pool2.shape

		conv3 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
		print "conv3 shape:",conv3.shape
		conv3 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
		print "conv3 shape:",conv3.shape
		pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
		print "pool3 shape:",pool3.shape

		conv4 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
		conv4 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
		drop4 = Dropout(0.5)(conv4)
		pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

		conv5 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool4)
		conv5 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)
		drop5 = Dropout(0.5)(conv5)

		up6 = Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(drop5))
		merge6 = merge([drop4,up6], mode = 'concat', concat_axis = 3)
		conv6 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
		conv6 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)

		up7 = Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv6))
		merge7 = merge([conv3,up7], mode = 'concat', concat_axis = 3)
		conv7 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)
		conv7 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)

		up8 = Conv2D(32, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv7))
		merge8 = merge([conv2,up8], mode = 'concat', concat_axis = 3)
		conv8 = Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8)
		conv8 = Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)

		up9 = Conv2D(16, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv8))
		merge9 = merge([conv1,up9], mode = 'concat', concat_axis = 3)
		conv9 = Conv2D(16, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge9)
		conv9 = Conv2D(16, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
		conv9 = Conv2D(2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
		conv10 = Conv2D(1, 1, activation = 'sigmoid')(conv9)

		model = Model(input = inputs, output = conv10)

		model.compile(optimizer = Adam(lr = 1e-6), loss = 'binary_crossentropy', metrics = ['binary_accuracy'])

		return model


	def train(self):

		print("loading data")
		imgs_train, imgs_mask_train, imgs_test = self.load_data()
		mean = np.mean(imgs_train)  # mean for data centering
		std = np.std(imgs_train)  # std for data normalization
		
		imgs_train -= mean
		imgs_train /= std

		imgs_mask_train /= 255. # scale masks to [0, 1]

		imgs_test -= mean
		imgs_test /= std

		for i in range(5):
			img = imgs_train[i]
			img = array_to_img(img)
			img.save("./train_imgs/%d.jpg"%(i))
		for i in range(5):
			img = imgs_mask_train[i]
			img = array_to_img(img)
			img.save("./train_imgs/%d_y.jpg"%(i))
		for i in range(5):
			img = imgs_test[i]
			img = array_to_img(img)
			img.save("./val_imgs/%d.jpg"%(i))

		print("loading data done")
		model = self.get_unet()
		#model = load_model('unet.hdf5')
		print("got unet")

		
		early_stopping = EarlyStopping(monitor='val_loss', min_delta=1e-6, patience=5, verbose=0, mode='auto')
		model_checkpoint = ModelCheckpoint('unet.hdf5', monitor='val_loss',verbose=1, save_best_only=True)
		print('Fitting model...')
		model.fit(imgs_train, imgs_mask_train, batch_size=32, epochs=100, verbose=1,validation_split=0.2, shuffle=True, callbacks=[model_checkpoint, early_stopping])

		print('predict test data')
		imgs_mask_test = model.predict(imgs_test, batch_size=1, verbose=1)
		np.save('./results/imgs_mask_test.npy', imgs_mask_test)

	def save_img(self):

		print("array to image")
		imgs = np.load('imgs_mask_test.npy')
		for i in range(imgs.shape[0]):
			img = imgs[i]
			img = array_to_img(img)
			img.save("./results/%d.jpg"%(i))




if __name__ == '__main__':
	myunet = myUnet()
	myunet.train()
	myunet.save_img()








