from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras import backend as keras
import numpy as np 
import os
import glob
import cv2

def merge_and_save():
	imgtype = "jpg"
	train = glob.glob("results/*."+imgtype)
	for i in range(len(train)):
		if i is not 54 and i is not 72:
			img_t = load_img("test/"+str(i)+"."+imgtype)
			img_l = load_img("test/"+str(i)+"_l."+imgtype)
			img_p = load_img("results/"+str(i)+"."+imgtype)
			
			x_t = img_to_array(img_t)
			x_l = img_to_array(img_l)
			x_t[:,:,2] = x_l[:,:,0]
			img_tmp = array_to_img(x_t)
			img_tmp.save("merged/"+str(i)+"."+imgtype)
			x_tp = img_to_array(img_t)
			x_p = img_to_array(img_p)
			x_tp[:,:,2] = x_p[:,:,0]
			img_tmp = array_to_img(x_tp)
			img_tmp.save("merged/"+str(i)+"_p."+imgtype)
			'''
			x_l = img_to_array(img_l)
			x_p = img_to_array(img_p)
			tmp = np.asarray(x_p).astype(np.bool)
			img_tmp = array_to_img(tmp)
			img_tmp.save("bool/"+str(i)+"."+imgtype)
			tmp = np.asarray(x_l).astype(np.bool)
			img_tmp = array_to_img(tmp)
			img_tmp.save("bool/"+str(i)+"_t."+imgtype)
			'''

def dice_coef(gt, seg):
	gt = np.asarray(gt).astype(np.bool)
	seg = np.asarray(seg).astype(np.bool)
	intersection = np.logical_and(gt, seg)
	return intersection.sum()*2.0 / (np.sum(seg) + np.sum(gt))


def calculate_dice():
	imgtype = "jpg"
	train = glob.glob("results/*."+imgtype)
	dice_sum = 0
	for i in range(len(train)):
		if i is not 54 and i is not 72:
			img_l = load_img("test/"+str(i)+"_l."+imgtype)
			img_p = load_img("results/"+str(i)+"."+imgtype)
			x_l = img_to_array(img_l)
			x_p = img_to_array(img_p)
			dice = dice_coef(x_l, x_p)
			dice_sum += dice
			print(i)
			print(dice)
	print(dice_sum / len(train))



if __name__ == "__main__":
	merge_and_save()
	calculate_dice()