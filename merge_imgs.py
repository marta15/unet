import numpy as np 
import os
import glob
import cv2

def merge_and_save():
	imgtype = "jpg"
	train = glob.glob("results/*."+imgtype)
	for i in range(len(train)):
				img_t = load_img("test/"+str(i)+"."+imgtype)
				img_l = load_img("test/"+str(i)+"_l."+imgtype)
				img_p = load_img("results/"+str(i)+".")
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


if __name__ == "__main__":
	merge_and_save()