import numpy as np
import scipy
import cv2
import glob
import argparse
import pdb
import sys
#sys.path.append('../../../../../train_src/analysis/')
import pathlib
from PredictionsLoader import PredictionsLoaderNPY, PredictionsLoaderModel
from utils import seq_add_padding, add_padding

parser = argparse.ArgumentParser(description='')
parser.add_argument('-ds', '--dataset', dest='dataset',
					default='cv', help='t len')
parser.add_argument('-mdl', '--model', dest='model_type',
					default='densenet', help='t len')

a = parser.parse_args()

dataset=a.dataset
model_type=a.model_type

dataset='cv'
model_type='unet'
skip_crf=True

def dense_crf(probs, img=None, n_iters=10, n_classes=19,
			  sxy_gaussian=(1, 1), compat_gaussian=4,
			  sxy_bilateral=(49, 49), compat_bilateral=5,
			  srgb_bilateral=(13, 13, 13)):
	import pydensecrf.densecrf as dcrf
	from pydensecrf.utils import create_pairwise_bilateral
	_, h, w, _ = probs.shape

	probs = probs[0].transpose(2, 0, 1).copy(order='C')	 # Need a contiguous array.

	d = dcrf.DenseCRF2D(w, h, n_classes)  # Define DenseCRF model.
	U = -np.log(probs)	# Unary potential.
	U = U.reshape((n_classes, -1))	# Needs to be flat.
	d.setUnaryEnergy(U)
	d.addPairwiseGaussian(sxy=sxy_gaussian, compat=compat_gaussian,
						  kernel=dcrf.DIAG_KERNEL, normalization=dcrf.NORMALIZE_SYMMETRIC)
	if img is not None:
		assert (img.shape[1:3] == (h, w)), "The image height and width must coincide with dimensions of the logits."
		pairwise_bilateral = create_pairwise_bilateral(sdims=(10,10), schan=(0.01), img=img[0], chdim=2)
		d.addPairwiseEnergy(pairwise_bilateral, compat=10)

	Q = d.inference(n_iters)
	preds = np.array(Q, dtype=np.float32).reshape((n_classes, h, w)).transpose(1, 2, 0)
	return np.expand_dims(preds, 0)

def patch_file_id_order_from_folder(folder_path):
	paths=glob.glob(folder_path+'*.npy')
	print(paths[:10])
	order=[int(paths[i].partition('patch_')[2].partition('_')[0]) for i in range(len(paths))]
	print(len(order))
	print(order[0:20])
	return order

path='../model/'

data_path='../../../../../dataset/dataset/'

if dataset=='lm':

	path+='lm/'
	if model_type=='unet':
		predictions_path=path+'model_best_UNet3D_ATPP4_f32_1.h5'
		#predictions_path=path+'prediction_BUnet4ConvLSTM_repeating2.npy'
		#predictions_path=path+'prediction_BUnet4ConvLSTM_repeating4.npy'


	mask_path=data_path+'lm_data/TrainTestMask.tif'
	location_path=data_path+'lm_data/locations/'
	folder_load_path=data_path+'lm_data/train_test/test/labels/'

	custom_colormap = np.array([[255,146,36],
					[255,255,0],
					[164,164,164],
					[255,62,62],
					[0,0,0],
					[172,89,255],
					[0,166,83],
					[40,255,40],
					[187,122,83],
					[217,64,238],
					[0,113,225],
					[128,0,0],
					[114,114,56],
					[53,255,255]])
elif dataset=='cv':

	path+='cv/'
	if model_type=='unet':
		#predictions_path=path+'prediction_BUnet4ConvLSTM_repeating2.npy'
		predictions_path=path+'model_best_UNet3D_ATPP4_f32_1.h5'

	mask_path=data_path+'cv_data/TrainTestMask.tif'
	location_path=data_path+'cv_data/locations/'
	fullimg_path=data_path+'cv_data/full_ims/'

	folder_load_path=data_path+'cv_data/train_test/test/labels/'

	custom_colormap = np.array([[255, 146, 36],
				   [255, 255, 0],
				   [164, 164, 164],
				   [255, 62, 62],
				   [0, 0, 0],
				   [172, 89, 255],
				   [0, 166, 83],
				   [40, 255, 40],
				   [187, 122, 83],
				   [217, 64, 238],
				   [45, 150, 255]])

print("Loading patch locations...")
order_id_load=False
if order_id_load==False:
	order_id=patch_file_id_order_from_folder(folder_load_path)
	np.save('order_id.npy',order_id)
else:
	order_id=np.load('order_id.npy')

cols=np.load(location_path+'locations_col.npy')
rows=np.load(location_path+'locations_row.npy')

print(cols.shape, rows.shape)
cols=cols[order_id]
rows=rows[order_id]

# ======== load labels and predictions

#labels=np.load(path+'labels.npy').argmax(axis=4)
#predictions=np.load(predictions_path).argmax(axis=4)

print("Loading labels and predictions...")

prediction_type = 'model'
results_path="../"
#path=results_path+dataset+'/'
#prediction_path=path+predictions_path
path_test='../../../../../dataset/dataset/'+dataset+'_data/patches_bckndfixed/test/'
print('path_test',path_test)

#prediction_type = 'model'
if prediction_type=='npy':
	predictionsLoader = PredictionsLoaderNPY()
	predictions, labels = predictionsLoader.loadPredictions(predictions_path,path+'labels.npy')
elif prediction_type=='model':
	#model_path=results_path + 'model/'+dataset+'/'+prediction_filename
	print('model_path',predictions_path)

	predictionsLoader = PredictionsLoaderModel(path_test)
	_, labels, model = predictionsLoader.loadPredictions(predictions_path)


#================= load labels and predictions






#class_n=np.max(predictions)+1
#print("class_n",class_n)
#labels[labels==class_n]=255 # background

# Print stuff
#print(cols.shape)
#print(rows.shape)
#print(labels.shape)
#print(predictions.shape)
#print("np.unique(labels,return_counts=True)",
#	np.unique(labels,return_counts=True))
#print("np.unique(predictions,return_counts=True)",
#	np.unique(predictions,return_counts=True))

# Specify variables
#sequence_len=labels.shape[1]
#patch_len=labels.shape[2]

# Load mask
mask=cv2.imread(mask_path,-1)
mask[mask==1]=0 # training as background
print("Mask shape",mask.shape)
#print((sequence_len,)+mask.shape)

# ================= LOAD THE INPUT IMAGE.
full_path = '../../../../../dataset/dataset/cv_data/full_ims/'
full_ims_test = np.load(full_path+'full_ims_test.npy')
full_label_test = np.load(full_path+'full_label_test.npy').astype(np.uint8)

# convert labels; background is last
class_n=len(np.unique(full_label_test))-1
full_label_test=full_label_test-1
full_label_test[full_label_test==255]=class_n

print(full_ims_test.shape)
print(full_label_test.shape)

# Reconstruct the image
print("Reconstructing the labes and predictions...")

patch_size=32
mask_pad=mask.copy()
stride=patch_size
overlap=0
print(full_ims_test.shape)
print(full_label_test.shape)
print(np.unique(full_label_test,return_counts=True))

sequence_len, row, col, bands = full_ims_test.shape
#pdb.set_trace()
prediction_rebuilt=np.ones((sequence_len,row,col,class_n)).astype(np.float32)*0


print("stride", stride)
print(len(range(patch_size//2,row-patch_size//2,stride)))
print(len(range(patch_size//2,col-patch_size//2,stride)))
for m in range(patch_size//2,row-patch_size//2,stride):
	for n in range(patch_size//2,col-patch_size//2,stride):
		patch_mask = mask_pad[m-patch_size//2:m+patch_size//2 + patch_size%2,
					n-patch_size//2:n+patch_size//2 + patch_size%2]
		if np.any(patch_mask==2):
			patch = full_ims_test[:,m-patch_size//2:m+patch_size//2 + patch_size%2,
						n-patch_size//2:n+patch_size//2 + patch_size%2]
			patch = np.expand_dims(patch, axis = 0)
			#patch = patch.reshape((1,patch_size,patch_size,bands))

			pred_cl = model.predict(patch)

			print(pred_cl[0].shape)
			if not skip_crf:
				for i,v in enumerate(pred_cl[0]):
					img_in = patch[0][i]
					img_in = np.array(img_in, dtype=np.uint8)
					img_in = np.expand_dims(img_in, axis=0)
					v = scipy.special.softmax(v, axis=-1)
					v = np.expand_dims(v, axis=0)
					pred_cl[0][i] = dense_crf(v,img=img_in,n_iters=10,sxy_gaussian=(3, 3), compat_gaussian=3,n_classes=class_n)

			_, _, x, y, _ = pred_cl.shape

			prediction_rebuilt[:,m-stride//2:m+stride//2,n-stride//2:n+stride//2] = pred_cl[:,overlap//2:x-overlap//2,overlap//2:y-overlap//2]
del full_ims_test
label_rebuilt=full_label_test.copy()

#carregar img

fullimg=np.load(fullimg_path+'full_ims_test.npy')
print("prediction_rebuilt")
print(np.shape(prediction_rebuilt))
print("fullimg")
print(np.shape(fullimg))
"""
for t,v in enumerate(prediction_rebuilt):
	img_in = fullimg[t]
	print(img_in.max(),img_in.min())
	img_in = np.array(img_in, dtype=np.uint8)
	print(np.unique(img_in,return_counts=True))
	img_in = np.expand_dims(img_in, axis=0)
	v = scipy.special.softmax(v, axis=-1)
	v = np.expand_dims(v, axis=0)
	print("crf")
	print(np.shape(img_in))
	print(img_in.dtype)
	print(np.shape(v))
	print(v.dtype)
	prediction_rebuilt[t] = dense_crf(v,img=img_in,n_iters=10,sxy_gaussian=(3, 3), compat_gaussian=3,n_classes=class_n)
"""

print("prediction_rebuilt")
print(np.shape(prediction_rebuilt))
np.save('model_best_UNet3D_ATPP5_1.npy',prediction_rebuilt)
