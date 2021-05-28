import numpy as np
import cv2
import glob
import argparse
import scipy
import pdb
import sys
#sys.path.append('../../../../../train_src/analysis/')
import pathlib
from PredictionsLoader import PredictionsLoaderNPY, PredictionsLoaderModel
from utils import seq_add_padding, add_padding
import pydensecrf.densecrf as dcrf
from pydensecrf.utils import create_pairwise_bilateral
def dense_crf(probs, img=None, n_iters=10, n_classes=19,
			  sxy_gaussian=(1, 1), compat_gaussian=4,
			  sxy_bilateral=(49, 49), compat_bilateral=5,
			  srgb_bilateral=(13, 13, 13)):
	
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
		pairwise_bilateral = create_pairwise_bilateral(sdims=(10,10), schan=(.01), img=img[0], chdim=2)
		d.addPairwiseEnergy(pairwise_bilateral, compat=10)

	Q = d.inference(n_iters)
	preds = np.array(Q, dtype=np.float32).reshape((n_classes, h, w)).transpose(1, 2, 0)
	return np.expand_dims(preds, 0)


parser = argparse.ArgumentParser(description='')
parser.add_argument('-ds', '--dataset', dest='dataset',
					default='cv', help='t len')
parser.add_argument('-mdl', '--model', dest='model_type',
					default='densenet', help='t len')

a = parser.parse_args()

dataset=a.dataset
model_type=a.model_type

direct_execution=True
if direct_execution==True:
	dataset='cv'
	model_type='unet'



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
	if model_type=='densenet':
		predictions_path=path+'prediction_DenseNetTimeDistributed_128x2_batch16_full.npy'
	elif model_type=='biconvlstm':
		predictions_path=path+'prediction_ConvLSTM_seq2seq_bi_batch16_full.npy'
	elif model_type=='convlstm':
		predictions_path=path+'prediction_ConvLSTM_seq2seq_batch16_full.npy'
	elif model_type=='unet':
		predictions_path=path+'prediction_BUnet4ConvLSTM_repeating1.npy'
		#predictions_path=path+'prediction_BUnet4ConvLSTM_repeating2.npy'
		#predictions_path=path+'prediction_BUnet4ConvLSTM_repeating4.npy'


	elif model_type=='atrous':
		predictions_path=path+'prediction_BAtrousConvLSTM_2convins5.npy'
	elif model_type=='atrousgap':
		predictions_path=path+'prediction_BAtrousGAPConvLSTM_raulapproved.npy'
		#predictions_path=path+'prediction_BAtrousGAPConvLSTM_repeating3.npy'
		#predictions_path=path+'prediction_BAtrousGAPConvLSTM_repeating4.npy'



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
	if model_type=='densenet':
		predictions_path=path+'prediction_DenseNetTimeDistributed_128x2_batch16_full.npy'
	elif model_type=='biconvlstm':
		predictions_path=path+'prediction_ConvLSTM_seq2seq_bi_batch16_full.npy'
	elif model_type=='convlstm':
		predictions_path=path+'prediction_ConvLSTM_seq2seq_batch16_full.npy'
	elif model_type=='unet':
		#predictions_path=path+'prediction_BUnet4ConvLSTM_repeating2.npy'
		predictions_path=path+'model_best_UNet3D_ATPP4_f32_1.h5'

	elif model_type=='atrous':
		predictions_path=path+'prediction_BAtrousConvLSTM_repeating2.npy'
	elif model_type=='atrousgap':
		#predictions_path=path+'prediction_BAtrousGAPConvLSTM_raulapproved.npy'
		#predictions_path=path+'prediction_BAtrousGAPConvLSTM_repeating4.npy'
		predictions_path=path+'prediction_BAtrousGAPConvLSTM_repeating6.npy'
	elif model_type=='unetend':
		predictions_path=path+'prediction_unet_convlstm_temouri2.npy'
	elif model_type=='allinputs':
		predictions_path=path+'prediction_bconvlstm_wholeinput.npy'

	mask_path=data_path+'cv_data/TrainTestMask.tif'
	location_path=data_path+'cv_data/locations/'

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
add_padding_flag=False
if add_padding_flag==True:
	full_ims_test, stride, step_row, step_col, overlap = seq_add_padding(full_ims_test,patch_size,0)
	#full_label_test, _, _, _, _ = seq_add_padding(full_label_test,32,0)
	mask_pad, _, _, _, _ = add_padding(mask,patch_size,0)
else:
	mask_pad=mask.copy()
	stride=patch_size
	overlap=0
print(full_ims_test.shape)
print(full_label_test.shape)
print(np.unique(full_label_test,return_counts=True))

sequence_len, row, col, bands = full_ims_test.shape
#pdb.set_trace()
prediction_rebuilt=np.ones((sequence_len,row,col)).astype(np.uint8)*255

print("stride", stride)
print(len(range(patch_size//2,row-patch_size//2,stride)))
print(len(range(patch_size//2,col-patch_size//2,stride)))
ctr = 0
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
			pred_cl = np.squeeze(pred_cl, axis=0)
			for i,v in enumerate(pred_cl):
				img_in = patch[0][i]
				img_in = np.array(img_in, dtype=np.uint8)
				img_in = np.expand_dims(img_in, axis=0)
				v = scipy.special.softmax(v, axis=-1)
				v = np.expand_dims(v, axis=0)
				pred_cl[i] = dense_crf(v,img=img_in,n_iters=10,sxy_gaussian=(3, 3), compat_gaussian=3,n_classes=class_n)
				ctr = ctr + 1
				print(ctr)
			pred_cl = np.expand_dims(pred_cl, axis = 0)
			pred_cl = pred_cl.argmax(axis=-1)

			_, _, x, y = pred_cl.shape

			prediction_rebuilt[:,m-stride//2:m+stride//2,n-stride//2:n+stride//2] = pred_cl[:,overlap//2:x-overlap//2,overlap//2:y-overlap//2]
del full_ims_test
label_rebuilt=full_label_test.copy()

del full_label_test
if add_padding_flag==True:
	prediction_rebuilt=prediction_rebuilt[:,overlap//2:-step_row,overlap//2:-step_col]

print("---- pad was removed")

print(prediction_rebuilt.shape, mask.shape, label_rebuilt.shape)
# everything outside mask is 255
for t_step in range(sequence_len):
	label_rebuilt[t_step][mask==0]=255

	prediction_rebuilt[t_step][mask==0]=255
#label_rebuilt[label_rebuilt==class_n]=255

print("everything outside mask is 255")
print(np.unique(label_rebuilt,return_counts=True))
print(np.unique(prediction_rebuilt,return_counts=True))


# Paint it!


print(custom_colormap.shape)
#class_n=custom_colormap.shape[0]
#=== change to rgb
print("Gray",prediction_rebuilt.dtype)
prediction_rgb=np.zeros((prediction_rebuilt.shape+(3,))).astype(np.uint8)
label_rgb=np.zeros_like(prediction_rgb)
print("Adding color...")

for t_step in range(sequence_len):
	prediction_rgb[t_step]=cv2.cvtColor(prediction_rebuilt[t_step],cv2.COLOR_GRAY2RGB)
	label_rgb[t_step]=cv2.cvtColor(label_rebuilt[t_step],cv2.COLOR_GRAY2RGB)

print("RGB",prediction_rgb.dtype,prediction_rgb.shape)

for idx in range(custom_colormap.shape[0]):
	print("Assigning color. t_step:",idx)
	for chan in [0,1,2]:
		prediction_rgb[:,:,:,chan][prediction_rgb[:,:,:,chan]==idx]=custom_colormap[idx,chan]
		label_rgb[:,:,:,chan][label_rgb[:,:,:,chan]==idx]=custom_colormap[idx,chan]

print("RGB",prediction_rgb.dtype,prediction_rgb.shape)

#for idx in range(custom_colormap.shape[0]):
#	for chan in [0,1,2]:
#		prediction_rgb[:,:,chan][prediction_rgb[:,:,chan]==correspondence[idx]]=custom_colormap[idx,chan]
print("Saving the resulting images for all dates...")
for t_step in range(sequence_len):

	label_rgb[t_step]=cv2.cvtColor(label_rgb[t_step],cv2.COLOR_BGR2RGB)
	prediction_rgb[t_step]=cv2.cvtColor(prediction_rgb[t_step],cv2.COLOR_BGR2RGB)
	save_folder=dataset+"/"+model_type+"/"
	pathlib.Path(save_folder).mkdir(parents=True, exist_ok=True)
	cv2.imwrite(save_folder+"prediction_t"+str(t_step)+"_"+model_type+".png",prediction_rgb[t_step])
	cv2.imwrite(save_folder+"label_t"+str(t_step)+"_"+model_type+".png",label_rgb[t_step])

print(prediction_rgb[0,0,0,:])
