import numpy as np
import keras
from scipy.io import loadmat
from bisect import bisect
from scipy.ndimage.interpolation import rotate
from variables import *

def load_mat_file(filename,input_size,edge_size,stride):
    matfile = loadmat(data_dir + filename)
    if 'Original' in matfile.keys():
        image = matfile['Original']
    if 'distmap' in matfile.keys():
        distmap = matfile['distmap']
    label = (distmap < 2)*1.0
    image = pad_image(image,input_size,edge_size,stride)
    label = pad_image(label,input_size,edge_size,stride)
    return image, label

def pad_image(Im,input_size,edge_size,stride):
    padding = [int(np.ceil((Im.shape[i] + edge_size[i] - input_size[i])/stride[i])*stride[i] \
                   + input_size[i] - Im.shape[i] - edge_size[i]) for i in range(3)]
    Im = np.pad(Im,((edge_size[0],padding[0]),(edge_size[1],padding[1]),(edge_size[2],padding[2])),mode = 'reflect')
    return Im

def Im2Data(Im,input_size,edge_size,stride,label = None):
    stride_size = tuple([int((Im.shape[i] - input_size[i])/stride[i] + 1) for i in range(3)])
    L = int(stride_size[0]*stride_size[1]*stride_size[2])
    x = np.zeros([L,input_size[0],input_size[1],input_size[2],1])
    output_size = tuple([int(input_size[i] - 2*edge_size[i]) for i in range(3)])
    if label is not None:
        y = np.zeros([L,output_size[0],output_size[1],output_size[2],1])
    else:
        y = None
    l = 0
    for k in range(stride_size[2]):
        for j in range(stride_size[1]):
            for i in range(stride_size[0]):
                pos_i,pos_j,pos_k = i*stride[0],j*stride[1],k*stride[2]
                x[l,:,:,:,0] = Im[pos_i:pos_i+input_size[0],pos_j:pos_j+input_size[1],pos_k:pos_k+input_size[2]]
                if label is not None:
                    y[l,:,:,:,0] = label[pos_i+edge_size[0]:pos_i+edge_size[0]+output_size[0],pos_j+edge_size[1]:pos_j+edge_size[1]+output_size[1],pos_k+edge_size[2]:pos_k+edge_size[2]+output_size[2]]
                l += 1
    return x/255.0,y

def Data_augmentation(images,labels,p_fliplr = 0.5,p_flipud = 0.5,p_flipz = 0.5,p_rot_90_270 = 0.5):
    # flip the image
    if np.random.rand() > 1-p_fliplr:
        images = np.flip(images,1)
        labels = np.flip(labels,1)
    if np.random.rand() > 1-p_flipud:
        images = np.flip(images,2)
        labels = np.flip(labels,2)
    if np.random.rand() > 1-p_flipz:
        images = np.flip(images,3)
        labels = np.flip(labels,3)
    # rotate the image
    p_rot = np.array([0.5-p_rot_90_270/2,p_rot_90_270/2,0.5-p_rot_90_270/2,p_rot_90_270/2])
    p_rot = np.cumsum(p_rot)
    if np.random.rand() < p_rot[0]:
        pass
    elif np.random.rand() < p_rot[1]:
        images = np.rot90(images,1,(1,2))
        labels = np.rot90(labels,1,(1,2))
    elif np.random.rand() < p_rot[2]:
        images = np.rot90(images,2,(1,2))
        labels = np.rot90(labels,2,(1,2))
    else:
        images = np.rot90(images,3,(1,2))
        labels = np.rot90(labels,3,(1,2))
    return images,labels

def create_circular_mask(Im_shape, radius=None):
    batch_size, h, w, z = Im_shape[0],Im_shape[1],Im_shape[2],Im_shape[3]
    center = w/2-0.5, h/2-0.5
    if radius is None: # use the smallest distance between the center and image walls
        radius = min(center[0], center[1], w-center[0], h-center[1])

    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)

    mask2d = dist_from_center <= radius
    mask3d = np.zeros((h,w,z))
    for i in range(z):
        mask3d[:,:,i] = mask2d
    masks = np.zeros(Im_shape)
    for i in range(batch_size):
        masks[i,:,:,:] = mask3d
    return masks[:,:,:,:,np.newaxis]

def rotate_image(x,y,masks,reshape=False):
    x_rot = np.zeros(x.shape)
    y_rot = np.zeros(y.shape)
    for i in range(x.shape[0]):
        angle = np.random.rand()*360
        x_rot[i,:] = rotate(x[i,:], angle, axes=(0, 1), reshape=reshape)
        y_rot[i,:] = rotate(y[i,:], angle, axes=(0, 1), reshape=reshape)
    return x_rot*masks,(y_rot>0.5)*1.0

class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, files, train_or_val,cross_validation_ind,input_size,edge_size,stride,seed = 1, batch_size=32, augmentation = False,shuffle = False):
        'Initialization'
        self.input_size = input_size
        self.edge_size = edge_size
        self.stride = stride
        self.output_size = tuple([int(input_size[i] - 2*edge_size[i]) for i in range(3)])
        np.random.seed(seed = seed)
        self.batch_size = batch_size
        self.augmentation = augmentation
        self.shuffle = shuffle
        self.masks = create_circular_mask((batch_size,input_size[0],input_size[1],input_size[2]))
        self._load_image_file(files)
        self._train_val_split(train_or_val,cross_validation_ind)
        self.n = 0
        self.max = self.__len__()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor( len(self.indexes) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        X,y = self._get_training_sample(indexes)
        angle = np.random.rand()*360
        X,y = rotate_image(X,y,masks=self.masks)
        y = y[:,self.edge_size[0]:self.edge_size[0]+self.output_size[0],self.edge_size[1]:self.edge_size[1]+self.output_size[1],self.edge_size[2]:self.edge_size[2]+self.output_size[2],:]
        if self.augmentation:
            X,y = Data_augmentation(X,y)
        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        if self.shuffle == True:
            np.random.shuffle(self.indexes)
    
    def _load_image_file(self,files):
        self.images = []
        self.labels = []
        image_crop_locations = []
        self.n_sample_per_image = []
        for file in files:
            image,label = load_mat_file(file,self.input_size,self.edge_size,self.stride)
            image_crop_location = self._get_crop_location(image)
            self.images.append(image)
            self.labels.append(label)
            self.n_sample_per_image.append(image_crop_location.shape[0])
            image_crop_locations.append(image_crop_location)
        self.image_crop_locations = np.concatenate(image_crop_locations).astype(int)
        self.total_sample_train_val = self.image_crop_locations.shape[0]
        self.n_sample_cumsum = np.cumsum(self.n_sample_per_image)
     
    def _train_val_split(self,train_or_val,cross_validation_ind):
        all_index = np.arange(self.total_sample_train_val)
        np.random.shuffle(all_index)
        cv_size = int(np.floor(self.total_sample_train_val/5))
        if train_or_val == 'train':
            self.indexes = np.concatenate([all_index[0:cross_validation_ind*cv_size],all_index[(cross_validation_ind+1)*cv_size:]])
        elif train_or_val == 'val':
            self.indexes = all_index[cross_validation_ind*cv_size:(cross_validation_ind+1)*cv_size]

    def _get_crop_location(self,Im):
        stride_size = tuple([int((Im.shape[i] - self.input_size[i])/self.stride[i] + 1) for i in range(3)])
        L = int(stride_size[0]*stride_size[1]*stride_size[2])
        image_crop_location = np.zeros([L,3])
        l = 0
        for k in range(stride_size[2]):
            for j in range(stride_size[1]):
                for i in range(stride_size[0]):
                    image_crop_location[l,:] = i*self.stride[0],j*self.stride[1],k*self.stride[2]
                    l += 1
        return image_crop_location
    
    def _get_training_sample(self,indexes):
        X = np.zeros((self.batch_size,self.input_size[0],self.input_size[1],self.input_size[2],1))
        y = np.zeros((self.batch_size,self.input_size[0],self.input_size[1],self.input_size[2],1))
        for i,index in enumerate(indexes):
            image_id = bisect(self.n_sample_cumsum, index)
            pos_i,pos_j,pos_k = self.image_crop_locations[index,:]
            X[i,:,:,:,0] = self.images[image_id][pos_i:pos_i+self.input_size[0],pos_j:pos_j+self.input_size[1],pos_k:pos_k+self.input_size[2]]
            y[i,:,:,:,0] = self.labels[image_id][pos_i:pos_i+self.input_size[0],pos_j:pos_j+self.input_size[1],pos_k:pos_k+self.input_size[2]]
            
        return X/255.0,y

    def __next__(self):
        if self.n >= self.max:
            self.n = 0
        result = self.__getitem__(self.n)
        self.n += 1
        return result