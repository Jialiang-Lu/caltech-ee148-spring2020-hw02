import os
import numpy as np
import json
from PIL import Image

class DisjointSet:
    '''
    a simple implementation of a disjoint-set data structure
    '''
    _data = list()

    def __init__(self, init_data=None):
        self._data = []
        if init_data:
            for item in list(set(init_data)):
                self._data.append({item})

    def __repr__(self):
        return self._data.__repr__()
    
    def index(self, elem):
        for item in self._data:
            if elem in item:
                return self._data.index(item)
        return None

    def find(self, elem):
        for item in self._data:
            if elem in item:
                return self._data[self._data.index(item)]
        return None
    
    def add(self, elem):
        index_elem = self.index(elem)
        if index_elem is None:
            self._data.append({elem})
    
    def union(self, elem1, elem2):
        index_elem1 = self.index(elem1)
        index_elem2 = self.index(elem2)
        if index_elem1 is None:
            self.add(elem1)
            self.union(elem1, elem2)
            return
        if index_elem2 is None:
            self.add(elem2)
            self.union(elem1, elem2)
            return
        if index_elem1 != index_elem2:
            self._data[index_elem2] = self._data[index_elem2].union(self._data[index_elem1])
            del self._data[index_elem1]
        
    def get(self):
        return self._data

def label_binary(im):
    '''
    label each connected patch within a binary image
    '''
    if not isinstance(im, np.ndarray):
        im = np.asarray(im)
    im = (im>0).astype(int)
    labels = np.zeros_like(im)
    n_labels = 0
    idc = DisjointSet()
    for r, c in np.ndindex(im.shape):
        v = im[r, c]
        vu = labels[r-1, c] if r>0 else 0
        vl = labels[r, c-1] if c>0 else 0
        if v>0:
            if vu==0 and vl==0:
                n_labels += 1
                idc.add(n_labels)
                labels[r, c] = n_labels
            elif vu==0 and vl>0:
                labels[r, c] = vl
            elif vu>0 and vl==0:
                labels[r, c] = vu
            else:
                labels[r, c] = vu if vu<vl else vl
                idc.union(vu, vl)
    for r, c in np.ndindex(im.shape):
        v = labels[r, c]
        labels[r, c] = 0 if v==0 else idc.index(v)+1
    return labels, len(idc.get())

def apply_over_labels(im, labels, func):
    '''
    apply function over the connected labels labeled by labels
    '''
    return [func(labels==c) for c in range(1, labels.max()+1)]

def remove_holes(im, labels, max_area=500):
    '''
    remove holes in binary image
    '''
    fill = True if im.dtype==np.dtype('bool') else 1
    for k in range(1, labels.max()+1):
        patch = labels==k
        if patch.sum()<max_area:
            im[patch] = fill

def perimeter(im):
    '''
    calculate perimeter of binary image
    '''
    im = (im>0).astype(int)
    ima = np.zeros((im.shape[0]+2, im.shape[1]+2), dtype=int)
    ima[1:-1, 1:-1] = im
    imf = im*4-ima[1:-1, 2:]-ima[1:-1, :-2]-ima[2:, 1:-1]-ima[:-2, 1:-1]
    return (np.logical_and(imf>0, imf<4)).sum()

def area(im):
    '''
    calculate area of binary image
    '''
    return (im>0).sum()

def aspect_ratio(im):
    '''
    calculate aspect ratio of binary image
    '''
    return perimeter(im)**2/(4*np.pi*area(im))
    
def bounding_box(im):
    '''
    find bounding box of the binary image im
    '''
    idc = np.argwhere(im>0)
    return list(idc.min(axis=0))+list(idc.max(axis=0))

def bounding_boxes(labels):
    '''
    find all bounding boxes of the labels
    '''
    return [bounding_box(labels==v) for v in range(1, labels.max()+1)]

def circ_2d(radius):
    '''
    generate 2d circular kernel
    '''
    radiusn = np.ceil(radius).astype(int)
    nr = nc = radiusn*2+1
    r, c = np.ogrid[-radiusn:radiusn:nr*1j, -radiusn:radiusn:nc*1j]
    kernel = ((r**2+c**2)<=radius**2).astype(float)
    kernel = kernel/kernel.sum()
    return kernel

def gaus_2d(sigma):
    '''
    generate 2d gaussian kernel
    '''
    radiusn = np.ceil(sigma*3).astype(int)
    nr = nc = radiusn*2+1
    r, c = np.ogrid[-radiusn:radiusn:nr*1j, -radiusn:radiusn:nc*1j]
    kernel = np.exp(-(r**2+c**2)/(2*sigma**2))
    kernel = kernel/kernel.sum()
    return kernel
    
def compute_convolution(I, T, padding=0.):
    '''
    This function takes an image <I> and a template <T> (both numpy arrays) 
    and returns a heatmap where each grid represents the output produced by 
    convolution at each location.  
    '''
    I = I[..., None] if I.ndim<3 else I
    T = T[..., None] if T.ndim<3 else T
    nr_I, nc_I, nd_I = I.shape
    nr_T, nc_T, nd_T = T.shape
    r_T = np.ceil(nr_T/2).astype(int)-1
    c_T = np.ceil(nc_T/2).astype(int)-1
    I_pad = np.zeros((nr_I+nr_T-1, nc_I+nc_T-1, nd_I))+padding
    I_pad[r_T:r_T+nr_I, c_T:c_T+nc_I, :] = I[:, :, :]
    heatmap = np.zeros((nr_I, nc_I, max([nd_I, nd_T])))
    for r, c in np.ndindex((nr_I, nc_I)):
        heatmap[r, c, :] = (I_pad[r:r+nr_T, c:c+nc_T, :]*T).sum((0, 1))
    return np.squeeze(heatmap)

def predict_boxes(heatmap):
    '''
    This function takes heatmap and returns the bounding boxes and associated
    confidence scores.
    '''
    I_thr = heatmap>thr
    labels, n_labels = label_binary(I_thr)
    labels_bg, n_labels_bg = label_binary(np.logical_not(I_thr))
    remove_holes(I_thr, labels_bg, 500)
    labels, n_labels = label_binary(I_thr)
    boxes = bounding_boxes(labels)
    areas = apply_over_labels(I_thr, labels, area)
    ratios = apply_over_labels(I_thr, labels, aspect_ratio)
    idc = [x[0] for x in np.argwhere(np.logical_and(np.logical_and(np.array(areas)>=area_min, np.array(areas)<area_max), np.array(ratios)<ratio_max))]
    boxes_final = np.array(boxes)[idc].tolist()
    scores_final = [heatmap[labels==idx+1].max() for idx in idc]
    boxes_combined = [x+[y] for x, y in zip(boxes_final, scores_final)]
    
    return boxes_combined

def detect_red_light_mf(I):
    '''
    This function takes a numpy array <I> and returns a list <output>.
    The length of <output> is the number of bounding boxes predicted for <I>. 
    Each entry of <output> is a list <[row_TL,col_TL,row_BR,col_BR,score]>. 
    The first four entries are four integers specifying a bounding box 
    (the row and column index of the top left corner and the row and column 
    index of the bottom right corner).
    <score> is a confidence score ranging from 0 to 1. 

    Note that PIL loads images in RGB order, so:
    I[:,:,0] is the red channel
    I[:,:,1] is the green channel
    I[:,:,2] is the blue channel
    '''
    I_float = I.astype(float)/255
    I_single = I_float[:, :, 0]-I_float[:, :, 2]
    heatmap = compute_convolution(compute_convolution(I_single, circ_2d(radius_circ_kernel)), gaus_2d(sigma_gaussian_kernel))
    output = predict_boxes(heatmap)
    
    for i in range(len(output)):
        assert len(output[i]) == 5
        assert (output[i][4] >= 0.0) and (output[i][4] <= 1.0)

    return output

if __name__=='__main__':

    # Parameters
    radius_circ_kernel = 4 # radius of the circular step kernel
    sigma_gaussian_kernel = 1 # radius of the gaussian smooth kernel
    thr = 0.18 # threshold for detection
    # area_min = 12 # mininum area of a patch to be considered as traffic light
    # area_max = 1500 # maximum area of a patch to be considered as traffic light
    # ratio_max = 0.9 # maximum aspect ratio of a patch to be considered as traffic light
    area_min = 0 # For weaker algorithm
    area_max = 500000 # For weaker algorithm
    ratio_max = 100 # For weaker algorithm

    # Note that you are not allowed to use test data for training.
    # set the path to the downloaded data:
    data_path = '../data/RedLights2011_Medium'

    # load splits: 
    split_path = '../data/hw02_splits'
    file_names_train = np.load(os.path.join(split_path,'file_names_train.npy'))
    file_names_test = np.load(os.path.join(split_path,'file_names_test.npy'))

    # set a path for saving predictions:
    preds_path = '../data/hw02_preds'
    os.makedirs(preds_path, exist_ok=True) # create directory if needed

    # Set this parameter to True when you're done with algorithm development:
    done_tweaking = True

    '''
    Make predictions on the training set.
    '''
    preds_train = {}
    for i in range(len(file_names_train)):

        print(f'Processing {os.path.join(data_path,file_names_train[i])}')

        # read image using PIL:
        I = Image.open(os.path.join(data_path,file_names_train[i]))

        # convert to numpy array:
        I = np.asarray(I)

        preds_train[file_names_train[i]] = detect_red_light_mf(I)

    # save preds (overwrites any previous predictions!)
    with open(os.path.join(preds_path,'preds_train_2.json'),'w') as f:
        json.dump(preds_train,f)

    if done_tweaking:
        '''
        Make predictions on the test set. 
        '''
        preds_test = {}
        for i in range(len(file_names_test)):

            # read image using PIL:
            I = Image.open(os.path.join(data_path,file_names_test[i]))

            # convert to numpy array:
            I = np.asarray(I)

            preds_test[file_names_test[i]] = detect_red_light_mf(I)

        # save preds (overwrites any previous predictions!)
        with open(os.path.join(preds_path,'preds_test_2.json'),'w') as f:
            json.dump(preds_test,f)
