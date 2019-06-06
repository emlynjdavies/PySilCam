import os
import random
# import _pickle as pickle
import numpy as np
import matplotlib.pyplot
from matplotlib.pyplot import imshow
import keras
from keras.preprocessing import image
from keras.applications.imagenet_utils import decode_predictions, preprocess_input
from keras.models import Model
from sklearn.decomposition import PCA
from scipy.spatial import distance
from tqdm import tqdm
from sklearn.manifold import TSNE
from PIL import Image
import pylab as plt
import skimage.io
import skimage.color
import skimage.exposure


# directory = '../EGGTOX/2017-04-23-27 - EGGTOX larvae 3 dph/'
directory = '/mnt/DATA/db2all/'
#TSNE_NAME = '/2017-04-23-27 - EGGTOX larvae 3 dph-test'
TSNE_NAME = 'testtsne'

def convert_im(im):
#    im = skimage.io.imread(path)
    
    im = np.array(im)
    
    im = skimage.color.rgb2gray(im)
    im = np.float64(im)/np.max(im)
    
#    im_ = np.copy(im[200:-200,:,0]*255)
    im_ = np.copy(im*255)
    
    im_ = skimage.exposure.rescale_intensity(im_, out_range=(5, 250))
    
    r, c = np.shape(im_)
    im_new = np.zeros((r,c,3), dtype=np.uint8)
    im_new[:,:,0] = im_
    im_new[:,:,1] = im_
    im_new[:,:,2] = im_
    
    im = Image.fromarray(np.uint8(im_new))
    
    return im

model = keras.applications.VGG16(weights='imagenet', include_top=True)

def get_image(path):
    img = image.load_img(path, target_size=model.input_shape[1:3])
#    img = image.load_img(path, target_size=None)
    img = convert_im(img)
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return img, x

feat_extractor = Model(inputs=model.input, outputs=model.get_layer("fc2").output)

#images_path = 'PROC/'
images_path = directory
max_num_images = 1000000

images = [os.path.join(dp, f) for dp, dn, filenames in os.walk(images_path) for f in filenames if os.path.splitext(f)[1].lower() in ['.jpg','.png','.jpeg','.tiff']]
if max_num_images < len(images):
    images = [images[i] for i in sorted(random.sample(range(len(images)), max_num_images))]

print('feature extraction')    
features = []
for image_path in tqdm(images):
    img, x = get_image(image_path);
    feat = feat_extractor.predict(x)[0]
    features.append(feat)
print('  OK.')
    
features = np.array(features)
# pca = PCA(n_components=300)
pca = PCA() # keep all the components. why limit to 300?!
pca.fit(features)
pca_features = pca.transform(features)

num_images_to_plot = max_num_images

if len(images) > num_images_to_plot:
    sort_order = sorted(random.sample(range(len(images)), num_images_to_plot))
    images = [images[i] for i in sort_order]
    pca_features = [pca_features[i] for i in sort_order]
    
X = np.array(pca_features)
print('TSNE training')
tsne = TSNE(n_components=2, learning_rate=150, perplexity=30, angle=0.2, verbose=2).fit_transform(X)
print('  OK.')

tx, ty = tsne[:,0], tsne[:,1]
tx = (tx-np.min(tx)) / (np.max(tx) - np.min(tx))
ty = (ty-np.min(ty)) / (np.max(ty) - np.min(ty))
plt.plot(tx, ty,'.')

width = 20000
height = 20000
max_dim = 500

print('PLotting')
full_image = Image.new('RGB', (width, height))
for img, x, y in tqdm(zip(images, tx, ty)):
    tile = Image.open(img)
    rs = max(1, tile.width/max_dim, tile.height/max_dim)
    #tile = tile.resize((int(tile.width/rs), int(tile.height/rs)), Image.ANTIALIAS)
    tile = tile.resize((max(1,int(tile.width / rs)), max(1,int(tile.height / rs))), Image.ANTIALIAS)
    full_image.paste(tile, (int((width-max_dim)*x), int((height-max_dim)*y)))

matplotlib.pyplot.figure(figsize = (16,12))
imshow(full_image)

full_image.save(directory +'tsne/' + TSNE_NAME + '-tsne.tiff')
    
print('----THE END ----')