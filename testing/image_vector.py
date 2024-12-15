import numpy as np
from img2vec_pytorch import Img2Vec
from PIL import Image

# Initialize Img2Vec with GPU
img2vec = Img2Vec(cuda=True)
fake_img = np.zeros((1280, 720, 3), dtype=np.uint8)
# Read in an image (rgb format)
img = Image.fromarray(fake_img)
# Get a vector from img2vec, returned as a torch FloatTensor
vec = img2vec.get_vec(img, tensor=True)
# Or submit a list
# vectors = img2vec.get_vec(list_of_PIL_images)
print(vec.shape)
