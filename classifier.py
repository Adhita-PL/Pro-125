import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import fetch_openml 
from PIL import Image
import PIL.ImageOps

X = np.load('image.npz')['arr_0']
y = pd.read_csv('labels.csv')['labels']
print(pd.Series(y).value_counts()) 
classes = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z'] 
nclasses = len(classes)

#splitting the data for training and testing
x_train, x_test, y_train, y_test = train_test_split(X, y, random_state = 9, train_size = 3500, test_size = 500)

x_train_scaled = x_train / 255.0
x_test_scaled = x_test / 255.0

#training the model
clf = LogisticRegression(solver = 'saga', multi_class='multinomial').fit(x_train_scaled, y_train)

def get_prediction(image) : 
    #opening the image 
    im_pil = Image.open(image)
    #converting image to greyscale
    image_bw = im_pil.convert('L') 
    #resizing the image to 28, 28 because the dataset image size is 28,28, to compare, we are resizing
    #and we used antialias for smothening it
    image_bw_resized = image_bw.resize((22,30), Image.ANTIALIAS) 

    
    pixel_filter = 20
    #using the percentile function we are removing the pixels which are less than 20.
    #we are trying to filter it out and make it look clean
    min_pixel = np.percentile(image_bw_resized, pixel_filter)

    image_bw_resized_inverted_scaled = np.clip(image_bw_resized - min_pixel, 0, 255)
    max_pixel = np.max(image_bw_resized) 
    #coverting it into array
    image_bw_resized_inverted_scaled = np.asarray(image_bw_resized_inverted_scaled) / max_pixel
    #creating a test sample and making prediction
    test_sample = np.array(image_bw_resized_inverted_scaled).reshape(1,660)
    test_pred = clf.predict(test_sample) 
    return test_pred[0]
