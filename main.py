#importing 

from PIL import Image
import cv2
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
import pandas as pd
import numpy as np
import skimage.transform as trans
import glob
import os
from pathlib import Path
from collections import Counter
from fastapi import FastAPI, File,UploadFile
from fastapi.middleware.cors import CORSMiddleware

#############################################################

#Start FastApi 
app = FastAPI()

#############################################################

# Adding Cors
origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

#############################################################


# Starting preparing Deep learning Models
# Types Of Tumor
classes = ['glioma', 'meiningioma','No Tumor','pituitary']


#############################################################

# Load three Models

# Frist Model Detect Is Mass Lesion found or not
model_type = load_model("model44.h5")

# Second Model Segmentation Mass Lesion
SegModel = load_model("Seg_Model2.h5")

# Thrid Model Classfiy which type of tumor
model = load_model("BTcnnModel 2.h5")

#############################################################

def Class_multi_Pridict():

    def scalar(img):
        return img

    dataa_paths_list = list(Path("Images").glob("*.jpg"))

    data_paths_list = []
    data_labels_list = []

    for p in dataa_paths_list:
        data_paths_list.append(p)
        data_labels_list.append("pred")

    others_generator = ImageDataGenerator(preprocessing_function=scalar)

    data_paths_series = pd.Series(data_paths_list, name="Images").astype(str)
    data_labels_Series = pd.Series(data_labels_list, name="TUMOR_Category")
    Main2 = pd.concat([data_paths_series, data_labels_Series], axis=1)

    img_gen = others_generator.flow_from_dataframe(dataframe=Main2,x_col="Images",y_col="TUMOR_Category",color_mode="rgb",class_mode="categorical",shuffle=False,target_size=(256, 256))



    predicted = model_type.predict(img_gen)

    results = []

    for p in predicted:
        results.append(classes[np.argmax(p)])

    c = Counter(results)
    c.most_common(1)
    r = c.most_common()[0][0]

    return r # Type

def SegPredict(path, color,n):
    r, g, b = color

    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    image = image / 255
    image = trans.resize(image, (256, 256, 1))

    rows, cols, ch = image.shape
    predicted = SegModel.predict(np.reshape(image, (1, 256, 256, 1)))
    predicted = np.reshape(predicted, (256, 256))
    predicted = predicted.astype(np.float32) * 255
    predicted = trans.resize(predicted, (rows, cols, 1))
    predicted = predicted.astype(np.uint8)
    predicted = cv2.cvtColor(predicted, cv2.COLOR_GRAY2BGR)

    x = np.array(image)
    x = trans.resize(image, (256, 256, 3))
    y = np.array(predicted)

    ret, mask = cv2.threshold(y, 120, 255, cv2.THRESH_BINARY)
    white_pixels = np.where((mask[:, :, 0] == 255) & (mask[:, :, 1] == 255) & (mask[:, :, 2] == 255))
    mask[white_pixels] = [r, g, b]
    mask = np.array(mask)
    s=abs(x - mask)
    s=np.array(s*255, dtype=np.uint8)
    img = Image.fromarray(s)
    img.save(f"SegImage/{n}.jpeg")

def Seg_Multi_Predict():

    Pathes = list(Path("Images").glob("*.jpg"))
    links = []

    for i in range(0,len(Pathes)) :

        SegPredict(f"Images/{i}.jpg", (0, 255, 255),i)

#############################################################

# Endpoint

@app.get('/')
async def home():
    return "Hello World"

@app.post("/Scan/")
async def Scan(files: list[bytes] = File(...)):

    #Remove Images in Images folder
    [f.unlink() for f in Path("Images").glob("*") if f.is_file()]

    # Save Images come from frontend in Images folder
    i=0
    for f in  files :
        with open(f'Images/{i}.jpg', 'wb') as image:
            image.write(f)
            image.close()
            i+=1

    # list OF Images Paths
    listOFImagesPath=[]

    mass = ""

    # call Class_multi_Pridict method
    r = Class_multi_Pridict()
    
    if r != "No Tumor":
        [f.unlink() for f in Path("SegImage").glob("*") if f.is_file()]

        Seg_Multi_Predict()
    
        for file in os.listdir("SegImage/"):
            listOFImagesPath.append("SegImage/" + file)

        mass = "Yes"

    else:
        for file in os.listdir("Images/"):
            listOFImagesPath.append("Images/" + file)
        
        mass = "No"

    return  {
            "Type":r , 
            "ListOfUrl":listOFImagesPath,
            "mass":mass
            }
