from torchvision import transforms
import torchvision
import torch
import torch.nn as nn
import glob
from PIL import Image
import numpy as np
from numpy import argmax
from pymongo import MongoClient

try:
    # Conectar a la db, host y puerto
    conn = MongoClient(host='localhost', port=27017)
    # Obtener base de datos
    db = conn.local
except:
    pass

# Definir modelo
class scratch_nn(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=100, kernel_size=5, stride=1, padding=0)
        self.conv2 = nn.Conv2d(100, 200, 3, stride=1, padding=0)
        self.conv3 = nn.Conv2d(200, 400, 3, stride=1, padding=0)
        self.mpool = nn.MaxPool2d(kernel_size=3)
        self.relu = nn.ReLU()
        self.linear1 = nn.Linear(19600,1024)
        self.linear2 = nn.Linear(1024,512)
        self.linear3 = nn.Linear(512,2)
        self.classifier = nn.Softmax(dim=1)
        
    def forward(self,x):
        x = self.mpool( self.relu(self.conv1(x)) )
        x = self.mpool( self.relu(self.conv2(x)) )
        x = self.mpool( self.relu(self.conv3(x)) )
        x = torch.flatten(x, start_dim=1)
        x = self.linear1(x)
        x = self.linear2(x)
        x = self.linear3(x)
        x = self.classifier(x)
        return x

# Cargar modelo entrenado
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = scratch_nn()
model.load_state_dict(torch.load("dogs_cats_model.pth"))
model.eval()
model = model.to(device)

# Definir preprocesados de la imagen
data_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((224, 224)),
])

# Realizar la prediccion de todas las imagenes en la carpeta
labels = ["Cat", "Dog"]
for image_path in glob.glob("predict_cat_dog/*.jpg"):
	img_orig = np.array(Image.open(image_path))
	img = data_transform(img_orig).unsqueeze(0).to(device)
	outputs = model(img)
	outputs = outputs.detach().cpu().numpy()
	output = argmax(outputs, axis=1)[0]
	print("Image path:" + image_path + ", Predicted label: " + labels[output])
	# Almacenar en base de datos
	try:
		db.data.insert_one({"path_img": image_path, "predicted_label": labels[output]})
	except:
		pass