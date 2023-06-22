from ultralytics import YOLO
import cv2
import numpy as np
#plots
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os
import subprocess

from tqdm.notebook import tqdm

import IPython
from IPython.display import Video, display

# constantes

FRAC = 0.65  # Fração para redimensionamento do vídeo
#PATH_VIDEO = '/Users/mateus/Downloads/vehicle-counting.mp4'
# display(Video(data=PATH_VIDEO, height=int(720*FRAC), width=int(1280*FRAC)))
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#loading a YOLO model
model = YOLO('yolov8x.pt')
#Mudou do YOLOV8x para YOLOV8n


# person, backpack, handbag, suitcase, laptop, celphone
class_IDS = [0, 24, 26, 28, 63, 67] # IDs das classes dos objetos monitorados
dict_classes = model.model.names # Dicionário de tradução das classes

# Lista de nomes dos objetos monitorados
objetos_monitorados = ['Pessoa','Mochila','Bolsa','Maleta','Laptop','Celular']

# Captura de vídeo usando a câmera (0 indica a câmera padrão)
cap = cv2.VideoCapture(0)

# Percentual de redimensionamento do frame
scale_percent = 50

# Número mínimo de frames para detectar a ausência de um objeto
min_frames_detectar = 7

# Inicialização dos contadores para cada classe de objeto
conta_frames_celular = 0
conta_frames_mochila = 0
conta_frames_bolsa = 0
conta_frames_maleta = 0
conta_frames_laptop = 0

# Função para traduzir as classes de objeto
def traduz_classe():
    dict_classes.update({0:'Pessoa'})
    dict_classes.update({24:'Mochila'})
    dict_classes.update({26:'Bolsa'})
    dict_classes.update({28:'Maleta'})
    dict_classes.update({63:'Laptop'})
    dict_classes.update({67:'Celular'})
traduz_classe()

# Redimensiona um frame para uma escala percentual
def resize_frame(frame, scale_percent):
    """Function to resize an image in a percent scale"""
    width = int(frame.shape[1] * scale_percent / 100)
    height = int(frame.shape[0] * scale_percent / 100)
    dim = (width, height)

    resized = cv2.resize(frame, dim, interpolation = cv2.INTER_AREA)
    return resized


# Função para detectar a ausência de um objeto
def detecta_ausencia_objeto(conta_frames, nome_objeto):

    if nome_objeto in labels:
        conta_frames += 1
    else:
        if conta_frames == 0:
            conta_frames = 0
        if conta_frames > 0 and (conta_frames < min_frames_detectar):
            conta_frames = 0
        if conta_frames > 0 and (conta_frames >= min_frames_detectar):
            conta_frames = 0
            print("ATENÇÃO!!!  ROUBARAM O OBJETO: " + nome_objeto + " ATENÇÃO!!!")
    print(f"{conta_frames} frames do objeto {nome_objeto}.", )

    return conta_frames


# main()
if (cap.isOpened( ) == False):
    print("Erro abrindo o arquivo de video")

# Loop principal para processar os frames do vídeo
while (cap.isOpened( )):
    ret, frame=cap.read( )
    frame  = resize_frame(frame, scale_percent)

    # Faz a detecção de objetos usando o modelo YOLO
    predicao_modelo=model.predict(frame, conf=0.65, classes=class_IDS, verbose=False)

    boxes=predicao_modelo[0].boxes.xyxy.cpu( ).numpy( )
    conf=predicao_modelo[0].boxes.conf.cpu( ).numpy( )
    classes=predicao_modelo[0].boxes.cls.cpu( ).numpy( )

    # Cria um DataFrame com as posições dos objetos detectados
    positions_frame=pd.DataFrame(predicao_modelo[0].cpu( ).numpy( ).boxes.data,
                                 columns=['xmin', 'ymin', 'xmax', 'ymax', 'conf', 'class'])

    # Obtém os rótulos das classes dos objetos detectados
    labels=[dict_classes[i] for i in classes]


    print("Objetos detectados:", labels)


    # Verifica a ausência de cada objeto monitoradoc
    conta_frames_mochila = detecta_ausencia_objeto(conta_frames_mochila, "Mochila")
    conta_frames_celular  = detecta_ausencia_objeto(conta_frames_celular, "Celular")
    conta_frames_bolsa  = detecta_ausencia_objeto(conta_frames_celular, "Bolsa")
    conta_frames_maleta  = detecta_ausencia_objeto(conta_frames_celular, "Maleta")
    conta_frames_laptop  = detecta_ausencia_objeto(conta_frames_celular, "Laptop")

    # Itera sobre as bounding boxes e desenha as caixas e os rótulos no frame
    for ix, row in enumerate(positions_frame.iterrows( )):
        # Pega as coordenadas para cada fileira
        xmin, ymin, xmax, ymax, confidence, category,=row[1].astype('int')

        # Calcula o centro da bounding box
        center_x, center_y=int(((xmax + xmin)) / 2), int((ymax + ymin) / 2)

        # Desenha a bounding box
        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (255, 0, 0), 5)  # box
        cv2.circle(frame, (center_x, center_y), 5, (255, 0, 0), -1)  # center of box

        # Coloca o nome da classe na bounding box
        cv2.putText(img=frame, text=labels[ix] + ' - ' + str(np.round(conf[ix], 2)),
                    org=(xmin, ymin - 10), fontFace=cv2.FONT_HERSHEY_TRIPLEX, fontScale=1, color=(255, 0, 0),
                    thickness=2)

    if ret == True:
        # Mostra o frame processado
        cv2.imshow('Frame', frame)

        # Aperta Q no keyboard para sair
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    else:
        break


cap.release( )
cv2.destroyAllWindows( )

