import os
from ultralytics import YOLO
import cv2


# # Início vídeo  no arquivo
# VIDEOS_DIR = os.path.join('.', 'videos')
# video_path = ('C:\\Users\\mattc\\PycharmProjects\\YOLO_aviario\\Arq_TESTE_AVES_001\\Arq_TESTE_AVES_001\\data\\videos\\video_aviario.mp4')
# video_path_out = '{}_out.mp4'.format(video_path)
#
# cap = cv2.VideoCapture(video_path)
# ret, frame = cap.read()
# H, W, _ = frame.shape
# out = cv2.VideoWriter(video_path_out, cv2.VideoWriter_fourcc(*'MP4V'), int(cap.get(cv2.CAP_PROP_FPS)), (W, H))
# # Fim vídeo no arquivo

# model = YOLO('yolov8x')

model_path = os.path.join('.', 'runs', 'detect', 'train9', 'weights', 'best.pt')
model = YOLO(model_path)

# Threshold for object detection
threshold = 0.5

# class_name_dict = {0: 'pessoa'}
class_name_dict = {0: 'comedouro'}

# Início vídeo na webcam
cap = cv2.VideoCapture(2)  # 0 = webcam padrão

# Pega as propriedades do vídeo
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

output_filename = "output_video.mp4"
fourcc = cv2.VideoWriter_fourcc(*'MP4V')

fps = 7  # FPS ajustado para a câmera Logitech (USB)
out = cv2.VideoWriter(output_filename, fourcc, fps, (frame_width, frame_height))
# FIM VÍDEO POR WEBCAM


# Variáveis para controlar o zoom
zoom_factor = 1.0
zoom_step = 0.1

def apply_zoom(frame, zoom_factor, center=None):
    height, width = frame.shape[:2]

    if center is None:
        center = (width // 2, height // 2)

    new_width, new_height = int(width / zoom_factor), int(height / zoom_factor)

    x1 = max(0, center[0] - new_width // 2)
    y1 = max(0, center[1] - new_height // 2)
    x2 = min(width, center[0] + new_width // 2)
    y2 = min(height, center[1] + new_height // 2)

    zoomed_frame = frame[y1:y2, x1:x2]

    return cv2.resize(zoomed_frame, (width, height))

capturas_dir = 'capturas'
if not os.path.exists(capturas_dir):
    os.makedirs(capturas_dir)

def captura_salva_frames(frame, frame_count):
    if frame_count % 50 ==0:  # Mudar aqui para variar a quantidade de frames
        image_filename = os.path.join(capturas_dir, f"Capturas_funcionamento_{frame_count:04d}.jpg")
        cv2.imwrite(image_filename, frame)

# Set the frame count for capturing and saving  frames
frame_count = 0



while True:
    # Captura frame por frame da webcam
    ret, frame = cap.read()

    if not ret:
        break

    # Aplicar zoom na ROI antes da detecção dos objetos
    frame_area_deteccao_zoomed = apply_zoom(frame, zoom_factor, center=None)

    # Substitui a ROI original pelo ROI com zoom no frame
    frame = frame_area_deteccao_zoomed
    #
    # # Detecção de objetos
    # results = model(frame)[0]
    #
    # for result in results.boxes.data.tolist():
    #     x1, y1, x2, y2, score, class_id = result
    #
    #     if score > threshold:
    #         cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 4)
    #         cv2.putText(frame, class_name_dict[int(class_id)].upper(), (int(x1), int(y1 - 10)),
    #                     cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)

    # Mostra o frame
    cv2.imshow('Deteccao de comedouros - Aviario', frame)

    # Escreve o frame no vídeo de saída
    out.write(frame)

    # Incrementa o contador de frames
    frame_count += 1

    # Chama função de captura (salvar) de frames
    captura_salva_frames(frame, frame_count)

    # Verifica as teclas pressionadas
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('z'):  # Pressione 'Z' para diminuir o zoom
        zoom_factor -= zoom_step
        if zoom_factor < 1.0:
            zoom_factor = 1.0
    elif key == ord('a'):  # Pressione 'A' para aumentar o zoom
        zoom_factor += zoom_step
        if zoom_factor > 10.0:
            zoom_factor = 10.0

cap.release()
out.release()

#
# import os
# from ultralytics import YOLO
# import cv2
#
#
# # Início vídeo  no arquivo
# VIDEOS_DIR = os.path.join('.', 'videos')
# video_path = ('C:\\Users\\mattc\\PycharmProjects\\YOLO_aviario\\Arq_TESTE_AVES_001\\Arq_TESTE_AVES_001\\data\\videos\\video_aviario.mp4')
# video_path_out = '{}_out.mp4'.format(video_path)
#
# cap = cv2.VideoCapture(video_path)
# ret, frame = cap.read()
# H, W, _ = frame.shape
# out = cv2.VideoWriter(video_path_out, cv2.VideoWriter_fourcc(*'MP4V'), int(cap.get(cv2.CAP_PROP_FPS)), (W, H))
# # Fim vídeo no arquivo
#
# model = YOLO('yolov8x')
#
# model_path = os.path.join('.', 'runs', 'detect', 'train9', 'weights', 'best.pt')
# model = YOLO(model_path)
#
# # Threshold for object detection
# threshold = 0.5
#
# # class_name_dict = {0: 'pessoa'}
# class_name_dict = {0: 'comedouro'}
#
# # Início vídeo na webcam
# cap = cv2.VideoCapture(0)  # 0 = webcam padrão
#
# # Pega as propriedades do vídeo
# frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
# frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#
# output_filename = "output_video.mp4"
# fourcc = cv2.VideoWriter_fourcc(*'MP4V')
#
# fps = 7  # FPS ajustado para a câmera Logitech (USB)
# out = cv2.VideoWriter(output_filename, fourcc, fps, (frame_width, frame_height))
# # FIM VÍDEO POR WEBCAM
#
#
# # Variáveis para controlar o zoom
# zoom_factor = 1.0
# zoom_step = 0.1
#
# # Função para aplicar zoom em um frame
# def apply_zoom(frame, zoom_factor, center=None):
#     height, width = frame.shape[:2]
#
#     if center is None:
#         center = (width // 2, height // 2)
#
#     new_width, new_height = int(width / zoom_factor), int(height / zoom_factor)
#
#     x1 = max(0, center[0] - new_width // 2)
#     y1 = max(0, center[1] - new_height // 2)
#     x2 = min(width, center[0] + new_width // 2)
#     y2 = min(height, center[1] + new_height // 2)
#
#     zoomed_frame = frame[y1:y2, x1:x2]
#
#     return cv2.resize(zoomed_frame, (width, height))
#
# # capturas_dir = 'capturas'
# # if not os.path.exists(capturas_dir):
# #     os.makedirs(capturas_dir)
# #
# # def captura_salva_frames(frame, frame_count):
# #     if frame_count % 25 ==0:  # Mudar aqui para variar a quantidade de frames
# #         image_filename = os.path.join(capturas_dir, f"Capturas_funcionamento_{frame_count:04d}.jpg")
# #         cv2.imwrite(image_filename, frame)
# #
# # # Set the frame count for capturing and saving  frames
# # frame_count = 0
# #
#
# # Configurações do vídeo de saída
# output_filename = "output_video.mp4"
# fourcc = cv2.VideoWriter_fourcc(*'MP4V')
# fps = 7
# frame_width = 640  # Largura padrão
# frame_height = 480  # Altura padrão
# out = cv2.VideoWriter(output_filename, fourcc, fps, (frame_width, frame_height))
#
# # Carregar o modelo YOLO
# model_path = os.path.join('.', 'runs', 'detect', 'train9', 'weights', 'best.pt')
# model = YOLO(model_path)
#
# # Iniciar a captura da webcam
# cap = cv2.VideoCapture(0)
#
# # Variáveis para controlar o zoom
# zoom_factor = 1.0
# zoom_step = 0.1
#
# # Loop principal
# while True:
#     ret, frame = cap.read()
#
#     if not ret:
#         break
#
#     # Aplicar zoom na ROI antes da detecção dos objetos
#     frame_area_deteccao_zoomed = apply_zoom(frame, zoom_factor, center=None)
#
#     # Substitui a ROI original pelo ROI com zoom no frame
#     frame = frame_area_deteccao_zoomed
#
#     # Detecção de objetos
#     results = model(frame)[0]
#
#     for result in results.boxes.data.tolist():
#         x1, y1, x2, y2, score, class_id = result
#
#         if score > threshold:
#             cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 4)
#             cv2.putText(frame, class_name_dict[int(class_id)].upper(), (int(x1), int(y1 - 10)),
#                         cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)
#
#     # Mostrar o frame
#     cv2.imshow('Deteccao de comedouros - Aviario', frame)
#
#     # Escrever o frame no vídeo de saída
#     out.write(frame)
#
#     # Incrementar o contador de frames
#     frame_count += 1
#
#     # Chamar a função de captura (salvar) de frames
#     captura_salva_frames(frame, frame_count)
#
#     # Verificar as teclas pressionadas
#     key = cv2.waitKey(1) & 0xFF
#     if key == ord('q'):
#         break
#     elif key == ord('z'):
#         zoom_factor -= zoom_step
#         if zoom_factor < 1.0:
#             zoom_factor = 1.0
#     elif key == ord('a'):
#         zoom_factor += zoom_step
#         if zoom_factor > 10.0:
#             zoom_factor = 10.0
#     elif key == ord('r'):
#         new_width = int(input("Digite a largura desejada do vídeo: "))
#         new_height = int(input("Digite a altura desejada do vídeo: "))
#         frame_width = new_width
#         frame_height = new_height
#         out.release()  # Liberar o objeto VideoWriter anterior
#         out = cv2.VideoWriter(output_filename, fourcc, fps, (frame_width, frame_height))
#
# cap.release()
# out.release()
# cv2.destroyAllWindows()
