from ultralytics import YOLO
import cv2
import math 
# start webcam  
cap = cv2.VideoCapture(0)
cap.set(3, 2000) #padrão inicial: 640
cap.set(4, 1080) #padrão inicial: 480

# model
model = YOLO("yolo-Weights/yolov8n.pt")

# object classes
classNames = ["pessoa", "bicicleta", "carro", "motocicleta", "avião", "ônibus", "trem", "caminhão", "barco",
                "semáforo", "hidrante", "placa de pare", "parquímetro", "banco", "pássaro", "gato",
                "cachorro", "cavalo", "ovelha", "vaca", "elefante", "urso", "zebra", "girafa", "mochila", "guarda-chuva",
                "bolsa", "gravata", "mala", "frisbee", "esquis", "snowboard", "bola esportiva", "pipa", "taco de beisebol",
                "luva de beisebol", "skate", "prancha de surfe", "raquete de tênis", "garrafa", "taça de vinho", "copo",
                "garfo", "faca", "colher", "tigela", "banana", "maca", "sanduíche", "laranja", "brócolis",
                "cenoura", "cachorro-quente", "pizza", "donut", "bolo", "cadeira", "sofa", "planta em vaso", "cama",
                "mesa de jantar", "vaso sanitário", "monitor de tv", "laptop", "mouse", "controle remoto", "teclado", "celular",
                "micro-ondas", "forno", "torradeira", "pia", "geladeira", "livro", "relogio", "vaso", "tesoura",
                "ursinho de pelúcia", "secador de cabelo", "escova de dentes"
              ]
while True:
    success, img = cap.read()
    results = model(img, stream=True)

    # coordinates
    for r in results:
        boxes = r.boxes

        for box in boxes:
            # bounding box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2) # convert to int values

            # put box in cam
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)

            # confidence
            confidence = math.ceil((box.conf[0]*100))/100
            print("Confidence --->",confidence)

            # class name
            cls = int(box.cls[0])
            print("Class name -->", classNames[cls])

            # object details
            org = [x1, y1]
            font = cv2.FONT_HERSHEY_SIMPLEX
            fontScale = 1
            color = (0,0,0)
            thickness = 2
            padding = 10

            # Calcular o tamanho do texto
            (text_width, text_height), baseline = cv2.getTextSize(classNames[cls], font, fontScale, thickness)

            # Ajustar a posição do texto com padding
            org = [org[0] + padding, org[1] + text_height + padding]

            cv2.putText(img, classNames[cls], org, font, fontScale, color, thickness, padding)

    cv2.imshow('Webcam', img)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()