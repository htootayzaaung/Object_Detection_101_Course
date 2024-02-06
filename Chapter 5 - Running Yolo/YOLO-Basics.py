from ultralytics import YOLO
import cv2
 
model = YOLO('../Models/yolov8n.pt')
results = model("../Images/The_War_Room.jpg", show=True)
cv2.waitKey(0)