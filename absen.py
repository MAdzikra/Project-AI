import cv2
import numpy as np
import os
from datetime import datetime
import tkinter as tk
from tkinter import simpledialog
from PIL import Image, ImageTk

model = cv2.face.EigenFaceRecognizer_create()
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
cam = cv2.VideoCapture(0)
model = cv2.face.EigenFaceRecognizer_create()

def train_existing_data():
    """Fungsi untuk melatih model dengan data yang sudah ada."""
    faces, ids = [], []
    for root, dirs, files in os.walk('dataset'):
        for dir in dirs:
            person_path = os.path.join(root, dir)
            person_id = int(dir.split('_')[1])
            for file in os.listdir(person_path):
                img_path = os.path.join(person_path, file)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                faces.append(np.array(img))
                ids.append(person_id)
    model.train(faces, np.array(ids))
    print("Model berhasil dilatih dengan data yang sudah ada!")

train_existing_data()

def resize_image(image, size=(200, 200)):
    """Fungsi untuk mengubah ukuran gambar."""
    return cv2.resize(image, size, interpolation=cv2.INTER_LINEAR)

def capture_face():
    """Fungsi untuk menangkap wajah dan menyimpannya."""
    identity = simpledialog.askstring("Input", "Masukkan ID Pengguna:")
    if identity:
        path = f'dataset/user_{identity}'
        if not os.path.exists(path):
            os.makedirs(path)
        count = 0
        while count < 30:
            ret, frame = cam.read()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)
            for (x, y, w, h) in faces:
                face_img = resize_image(gray[y:y+h, x:x+w])
                cv2.imwrite(f"{path}/{count}.jpg", face_img)
                count += 1
            show_frame(frame)
        print("Wajah berhasil direkam!")

def train_faces():
    """Fungsi untuk melatih model dengan gambar yang sudah ditangkap."""
    faces, ids = [], []
    for root, dirs, files in os.walk('dataset'):
        for dir in dirs:
            person_path = os.path.join(root, dir)
            person_id = int(dir.split('_')[1])
            for file in os.listdir(person_path):
                img_path = os.path.join(person_path, file)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                img = resize_image(img)
                faces.append(np.array(img))
                ids.append(person_id)
    model.train(faces, np.array(ids))
    print("Model berhasil dilatih!")

def recognize_face():
    """Fungsi untuk mengenali wajah dan mencatat absensi."""
    ret, frame = cam.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        face_img = resize_image(gray[y:y+h, x:x+w])
        id, conf = model.predict(face_img)
        accuracy = (1 - (conf / 4000)) * 100  
        if conf < 5000:
            now = datetime.now()
            time_string = now.strftime("%d/%m/%Y %H:%M:%S")
            with open("absensi.txt", "a") as file:
                file.write(f"ID {id} hadir pada {time_string} dengan akurasi {accuracy:.2f}%\n")
            print(f"ID {id} hadir pada {time_string} dengan akurasi {accuracy:.2f}%")
    show_frame(frame)


def show_frame(frame=None):
    """Fungsi untuk menampilkan frame kamera di GUI."""
    if frame is not None:
        cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
        img = Image.fromarray(cv2image)
        imgtk = ImageTk.PhotoImage(image=img)
        display_img.imgtk = imgtk
        display_img.configure(image=imgtk)
    display_img.after(10, show_frame)

root = tk.Tk()
root.title("Sistem Absensi Otomatis")

frame = tk.Frame(root)
frame.pack()

display_img = tk.Label(root)
display_img.pack()

btn_capture = tk.Button(frame, text="Rekam Wajah", command=capture_face)
btn_capture.pack(side=tk.LEFT)

btn_train = tk.Button(frame, text="Latih Model", command=train_faces)
btn_train.pack(side=tk.LEFT)

btn_recognize = tk.Button(frame, text="Absen", command=recognize_face)
btn_recognize.pack(side=tk.LEFT)

root.mainloop()
