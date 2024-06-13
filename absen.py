import cv2
import os
import numpy as np
import csv
import time
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk  # Menggunakan PIL untuk memproses gambar

# Function to capture images
def capture_images(name, npm, save_path='dataset'):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    user_folder = os.path.join(save_path, f"{name}_{npm}")
    if not os.path.exists(user_folder):
        os.makedirs(user_folder)

    camera = cv2.VideoCapture(0)
    face_ref = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    count = 0

    # Countdown before starting
    start_time = time.time()
    while True:
        ret, frame = camera.read()
        if not ret:
            break
        
        elapsed_time = int(time.time() - start_time)
        countdown = 5 - elapsed_time
        if countdown > 0:
            cv2.putText(frame, f"Starting in {countdown}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
        else:
            break
        
        cv2.imshow("Capturing Images", frame)
        cv2.waitKey(1)

    while count < 50:
        ret, frame = camera.read()
        if not ret:
            break
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_ref.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=1)

        for (x, y, w, h) in faces:
            face = gray[y:y+h, x:x+w]
            count += 1
            cv2.imwrite(os.path.join(user_folder, f"{name}_{npm}_{count}.jpg"), face)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        
        cv2.imshow("Capturing Images", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    camera.release()
    cv2.destroyAllWindows()

# Function to train the model
def train_model(dataset_path='dataset', model_save_path='models/face_recognition_model.xml'):
    face_recognizer = cv2.face.LBPHFaceRecognizer_create()
    faces = []
    labels = []
    label_dict = {}

    for idx, user_folder in enumerate(os.listdir(dataset_path)):
        user_path = os.path.join(dataset_path, user_folder)
        if not os.path.isdir(user_path):
            continue
        
        name, npm = user_folder.split('_')
        label_dict[idx] = (name, npm)

        for image_name in os.listdir(user_path):
            image_path = os.path.join(user_path, image_name)
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            faces.append(image)
            labels.append(idx)
    
    face_recognizer.train(faces, np.array(labels))
    if not os.path.exists('models'):
        os.makedirs('models')
    face_recognizer.save(model_save_path)

    # Save label dictionary
    with open(model_save_path.replace('.xml', '_labels.csv'), 'w', newline='') as f:
        writer = csv.writer(f)
        for idx, (name, npm) in label_dict.items():
            writer.writerow([idx, name, npm])

# Function to recognize faces and record attendance
def recognize_and_attend(model_path='models/face_recognition_model.xml', output_csv='attendance.csv'):
    face_recognizer = cv2.face.LBPHFaceRecognizer_create()
    face_recognizer.read(model_path)

    # Load label dictionary
    label_dict = {}
    with open(model_path.replace('.xml', '_labels.csv'), 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            label_dict[int(row[0])] = (row[1], row[2])

    camera = cv2.VideoCapture(0)
    face_ref = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    attendance_set = set()
    threshold = 60  # Confidence threshold
    recognized = False

    while True:
        ret, frame = camera.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_ref.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=2)

        for (x, y, w, h) in faces:
            face = gray[y:y+h, x:x+w]
            label, confidence = face_recognizer.predict(face)
            name, npm = "Unknown", ""
            if confidence < threshold:
                name, npm = label_dict.get(label, ("Unknown", ""))
                attendance_set.add((name, npm))
                recognized = True

            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
            cv2.putText(frame, f"{name} {round(confidence, 2)}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
        cv2.imshow("Attendance", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    camera.release()
    cv2.destroyAllWindows()

    if recognized:
        for name, npm in attendance_set:
            with open(output_csv, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([name, npm, time.strftime("%Y-%m-%d %H:%M:%S")])
                # Display welcome message
                messagebox.showinfo("Welcome", f"Selamat datang {name} NPM {npm}!!")
    else:
        messagebox.showinfo("Unrecognized", "Wajah tidak dikenali!!")

# Functions to connect with GUI buttons
def register():
    name = entry_name.get()
    npm = entry_npm.get()
    capture_images(name, npm)
    messagebox.showinfo("Info", f"User {name} registered successfully!")

def train():
    train_model()
    messagebox.showinfo("Info", "Model trained successfully!")

def attend():
    recognize_and_attend()
    messagebox.showinfo("Info", "Attendance recorded successfully!")

# Setting up the GUI
app = tk.Tk()
app.title("Attendance System")

# Styling the GUI
app.configure(bg='#222831')

# Function to set font and size for labels, entries, and buttons
def set_font(widget, size):
    return ('Helvetica', size)

label_font = set_font(tk.Label, 12)
entry_font = set_font(tk.Entry, 12)
button_font = set_font(tk.Button, 12)
title_font = ('Helvetica', 20, 'bold')

# Create title label
title_label = tk.Label(app, text="AUTOMATIC ATTENDANCE USING FACE RECOGNITION", font=title_font, bg='#31363F', fg='white', pady=20, padx=20)
title_label.grid(row=0, columnspan=3, pady=(20, 10))

# Create labels and entries
tk.Label(app, text="Name", font=label_font, bg='#222831', fg='white').grid(row=1, column=0, padx=10, pady=10)
tk.Label(app, text="NPM", font=label_font, bg='#222831', fg='white').grid(row=2, column=0, padx=10, pady=10)

# Function to make the entry fields rounded
def round_entry(entry, size=15):
    entry.configure(relief='flat', highlightthickness=1, highlightbackground='#424242')
    entry.configure(bd=0, highlightcolor='#424242', highlightbackground='#424242', insertbackground='#E0E0E0', insertwidth=2)
    entry.bind('<FocusIn>', lambda e: e.widget.config(highlightbackground='#E91E63', highlightcolor='#E91E63'))
    entry.bind('<FocusOut>', lambda e: e.widget.config(highlightbackground='#424242', highlightcolor='#424242'))

entry_name = tk.Entry(app, font=entry_font, width=30, bg='#424242', fg='white', insertbackground='white')
entry_npm = tk.Entry(app, font=entry_font, width=30, bg='#424242', fg='white', insertbackground='white')
round_entry(entry_name)
round_entry(entry_npm)

entry_name.grid(row=1, column=1, padx=10, pady=10)
entry_npm.grid(row=2, column=1, padx=10, pady=10)

# Image
img_path = 'img/absensi12.png' 
img = Image.open(img_path)
img = img.resize((200, 150), Image.LANCZOS)
img = ImageTk.PhotoImage(img)

# Label for the image
img_label = tk.Label(app, image=img, bg='#222831')
img_label.grid(row=1, column=2, rowspan=2, padx=10, pady=10)

# Buttons
tk.Button(app, text="Register", font=button_font, command=register, bg='#E91E63', fg='white', relief='flat').grid(row=3, column=0, padx=10, pady=20)
tk.Button(app, text="Train Model", font=button_font, command=train, bg='#2196F3', fg='white', relief='flat').grid(row=3, column=1, padx=10, pady=20)
tk.Button(app, text="Attend", font=button_font, command=attend, bg='#FF5722', fg='white', relief='flat').grid(row=3, column=2, padx=10, pady=20)

app.mainloop()
