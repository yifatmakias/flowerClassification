from tkinter.ttk import Separator

from keras.preprocessing import image
import numpy as np
import os
import pandas as pd
import tkinter as tk
from tkinter import filedialog, messagebox


def load_model(path):
    # load model and weights
    from keras.models import load_model
    loaded_model = load_model(path)
    return loaded_model


def insert(df, row):
    insert_loc = df.index.max()

    if np.isnan(insert_loc):
        df.loc[0] = row
    else:
        df.loc[insert_loc + 1] = row


def predict(model, path):
    # activate trained model on a path that contains flower photos
    flowers = ['daisy', 'dandelion', 'rose', 'sunflower', 'tulip']
    df = pd.DataFrame(columns=['image_name', 'classification', 'definition'])
    images = [i for i in os.listdir(path)]
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    for image_name in images:
        try:
            test_image = image.load_img(path + '\\' + image_name, target_size=(128, 128))
            test_image = image.img_to_array(test_image)
            test_image = np.expand_dims(test_image, axis=0)
            # predict the result
            result = model.predict(test_image)
            result[0] = np.around(result[0], decimals=3)
            definition = get_def(flowers, result)
            insert(df, [image_name, result, definition])
        except:
            print(image_name + "is not a valid image")
    return df


def get_def(flowers, results):
    defi = ""
    for i in range(len(results[0])):
        if not results[0][i] == 0:
            defi += flowers[i] + ":" + str(results[0][i]) + ","
    return defi[:-1]


def start(file, folder):
    flag = False
    if not file.get() or not folder.get():
        messagebox.showerror("Error", "The given path is empty")
        flag = True
    if not os.path.isdir(folder.get()):
        messagebox.showerror("Error", "The given path to image folder is incorrect")
        flag = True
    if not flag:
        try:
            model = load_model(file.get())
        except:
            messagebox.showerror("Error", "Oops, there was an error loading the model")
        if model is not None:
            df = predict(model, folder.get())
            present_table(df)


def save_to_csv(df):
    f = filedialog.asksaveasfile(mode='w', defaultextension=".csv")
    if f is None:  # asksaveasfile return `None` if dialog closed with "cancel".
        return
    df.to_csv(f, index=None)


def present_table(data):
    tree = tk.ttk.Treeview(frame2, columns=(1, 2, 3), height=frame2.winfo_height(), show="headings")
    tree.pack(side='left')

    tree.heading(1, text="Image Name")
    tree.heading(2, text="Prediction")
    tree.heading(3, text="Definition")

    tree.column(1)
    tree.column(2)
    tree.column(3)

    scroll = tk.ttk.Scrollbar(frame2, orient="vertical", command=tree.yview)
    scroll.pack(side='right', fill='y')

    tree.configure(yscrollcommand=scroll.set)

    for index, row in data.iterrows():
        tree.insert("", tk.END, values=(row[0], row[1], row[2]))
    btn_save = tk.Button(frame2, text="Save to CSV", command=lambda: save_to_csv(data))
    btn_save.pack(side='right')


def clear():
    global frame2
    frame2.destroy()
    frame2 = tk.Frame(window)
    frame2.pack(side='bottom', fill='both', expand=True)


def browse(look_for, var):
    try:
        if look_for == 'd':
            path = filedialog.askdirectory(parent=window, title='Choose directory')
        if look_for == 'f':
            path = filedialog.askopenfile(parent=window, title='Choose a file').name
        var.set(path)
    except:
        pass


window = tk.Tk()
file_path = tk.StringVar()
folder_path = tk.StringVar()
window.title("Flower Classification")
window.geometry('800x600')

frame1 = tk.Frame(window)
separator = Separator(window, orient='horizontal')
frame2 = tk.Frame(window)

frame1.pack(side='top', fill='both', expand=True)
separator.pack(side='top', fill='x')
frame2.pack(side='bottom', fill='both', expand=True)

# Frame 1 (left side) components insertion
lbl_model = tk.Label(frame1, text="Path to model file:")
lbl_model.grid(column=0, row=1, padx=3, pady=3)
entry_model = tk.Entry(frame1, width=60, textvariable=file_path)
entry_model.grid(column=1, row=1, padx=3, pady=3)
btn_model = tk.Button(frame1, text="Browse...", command=lambda: browse('f', file_path))
btn_model.grid(column=2, row=1, padx=3, pady=3)
lbl_image = tk.Label(frame1, text="Path to image folder:")
lbl_image.grid(column=0, row=2, padx=3, pady=3)
entry_image = tk.Entry(frame1, width=60, textvariable=folder_path)
entry_image.grid(column=1, row=2, padx=3, pady=3)
btn_image = tk.Button(frame1, text="Browse...", command=lambda: browse('d', folder_path))
btn_image.grid(column=2, row=2, padx=3, pady=3)
btn_start = tk.Button(frame1, text="Predict", command=lambda: start(file_path, folder_path))
btn_start.grid(column=0, row=3, padx=3, pady=10)
btn_start = tk.Button(frame1, text="Clear", command=clear)
btn_start.grid(column=1, row=3, padx=3, pady=10)

window.mainloop()
