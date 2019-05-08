import tkinter
import cv2
import numpy as np
import sympy as sp
from io import BytesIO
import PIL.Image, PIL.ImageTk
import time
from solver import Solver
from PIL import Image, ImageTk
from filters import *


class GUI:
    def __init__(self, window, title, display_filtered=False, video_source=1):
        self.window = window
        self.window.title(title)
        self.window.configure(background='white')
        self.video = VideoCapture(video_source)
        self.display_filtered = display_filtered
        self.filter = solver.filter

        # Create a canvas for video
        self.canvas = tkinter.Canvas(window, width=self.video.width, height=self.video.height)
        self.canvas.pack()

        # Create calculate button
        self.calculate_btn=tkinter.Button(window, text="Calculate", width=50, command=self.calculate)
        self.calculate_btn.pack(anchor=tkinter.CENTER, expand=True)

        # Create output labels
        self.latex_label = tkinter.Label(window)
        self.latex_label.pack(anchor=tkinter.CENTER, pady=10)

        self.equation = tkinter.Label(window)
        self.equation.pack(anchor=tkinter.CENTER, pady=10)

        self.result = tkinter.Label(window)
        self.result.pack(anchor=tkinter.CENTER)

        # Update method will be called every delay milliseconds
        self.delay = 15
        self.update()

        self.window.mainloop()

    def calculate(self):
        # Get a frame from the video source
        ret, frame = self.video.get_frame()
        latex, rez = solver.solve(frame)

        # Update latex label
        self.latex_label.configure(text='LaTex: ' + latex)

        # Update image
        obj = BytesIO()
        sp.preview('$$'+latex+'$$', viewer='BytesIO', output='png', outputbuffer=obj)
        obj.seek(0)
        self.equation.img = ImageTk.PhotoImage(Image.open(obj))
        self.equation.config(text='Image', image=self.equation.img)

        # Update result label
        self.result.configure(text='Result: ' + str(rez))

    def update(self):
        # Get a frame from the video source
        ret, frame = self.video.get_frame()

        # Display image as cnn input
        if self.display_filtered:
            frame = self.filter(frame)
        else:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if ret:
           self.photo = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(frame))
           self.canvas.create_image(0, 0, image=self.photo, anchor=tkinter.NW)

        self.window.after(self.delay, self.update)


class VideoCapture:

    def __init__(self, video_source=0):
        self.video = cv2.VideoCapture(video_source)
        if not self.video.isOpened():
            raise ValueError("Unable to open video source", video_source)

        self.width = self.video.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.height = self.video.get(cv2.CAP_PROP_FRAME_HEIGHT)

    def get_frame(self):

        if self.video.isOpened():
            ret, frame = self.video.read()
            if ret: return ret, frame

        return False, None

    def __del__(self):
        # Release the video source when the object is destroyed
        if self.video.isOpened():
            self.video.release()


if __name__ == '__main__':
    solver = Solver(
        model_dir='models/mnist_convnet_model_48x48',
        labels_file='utility/labels.txt',
        bb_plot=True,
        image_filter=basic_filter
    )

    GUI(
        window=tkinter.Tk(),
        title='Math solver',
        display_filtered=True
    )
