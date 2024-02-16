"""
Package Configurations
python                    3.11.3
simpy                     4.0.1
"""

import io
from tkinter import *
from PIL import Image, ImageTk




class GUI():
    def __init__(self, image_list):
        self.n_show = len(image_list)
        self.RANDOM_SEED = 42
        self.tk = Tk()
        self.tk.title("SSY Stacking - Jiwon Baek")
        # self.tk.geometry("1960x1080+400+200")
        # self.tk.resizable(False, False)
        self.tk_images = []

        self.get_image(image_list)
        self.current_image = 0
        self.frame1 = LabelFrame(self.tk, text="SSY Stacking")
        self.frame1.grid(column=0, row=0)
        self.gantt = Label(self.frame1, text="SSY Stacking")
        self.gantt.grid(column=0, row=0, sticky=N + E + W + S)

        self.update()
        self.tk.mainloop()

    def get_image(self, images):
        for i in range(self.n_show):
            self.tk_images.append(ImageTk.PhotoImage(Image.open(images[i])))

            # self.image[i] = Image.open(io.BytesIO(self.image_bytes[i]))  # 0으로 초기화하긴 했지만 byte가 들어올거라 괜찮음
            # self.tk_images[i] = ImageTk.PhotoImage(self.image[i])

    def update(self):

        self.gantt.config(image=self.tk_images[self.current_image])
        self.current_image += 1

        # tk.after는 인자로 받은 함수를 계속해서 callback함
        # 재귀적으로 호출함으로써 계속 동작하게 만들 수 있음
        if self.current_image >= self.n_show:
            self.current_image = 0

        self.tk.after(100, self.update)
        # if self.current_image == 0:
        #     self.tk.after(1000, self.update)  # Pause for 2000 milliseconds (2 seconds)
        # else:
        #     self.tk.after(5000, self.update)  # Pause for 2000 milliseconds (2 seconds)


if __name__ == "__main__":
    gui = GUI()
