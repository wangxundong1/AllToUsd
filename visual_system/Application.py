from tkinter import *
from tkinter import messagebox

class  Application(Frame):
    def __init__(self, master=None):    # master表示当前组件的父组件
        super().__init__(master)        # super表示父类的定义
        self.master = master
        self.input_var = StringVar()    # 输入文件
        self.output_var = StringVar()   # 输出文件
    
    def create_btn(self,text,command):
        """创建按钮"""
        btn = Button(self,text=text,command=command)
        return btn
    
    def create_text(self,width,height,bg="grey"):
        """创建文本"""
        text = Text(self,width=width,height=height,bg=bg)
        return text
    
    def create_label(self,width,height,bg="grey",fg="white"):
        """创建标签"""
        label = Label(self,width=width,height=height,bg=bg,fg=fg)
        return label

    def test(self):
        messagebox.showinfo("Message","点击成功")

    def exit(self):
        self.master.destroy()





