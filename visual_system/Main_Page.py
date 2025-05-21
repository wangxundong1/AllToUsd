from tkinter import *
from tkinter import messagebox, W, E
from tkinter.filedialog import *
from visual_system.Application import Application
from visual_system.Model_Visual_Page import Model_Visual
from AllToUsd.DaeToUsdConverter import DaeToUsdConverter
from AllToUsd.AllToDaeConverter import AllToDae
import time


class Main_Page():
    def __init__(self):
        root = Tk()
        root.title("ALLTOUSD_GUI")
        root.geometry("800x300+1000+500")
        root.configure(bg="#f0f0f0")  # 设置背景颜色

        self.app = Application(master=root)
        self.create_main()

        root.mainloop()

    def choose_file(self):
        f = askopenfilename(title="选择模型",initialdir="D:/wangxundong/3D Asset")
        self.app.input_var.set(f)

    def alltousd(self):
        input_file = self.app.input_var.get()
        alltodae = AllToDae(input_file)
        if not alltodae.AllToDaeConvert():
            messagebox.showinfo("Message","AllToDae is error")
        else:
            dae_file = alltodae.output_file
            starttime = time.time()
            daetousd = DaeToUsdConverter(dae_file)
            filepath = daetousd.convert_dae_to_usd()
            # alltodae.deleteDae()
            endtime = time.time()
            usetime = endtime-starttime
            messagebox.showinfo("Message",f"USD 文件已保存到: {filepath}\n转换用时：{usetime:.4f}秒")
            self.app.output_var.set(filepath)

    def create_main(self):
        self.app.pack()
        title = self.app.create_label(30,1,None,"black",font=("Arial", 16,"bold"))
        title["text"]="基于OpenUSD的三维转换工具"
        title.grid(row=0,column=0,columnspan=5,pady=(0,5))

        introduce = self.app.create_label(80,3,None,"black",font=("Arial", 9))
        introduce["text"]="支持加载的三维格式：FBX、DAE、GLTF、3DS、BVH、ASE、OBJ、IFC、DXF、LWO、NDO\n" \
                          "LWS、LXO、X、AC、MS3D、COB、SCN、CSM、STL、MD1、MD2、MD3、PK3、MDC\n" \
                          "MD5、SMD、VTA、OGEX、3D、B3D、Q3D、Q3S、NFF、OFF、RAW、TER、MDL、HMP、BLEND"
        introduce.grid(row=1,column=0,columnspan=5,pady=(0,10))

        input = self.app.create_label(12,2,None,"black")
        input["text"]="输入文件:"
        input.grid(row=2,column=0,sticky=W)

        output = self.app.create_label(12,2,None,"black")
        output["text"] = "输出文件:"
        output.grid(row=3,column=0,sticky=W,pady=(0,10))

        entry_input = Entry(self.app,textvariable=self.app.input_var,width=46,font=("Arial", 10))
        entry_input.grid(row=2,column=1,columnspan=3)

        entry_output = Entry(self.app,textvariable=self.app.output_var,width=46, font=("Arial", 10))
        entry_output.grid(row=3,column=1,columnspan=3,pady=(0,10))

        
        file = self.app.create_btn("选择模型文件",command=self.choose_file)
        file.grid(row=2,column=4,sticky=E)

        convert = self.app.create_btn("转换",command=self.alltousd)
        convert["width"]=15
        convert.grid(row=4,column=0,columnspan=2,pady=(10,0))

        exit = self.app.create_btn("退出",self.app.exit)
        exit["width"]=15
        exit.grid(row=4,column=3,columnspan=2,pady=(10,0))

        show = self.app.create_btn("模型展示",command=lambda:self.show_page(self.app.output_var.get(),self.app.input_var.get()))
        show["width"]=30
        show.grid(row=5,column=1,columnspan=3,pady=(10, 0))

    def show_page(self,usd_file,all_file):
        print(usd_file)
        Model_Visual(usd_file,all_file)