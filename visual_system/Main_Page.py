from tkinter import *
from tkinter import messagebox
from tkinter.filedialog import *
from visual_system.Application import Application
from visual_system.Model_Visual_Page import Model_Visual
from AllToUsd.DaeToUsdConverter import DaeToUsdConverter
from AllToUsd.AllToDaeConverter import AllToDae



class Main_Page():
    def __init__(self):
        root = Tk()
        root.title("ALLTOUSD_GUI")
        root.geometry("600x200+1000+500")

        self.app = Application(master=root)
        self.create_main()

        root.mainloop()

    def choose_file(self):
        f = askopenfilename(title="选择模型",initialdir="c:")
        self.app.input_var.set(f)

    def alltousd(self):
        input_file = self.app.input_var.get()
        alltodae = AllToDae(input_file)
        if not alltodae.AllToDaeConvert():
            messagebox.showinfo("Message","AllToDae is error")
        else:
            dae_file = alltodae.output_file
            daetousd = DaeToUsdConverter(dae_file)
            filepath = daetousd.convert_dae_to_usd()
            alltodae.deleteDae()
            messagebox.showinfo("Message",f"USD 文件已保存到: {filepath}")
            self.app.output_var.set(filepath)

    def create_main(self):
        self.app.pack()
        title = self.app.create_label(30,2,None,"black")
        title["text"]="基于OpenUSD的三维转换工具"
        title.grid(row=0,column=1,columnspan=3)

        input = self.app.create_label(12,2,None,"black")
        input["text"]="输入文件:"
        input.grid(row=1,column=0,sticky=W)

        output = self.app.create_label(12,2,None,"black")
        output["text"] = "输出文件:"
        output.grid(row=2,column=0,sticky=W)

        entry_input = Entry(self.app,textvariable=self.app.input_var,width=40)
        entry_input.grid(row=1,column=1,columnspan=3)

        entry_output = Entry(self.app,textvariable=self.app.output_var,width=40)
        entry_output.grid(row=2,column=1,columnspan=3)

        
        file = self.app.create_btn("选择你的文件",command=self.choose_file)
        file.grid(row=1,column=4,sticky=E,padx=5)

        convert = self.app.create_btn("Convert",command=self.alltousd)
        convert["width"]=15
        convert.grid(row=3,column=0,columnspan=2)

        exit = self.app.create_btn("Exit",self.app.exit)
        exit["width"]=15
        exit.grid(row=3,column=3,columnspan=2)

        show = self.app.create_btn("Show and Compare",command=lambda:self.show_page(self.app.output_var.get(),self.app.input_var.get()))
        show["width"]=30
        show.grid(row=4,column=1,columnspan=3,pady=(8, 0))

    def show_page(self,usd_file,all_file):
        print(usd_file)
        Model_Visual(usd_file,all_file)