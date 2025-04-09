import os
import subprocess
import tkinter as tk
from tkinter import messagebox
import psutil
from ctypes import windll, byref, c_ulong, create_unicode_buffer,c_int




# usdview的程序路径
USDVIEW_PATH = r"D:\Environment\OpenUSD\tools\usd\install_openusd\bin\usdview.cmd"

# 暂时写死的usd测试文件
USD_FILE = os.path.normpath(r"D:\code\py\DaeToUsd\results\test.usd") # os.path.normpath用于规范化路径（处理斜杠）

# 暂时写死的input测试文件
INPUT_FILE = os.path.normpath(r"D:\wangxundong\3D Asset\OBJ\backpack\backpack.obj")

class USD_Model():
    def __init__(self,usd_file,container,root):
        self.usd_file = usd_file
        self.process = None
        self.container = container
        self.root = root
        pass

    def close_usdview(self):
        if self.process:
            try:
                parent = psutil.Process(self.process.pid)
                children = parent.children(recursive=True)  # 获取所有子进程
                for child in children:
                    child.terminate()  # 终止子进程
                parent.terminate()  # 终止父进程
                self.process = None
                messagebox.showinfo("Message","usdview进程及其子进程已终止")
            except Exception as e:
                messagebox.showinfo("Message",f"终止失败: {e}")

    def embed_usdview(self):
        """
        使用usdview打开usd文件
        """
        if self.process:
            messagebox.showinfo("Message","当前存在正在展示的模型")
            return
        if not os.path.exists(USDVIEW_PATH):
            messagebox.showinfo("Message",f"错误: usdview 未找到于 {USDVIEW_PATH}")
            return
        if not os.path.exists(self.usd_file):
            messagebox.showinfo("Message",f"错误: USD 文件未找到于 {self.usd_file}")
            return

        try:
            # 启动usdview子进程（无控制台窗口）
            self.process = subprocess.Popen(
                [USDVIEW_PATH, self.usd_file],   # 命令行参数
                creationflags=subprocess.CREATE_NO_WINDOW   # 隐藏控制台窗口
            )
            # 延迟1秒后开始窗口嵌入操作，防止usdview进程未加载完成
            self.root.after(1000, lambda: self.find_and_embed_window(self.process.pid, retries=5))
        except Exception as e:
            messagebox.showinfo("Message",f"启动失败: {e}")

    def find_and_embed_window(self,pid, retries=5):
        """
        实现窗口嵌入
        pid: usdview进程的PID
        retries: 重复的次数
        """
        try:
            parent = psutil.Process(pid)
            child_pids = [child.pid for child in parent.children()] # 获得usdview进程的全部子进程PID
        except psutil.NoSuchProcess:
            child_pids = []

        # 遍历子进程 PID
        for child_pid in child_pids:
            # 通过PID查询窗口句柄
            hwnd = self.find_window_by_pid(child_pid)
            if hwnd:
                # 将窗口嵌入Tkinter容器中
                windll.user32.SetParent(hwnd, self.container.winfo_id())
                
                # 修改窗口样式
                style = windll.user32.GetWindowLongW(hwnd, -16) # 获取当前窗口样式（-16对应GWL_STYLE，指定获取当前窗口样式）  
                # print(f"转换前：\n{style}")
                style = c_int(style).value  # 确保转换为32位有符号整数
                # print(f"转换后：\n{style}")
                style &= ~0x80000000  # 移除弹弹窗样式（独立窗口属性）
                style &= ~0x00C00000  # 移除标题栏
                style |= 0x40000000   # 添加子窗口样式
                # print(f"修改后:\n{style}")
                
                # 设置新样式
                windll.user32.SetWindowLongW(hwnd, -16, c_int(style).value)
                #windll.user32.SetWindowLongW(hwnd, -16, style)
                windll.user32.MoveWindow(hwnd, 0, 0, 500, 400, True)    # 设置窗口尺寸  (0,0)左上角坐标;(500,400)窗口长宽;True立即重绘   
                break
            elif retries > 0:
                self.root.after(500, lambda: self.find_and_embed_window(pid, retries-1))  # 0.5s后再次尝试

    def find_window_by_pid(self,target_pid):
        hwnd = windll.user32.GetTopWindow(0)    # 获取顶层窗口
        while hwnd:
            # 获取窗口进程ID
            pid = c_ulong()
            windll.user32.GetWindowThreadProcessId(hwnd, byref(pid))       # hwnd当前窗口；byref(pid)用来接收输出参数即hwnd的PID
            if pid.value == target_pid:
                # 获取窗口类名
                buffer_class = create_unicode_buffer(255)   # 创建缓冲区
                windll.user32.GetClassNameW(
                    hwnd,           # 窗口
                    buffer_class,   # 接收类名的缓冲区
                    255             # 缓冲区大小
                )
                class_name = buffer_class.value
                # print(f"找到窗口: PID={pid.value}, 类名={class_name}")
                # 检查是否为Qt窗口（usdview基于Qt）
                if "Qt" in class_name and "Window" in class_name:
                    return hwnd

            hwnd = windll.user32.GetWindow(hwnd, 2) # 获取下一个窗口,2表示GW_HWNDNEXT，即下一个同级别的窗口
        return None

class ALL_Model():
    def __init__(self,all_file,container,root):
        self.all_file = all_file
        self.process = None
        self.container = container
        self.root = root
        
    def open_model(self):
        print("open_model")
    
    def close_model(self):
        print("close_model")

class Model_Visual():
    def __init__(self,usd_file,all_file):

        self.root = tk.Tk()
        self.root.title("Show and Compare")
        self.root.geometry("1200x800+800+250")

        # 创建用于嵌入模型的容器
        self.container = tk.Frame(self.root)
        self.container.pack(fill=tk.BOTH, expand=True)   # 填充整个窗口

        # 创建两个展示模型的实例
        self.input = ALL_Model(all_file,self.container,self.root)
        self.output = USD_Model(usd_file,self.container,self.root)

        # 创建按钮容器
        button_frame = tk.Frame(self.root)
        button_frame.pack(side=tk.BOTTOM, pady=10)  # 将容器固定在底部

        # 创建按钮
        btn_open_usd = tk.Button(button_frame, text="打开USD", command=self.output.embed_usdview)
        btn_open_usd["width"]=25
        btn_open_usd.grid(row=0,column=0,padx=(0,10))  
        btn_close_usd = tk.Button(button_frame, text="关闭USD", command=self.output.close_usdview)
        btn_close_usd["width"]=25
        btn_close_usd.grid(row=0,column=1,padx=(0,150)) 
        btn_open_all = tk.Button(button_frame, text="打开All", command=self.input.open_model)
        btn_open_all["width"]=25
        btn_open_all.grid(row=0,column=2,padx=(0,10))   
        btn_close_all = tk.Button(button_frame, text="关闭All", command=self.input.close_model)
        btn_close_all["width"]=25
        btn_close_all.grid(row=0,column=3) 



        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

        self.root.mainloop()
    
    def on_close(self):
        self.output.close_usdview()
        self.input.close_model()
        self.root.destroy()



