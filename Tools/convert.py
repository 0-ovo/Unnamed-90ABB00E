# -*- coding: utf-8 -*-
# 此文件用於轉換Windows的反斜杠(异端)
# 也可直接運行
from pathlib import PurePosixPath

def convertToPosixStyle(
    *args : str
    ) -> str:
    """
    将路径拼接后转换为Posix风格
    实现不同系统的路径一致处理(Windows 2000起兼容Posix风格)
    Args:
        *args(列表:字符串):要处理的路径
        
    Return:
        字符串:处理后的路径"""
    
    def delete_commas(args : str):
        """删除两端单/双引号"""

        return args.strip("\"\'")
    x = map(delete_commas, args)
    path = str(PurePosixPath(*x))
    return path.replace("\\","/").replace("//","/")

if __name__ == "__main__":

    import tkinter
    root = tkinter.Tk()
    root.title("Path Converter")
    root.geometry("340x320")

    # 设置窗口名稱
    label = tkinter.Label(root, text="Path Converter")
    label.grid(row = 0,pady=10)

    # 设置输入框
    entry = tkinter.Entry(root, width=45)
    entry.grid(row = 1,padx = (8,0),pady=10)

    
    # 绑定回车键事件
    entry.bind("<Return>", lambda event: update_result())

    # 设置轉換按钮
    convert = tkinter.Button(root, text="Convert", command=lambda:update_result())
    convert.grid(row = 2,padx=(0,225))

    # 设置清除按钮
    clear = tkinter.Button(root, text="Clear", command=lambda:entry.delete(0,"end"))
    clear.grid(row = 2,padx = (0,115))

    # 設置粘貼按鈕
    paste = tkinter.Button(root, text="Paste", command=lambda:pastecmd(),bg="yellow", fg="black")
    paste.grid(row = 2,padx=(0,20))


    # 设置一键转换按钮
    allinonebottom = tkinter.Button(root, text="One click", command=lambda:allinone(),bg="green", fg="white")
    allinonebottom.grid(row = 2,padx=(98,0))

    # 设置文件选择按钮
    file = tkinter.Button(root, text="FileSelect", command=lambda:fileselect(),bg="blue", fg="white")
    file.grid(row = 2,padx=(239,0))

    # 結果標簽
    result_label = tkinter.Label(root, text="Converted Path:")
    result_label.grid(row = 4,pady=10)

        
    # 设置结果显示框
    result_entry = tkinter.Entry(root, width=45)
    result_entry.grid(row = 5,padx = (8,0),pady=10)
    result_entry.config(state="readonly")

    # 設置複製按鈕
    copy = tkinter.Button(root, text="Copy", command=lambda:updatechipboard(result_entry.get()))
    copy.grid(row = 6,pady=10)


    # 一鍵轉換
    def allinone():
        """一鍵轉換"""
        entry.delete(0,"end")
        entry.insert(0,root.clipboard_get())
        update_result()
        updatechipboard(result_entry.get())

    # 更新結果框
    def update_result():
        
        input_path = entry.get()
        result = convertToPosixStyle(*input_path.split("|"))
        result_entry.config(state="normal")
        result_entry.delete(0, "end")
        result_entry.insert(0, f"\"{result}\"")
        result_entry.config(state="readonly")

    # 更新剪切板
    def updatechipboard(string : str):
        root.clipboard_clear()
        root.clipboard_append(string)

    # 粘贴到输入框
    def pastecmd():
        if entry.get() != "" and entry.get()[-1] != "|":
            entry.insert("end","|")
            
        entry.insert("end",f"{root.clipboard_get()}|")
        update_result()

    # 文件选择
    def fileselect():
        import tkinter.filedialog as filedialog
        import os
        file_path = filedialog.askopenfilename(initialdir = entry.get() if os.path.exists(entry.get()) else ".", title = "Select a file")
        if file_path:
            entry.delete(0, "end")
            entry.insert(0, file_path)
            update_result()
            updatechipboard(result_entry.get())

    root.mainloop()
