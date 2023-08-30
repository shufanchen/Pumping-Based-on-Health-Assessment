from tkinter import *
from data_process import *
import tkinter.filedialog
from knn_pred import *
from ttkbootstrap import Style
from PIL import ImageTk, Image
from tkinter import ttk
from detection import detection_abn
from tem_pre import train_con
from tem_pre import pred
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像中汉字显示为方格的问题



def newwind():
    winNew = Toplevel(root)
    winNew.geometry('900x800')
    winNew.title('数据处理模块')
    msg1 = Label(winNew, text='提示:原始数据文件路径是存放同一类型的水泵传感器数据\n(xlsx或csv格式)的目录，异常值阈值是所需要的最低传感器数值'
                              '\n所提供数据类型可以填vibrate或temperature',
                 font=('华文新魏', 16), relief=SUNKEN)
    msg1.place(relx=0.1, rely=0.05, relwidth=0.8, relheight=0.25)
    lb2 = Label(winNew, text='原始数据文件路径:', font=('华文新魏', 16))
    lb2.place(relx=0.1, rely=0.3)
    lb3 = Label(winNew, text='数据文件备份路径:', font=('华文新魏', 16))
    lb3.place(relx=0.1, rely=0.5)
    inp1 = Entry(winNew, font=('华文新魏', 20))
    inp1.place(relx=0.1, rely=0.35, relwidth=0.8, relheight=0.1)
    inp2 = Entry(winNew, font=('华文新魏', 20))
    inp2.place(relx=0.1, rely=0.55, relwidth=0.8, relheight=0.1)
    lb3 = Label(winNew, text='异常值阈值:', font=('华文新魏', 16))
    lb3.place(relx=0.1, rely=0.65)
    lb4 = Label(winNew, text='所提供数据类型:', font=('华文新魏', 16))
    lb4.place(relx=0.7, rely=0.65)
    inp3 = Entry(winNew, font=('华文新魏', 20))
    inp3.place(relx=0.1, rely=0.7, relwidth=0.2, relheight=0.1)
    inp4 = Entry(winNew, font=('华文新魏', 20))
    inp4.place(relx=0.7, rely=0.7, relwidth=0.2, relheight=0.1)
    var = StringVar()
    var.set('状态：尚未开始')
    lb5 = Label(winNew, textvariable=var, fg='red', font=('华文新魏', 18), relief=GROOVE)
    lb5.place(relx=0.35, rely=0.85)

    def start():
        temp_list = []
        listdir(inp1.get(),temp_list)
        if len(temp_list)==1:
            temp_data = pd.read_csv(temp_list[0],encoding='GBK')
            temp_data = minor_delete(temp_data,float(inp3.get()))
            database_name = temp_list[0]
            database_name = database_name.split('\\')[1].split('.')[-2]+'_new.csv'
            temp_data.to_csv(inp2.get()+database_name, index=False, encoding='GBK')
            temp_data.to_csv('../output/'+database_name, index=False, encoding='GBK')
            plt.boxplot(temp_data, labels=temp_data.columns)
            plt.show()
            detection_abn('../output/'+database_name)
            var.set('状态:已生成数据文件:\n' + database_name)
        else:
            database_name,merge_d = merge_data(inp1.get(),inp4.get())
            merge_d = minor_delete(merge_d,float(inp3.get()))
            merge_d.to_csv(database_name, index=True, encoding='GBK')
            merge_d.to_csv(inp2.get()+'beifen.csv', index=True, encoding='GBK')
            plt.boxplot(merge_d,labels = merge_d.columns)
            plt.show()
            detection_abn(database_name)
            var.set('状态:已生成数据文件:\n' + database_name)

    bt_start = Button(winNew, text='开始生成', command=start)
    bt_start.place(relx=0.1, rely=0.85, height=70, width=100)
    btClose = Button(winNew, text='关闭', command=winNew.destroy)
    btClose.place(relx=0.7, rely=0.85, height=70, width=100)


database_name_train = ''


def newwind1():
    winNew1 = Toplevel(root)
    winNew1.geometry('600x400')
    winNew1.title('温度预测模块')


    def subwin1():
        def xz():
            filename = tkinter.filedialog.askopenfilename()
            if filename != '':
                lb.config(text='您选择的训练集是' + filename)
            else:
                lb.config(text='您没有选择任何文件')
            global database_name_train
            database_name_train = filename
        subwin_1 = Toplevel(winNew1)
        subwin_1.geometry('900x640')
        subwin_1.title('训练LSTM模型')

        btn = Button(subwin_1, text='选择训练集', command=xz)
        btn.pack()
        lb = Label(subwin_1, text='', font=('华文新魏', 15))
        lb.pack()

        inp1 = Entry(subwin_1, font=('华文新魏', 20))
        inp1.place(relx=0.1, rely=0.18, relwidth=0.8, relheight=0.08)
        lb3 = Label(subwin_1, text='模型保存路径:', font=('华文新魏', 15))
        lb3.place(relx=0.1, rely=0.13)

        lb4 = Label(subwin_1, text='训练模型参数设定:', font=('华文新魏', 20), fg='red')
        lb4.place(relx=0.1, rely=0.28)
        # 参数设定N_HIDDEN1, N_LAYER1, N_EPOCH1, LR1, batch_size1, seqLen1, input_size1:
        ddl0 = ttk.Combobox(subwin_1, font=('华文新魏', 15))
        ddl0['value'] = ('160', '120', '80', '60')
        ddl0.current(0)
        ddl0.place(relx=0.25, rely=0.35, relwidth=0.1, relheight=0.08)
        lb5 = Label(subwin_1, text='Hidden_size:', font=('华文新魏', 15))
        lb5.place(relx=0.08, rely=0.37)

        ddl1 = ttk.Combobox(subwin_1, font=('华文新魏', 15))
        ddl1['value'] = ('1', '2', '3')
        ddl1.current(0)
        ddl1.place(relx=0.7, rely=0.35, relwidth=0.1, relheight=0.08)
        lb6 = Label(subwin_1, text='Num_layers:', font=('华文新魏', 15))
        lb6.place(relx=0.55, rely=0.37)

        ddl2 = ttk.Combobox(subwin_1, font=('华文新魏', 15))
        ddl2['value'] = ('30', '50', '70')
        ddl2.current(0)
        ddl2.place(relx=0.25, rely=0.5, relwidth=0.1, relheight=0.08)
        lb7 = Label(subwin_1, text='Num_Epoch:', font=('华文新魏', 15))
        lb7.place(relx=0.1, rely=0.52)

        ddl3 = ttk.Combobox(subwin_1, font=('华文新魏', 15))
        ddl3['value'] = ('0.001', '0.01', '0.1')
        ddl3.current(0)
        ddl3.place(relx=0.7, rely=0.5, relwidth=0.1, relheight=0.08)
        lb8 = Label(subwin_1, text='Learning_Rate:', font=('华文新魏', 15))
        lb8.place(relx=0.52, rely=0.52)

        ddl4 = ttk.Combobox(subwin_1, font=('华文新魏', 15))
        ddl4['value'] = ('64', '128')
        ddl4.current(0)
        ddl4.place(relx=0.25, rely=0.65, relwidth=0.1, relheight=0.08)
        lb9 = Label(subwin_1, text='Batch_Size:', font=('华文新魏', 15))
        lb9.place(relx=0.11, rely=0.67)

        ddl5 = ttk.Combobox(subwin_1, font=('华文新魏', 15))
        ddl5['value'] = ('30', '40','50','60','70')
        ddl5.current(0)
        ddl5.place(relx=0.7, rely=0.65, relwidth=0.1, relheight=0.08)
        lb10 = Label(subwin_1, text='Seq_Len:', font=('华文新魏', 15))
        lb10.place(relx=0.58, rely=0.67)

        ddl6 = ttk.Combobox(subwin_1, font=('华文新魏', 15))
        ddl6['value'] = ('1','5', '10', '15', '20')
        ddl6.current(0)
        ddl6.place(relx=0.25, rely=0.8, relwidth=0.1, relheight=0.08)
        lb11 = Label(subwin_1, text='pre_length:', font=('华文新魏', 15))
        lb11.place(relx=0.11, rely=0.82)

        ddl7 = Entry(subwin_1, font=('华文新魏', 15))
        ddl7.place(relx=0.7, rely=0.8, relwidth=0.1, relheight=0.08)
        lb12 = Label(subwin_1, text='标签列名:', font=('华文新魏', 15))
        lb12.place(relx=0.58, rely=0.82)



        def start_train():
            params = {}
            params['seqLen'] = int(ddl5.get())
            params['batch_size'] = int(ddl4.get())
            params['pre_length'] = int(ddl6.get())
            params['learning_rate'] = eval(ddl3.get())
            params['epochs'] = int(ddl2.get())
            params['hidden_size'] = int(ddl0.get())
            params['num_layers'] = int(ddl1.get())
            params['model_type'] = 'LSTM'
            train_con(database_name_train,params,inp1.get(),ddl7.get())

        btn_start = Button(subwin_1, text='开始训练', command=start_train)
        btn_start.place(relx=0.45, rely=0.9, height=40, width=50)



    def subwin3():
        def xz():
            filename = tkinter.filedialog.askopenfilename()
            if filename != '':
                lb.config(text='您选择的模型是' + filename)
            else:
                lb.config(text='您没有选择任何文件')
            global model_name_eval
            model_name_eval = filename
        def xz1():
            filename = tkinter.filedialog.askopenfilename()
            if filename != '':
                lb1.config(text='您选择的数据文件是' + filename)
            else:
                lb1.config(text='您没有选择任何文件')
            global file_name_eval
            file_name_eval = filename
        subwin_1 = Toplevel(winNew1)
        subwin_1.geometry('900x640')
        subwin_1.title('预测温度')

        btn = Button(subwin_1, text='选择预测模型', command=xz)
        btn.pack()
        lb = Label(subwin_1, text='', font=('华文新魏', 15))
        lb.pack()

        btn1 = Button(subwin_1, text='选择数据文件', command=xz1)
        btn1.pack()
        lb1 = Label(subwin_1, text='', font=('华文新魏', 15))
        lb1.pack()


        # inp1 = Entry(subwin_1, font=('华文新魏', 20))
        # inp1.place(relx=0.1, rely=0.28, relwidth=0.4, relheight=0.08)
        # lb4 = Label(subwin_1, text='预测列名:', font=('华文新魏', 20), fg='red')
        # lb4.place(relx=0.1, rely=0.15)

        txt = Text(subwin_1)
        txt.place(relx=0.1, rely=0.5, relwidth=0.8, relheight=0.7)

        def start_train():
            pre = pred(file_name_eval,model_name_eval)
            pre = pre.detach().numpy().flatten()
            txt.insert(END, '*' * 40 + '\n')
            txt.insert(END, '测试模型:' + model_name_eval + '\n' + '数据文件:' + file_name_eval + '\n')
            txt.insert(END, '预测温度是:')
            for i in range(len(pre)):
                txt.insert(END, '%.06f\t' % pre[i])
            txt.insert(END, '\n')
            #txt.insert(END, '预测温度是:'+pre+'\n' )
            txt.insert(END, '*' * 40 + '\n')


        btn_start = Button(subwin_1, text='开始预测', command=start_train)
        btn_start.place(relx=0.45, rely=0.9, height=40, width=50)


    btn_start0 = Button(winNew1, text='训练LSTM模型', command=subwin1)
    btn_start0.place(relx=0.1, rely=0.45, height=70, width=100)


    btn_start2 = Button(winNew1, text='LSTM预测温度', command=subwin3)
    btn_start2.place(relx=0.7, rely=0.45, height=70, width=100)

model_name_eval = ''
test_set_name = ''


def newwind2():
    def xz():
        filename = tkinter.filedialog.askopenfilename()
        if filename != '':
            lb.config(text='您选择的待诊断数据是' + filename)
        else:
            lb.config(text='您没有选择任何文件')
        global fault_data
        fault_data = filename

    def xz1():
        filename = tkinter.filedialog.askopenfilename()
        if filename != '':
            lb1.config(text='您选择的参考数据集是' + filename)
        else:
            lb1.config(text='您没有选择任何文件')
        global reference_data
        reference_data = filename



    def start_eval():
        one_n,mone_n = build_knn(5,reference_data,fault_data,0)
        if one_n>mone_n:
            s = '水泵运行正常'
        else:
            s = '水泵可能故障'
        txt.insert(END, '*' * 40 + '\n')
        txt.insert(END, '待诊断文件:' + fault_data + '\n' + '参考数据集:' + reference_data + '\n')
        txt.insert(END, '正常点数为:' + str(one_n) + '\n')
        txt.insert(END, '异常点数为:' + str(mone_n) + '\n')
        txt.insert(END, '预测结果为:' +s+'\n')
        txt.insert(END, '*' * 40 + '\n')

    winNew2 = Toplevel(root)
    winNew2.geometry('900x640')
    winNew2.title('故障诊断模块')

    btn = Button(winNew2, text='选择待诊断数据文件', command=xz, relief=GROOVE)
    btn.pack()
    lb = Label(winNew2, text='', font=('华文新魏', 15))
    lb.pack()
    btn1 = Button(winNew2, text='选择参考数据集', command=xz1, relief=GROOVE)
    btn1.pack()
    lb1 = Label(winNew2, text='', font=('华文新魏', 15))
    lb1.pack()



    btn2 = Button(winNew2, text='开始诊断', command=start_eval, relief=GROOVE)
    btn2.place(relx=0.45, rely=0.41, height=70, width=100)
    txt = Text(winNew2)
    txt.place(relx=0.1, rely=0.5, relwidth=0.8, relheight=0.7)




def newwind3():
    winNew3 = Toplevel(root)
    winNew3.geometry('900x640')
    winNew3.title('健康评估模块')

    def xz():
        filename = tkinter.filedialog.askopenfilename()
        if filename != '':
            lb.config(text='您选择的待诊断数据是' + filename)
        else:
            lb.config(text='您没有选择任何文件')
        global fault_data
        fault_data = filename

    def xz1():
        filename = tkinter.filedialog.askopenfilename()
        if filename != '':
            lb1.config(text='您选择的参考数据集是' + filename)
        else:
            lb1.config(text='您没有选择任何文件')
        global reference_data
        reference_data = filename

    def cal_score(temp,stan:list):
        if temp>stan[-1] or temp<stan[0]:
            score = -10
        elif stan[1]<temp and temp<stan[-2]:
            score = 10
        else:
            score =0
        return score




    def start_eval():
        one_n,mone_n = build_knn(5,reference_data,fault_data,0)
        score1 = cal_score(float(inp2.get()),[37,38,40,41])
        score2 = cal_score(float(inp3.get()), [37, 38, 40, 41])
        score3 = cal_score(float(inp4.get()), [37, 38, 40, 41])
        score4 = cal_score(float(inp5.get()), [37, 38, 40, 41])
        score_sum = score1+score2+score3+score4+one_n-mone_n
        txt.insert(END, '*' * 40 + '\n')
        txt.insert(END, '待诊断文件:' + fault_data + '\n' + '参考数据集:' + reference_data + '\n')
        txt.insert(END, '该水泵健康分数是是:')
        txt.insert(END, '%.06f\t' % score_sum)
        txt.insert(END, '\n')
        # txt.insert(END, '预测温度是:'+pre+'\n' )
        txt.insert(END, '*' * 40 + '\n')


    btn = Button(winNew3, text='选择待诊断数据文件', command=xz, relief=GROOVE)
    btn.pack()
    lb = Label(winNew3, text='', font=('华文新魏', 15))
    lb.pack()
    btn1 = Button(winNew3, text='选择参考数据集', command=xz1, relief=GROOVE)
    btn1.pack()
    lb1 = Label(winNew3, text='', font=('华文新魏', 15))
    lb1.pack()

    inp2 = Entry(winNew3, font=('华文新魏', 20))
    inp2.place(relx=0.05, rely=0.25, relwidth=0.18, relheight=0.1)
    lb3 = Label(winNew3, text='上导轴瓦温度:', font=('华文新魏', 14))
    lb3.place(relx=0.05, rely=0.17)
    inp3 = Entry(winNew3, font=('华文新魏', 20))
    inp3.place(relx=0.25, rely=0.25, relwidth=0.18, relheight=0.1)
    lb4 = Label(winNew3, text='下导轴瓦温度:', font=('华文新魏', 14))
    lb4.place(relx=0.25, rely=0.17)
    inp4 = Entry(winNew3, font=('华文新魏', 20))
    inp4.place(relx=0.45, rely=0.25, relwidth=0.18, relheight=0.1)
    lb5 = Label(winNew3, text='推力轴瓦温度:', font=('华文新魏', 14))
    lb5.place(relx=0.45, rely=0.17)
    inp5 = Entry(winNew3, font=('华文新魏', 20))
    inp5.place(relx=0.65, rely=0.25, relwidth=0.18, relheight=0.1)
    lb6 = Label(winNew3, text='排水总管温度:', font=('华文新魏', 14))
    lb6.place(relx=0.65, rely=0.17)



    btn2 = Button(winNew3, text='开始评估', command=start_eval, relief=GROOVE)
    btn2.place(relx=0.45, rely=0.41, height=70, width=100)
    txt = Text(winNew3)
    txt.place(relx=0.1, rely=0.5, relwidth=0.8, relheight=0.7)

    return



style = Style(theme='journal')
root = style.master
root.geometry('800x800')
root.title('水泵健康管理系统——1.0')

image2 =Image.open(r'..\图片\zssl.jpg')
background_image = ImageTk.PhotoImage(image2)

background_label = Label(root, image=background_image)
background_label.place(x=0, y=0, relwidth=1, relheight=0.6)
photo_user = Image.open(r'..\图片\button.gif')
photo_user1 = ImageTk.PhotoImage(photo_user)

btn1 = Button(root, text='数据处理', command=newwind, font=('华文新魏', 18), image=photo_user1, compound="center")
btn1.place(relx=0.15, rely=0.3, relwidth=0.3, relheight=0.1)

btn2 = Button(root, text='温度预测', command=newwind1, font=('华文新魏', 18), image=photo_user1, compound="center")
btn2.place(relx=0.55, rely=0.3, relwidth=0.3, relheight=0.1)

btn3 = Button(root, text='故障诊断', command=newwind2, font=('华文新魏', 18), image=photo_user1, compound="center")
btn3.place(relx=0.15, rely=0.6, relwidth=0.3, relheight=0.1)

btn4 = Button(root, text='健康评估', command=newwind3, font=('华文新魏', 18), image=photo_user1, compound="center")
btn4.place(relx=0.55, rely=0.6, relwidth=0.3, relheight=0.1)

root.mainloop()
