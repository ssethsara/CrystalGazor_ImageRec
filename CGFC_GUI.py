from tkinter import *
#import CGFC as cgfc


class Window(Frame):


    def __init__(self, master=None):
        Frame.__init__(self, master)                 
        self.master = master
        self.init_window()

    #Creation of init_window
    def init_window(self):
        
        # changing the title of our master widget      
        self.master.title("GUI")

        # allowing the widget to take the full space of the root window
        self.pack(fill=BOTH, expand=1)

        # creating a menu instance
        menu = Menu(self.master)
        self.master.config(menu=menu)

        # create the file object)
        file = Menu(menu)

        # adds a command to the menu option, calling it exit, and the
        # command it runs on event is client_exit
        file.add_command(label="Exit", command=self.client_exit)

        #added "file" to our menu
        menu.add_cascade(label="File", menu=file)
        # create the file object)
        edit = Menu(menu)
   
        #added "file" to our menu
        menu.add_cascade(label="Edit", menu=edit)

        StartAnalysing = Button(self, text="StartAnalysing",command=self.StartAnalysing)

        # placing the button on my window
        StartAnalysing.place(x=300, y=10)

        self.var = StringVar()
        ProcessDisplay = Text(self, height=20, width=60)
        scroll = Scrollbar(root, command=ProcessDisplay.yview)
        ProcessDisplay.configure(yscrollcommand=scroll.set)
        ProcessDisplay.place(x=70, y=70)
        ProcessDisplay.pack(CENTER)
        scroll.pack(side=RIGHT, fill=Y)
        #label.pack()
        self.var.set("Ready")
   
        

    def showText(self):
        text = Label(self, text="Hey there good lookin!")
        text.pack()

    def StartAnalysing(self):
        self.var.set("Processing..")

    def client_exit(self):
        exit()


root = Tk()


#size of the window
root.geometry("640x480")


app = Window(root)
root.mainloop()  


