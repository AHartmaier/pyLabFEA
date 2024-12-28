import os
import numpy as np
import tkinter as tk
import tkinter.font as tkFont
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from tkinter import ttk, Toplevel, END
from pylabfea import Model, Material


def self_closing_message(message, duration=4000):
    """
    Display a self-closing message box.

    :param message: The message to be displayed.
    :param duration: The time in milliseconds before the message box closes automatically.
    :return: A reference to the popup window.
    """
    popup = Toplevel()
    popup.title("Information")
    popup.geometry("300x100")

    screen_width = popup.winfo_screenwidth()
    screen_height = popup.winfo_screenheight()
    window_width = 300
    window_height = 100
    x_coordinate = int((screen_width / 2) - (window_width / 2))
    y_coordinate = int((screen_height / 2) - (window_height / 2))
    popup.geometry(f"{window_width}x{window_height}+{x_coordinate}+{y_coordinate}")

    label = ttk.Label(popup, text=message, font=("Helvetica", 12), wraplength=250)
    label.pack(expand=True)
    popup.after(duration, popup.destroy)
    popup.update_idletasks()
    return


def add_label_and_entry(frame, row, label_text, entry_var, entry_type="entry", bold=True, options=None):
    label_font = ("Helvetica", 12, "bold") if bold else ("Helvetica", 12)
    ttk.Label(frame, text=label_text, font=label_font).grid(row=row, column=0, sticky='w')
    entry = None
    if entry_type == "entry":
        entry = ttk.Entry(frame, textvariable=entry_var, width=15, font=("Helvetica", 12))
        entry.grid(row=row, column=1, sticky='e')
    elif entry_type == "checkbox":
        ttk.Checkbutton(frame, variable=entry_var).grid(row=row, column=1, sticky='e')
    elif entry_type == "combobox" and options is not None:
        combobox = ttk.Combobox(frame, textvariable=entry_var, values=options, state='readonly',
                                width=14)
        combobox.grid(row=row, column=1, sticky='e')
        combobox.configure(font=("Helvetica", 12))
        combobox.current(0)
    return entry


class UserInterface(object):
    def __init__(self, app, notebook):
        self.app = app
        # geometry
        self.geom = tk.DoubleVar(value=18)
        # boundary conditions
        self.sides = tk.StringVar(value='force')  # free sides, change to 'disp' for fixed lateral sides
        self.eps_tot = tk.DoubleVar(value=0.01)  # total strain in y-direction
        # elastic material parameters
        self.E1 = tk.DoubleVar(value=10.e3)  # Young's modulus of matrix
        self.E2 = tk.DoubleVar(value=300.e3)  # Young's modulus of filler phase (fibers or particles)

        # plot frames
        tab1 = ttk.Frame(notebook)
        notebook.add(tab1, text="Composite")

        main_frame1 = ttk.Frame(tab1)
        main_frame1.grid(row=0, column=0, sticky='nsew', padx=20, pady=20)

        plot_frame1 = ttk.Frame(tab1)
        plot_frame1.grid(row=0, column=1, sticky='nsew', padx=20, pady=20)
        plot_frame1.rowconfigure(0, weight=1)
        plot_frame1.columnconfigure(0, weight=1)

        self.plot_frame = ttk.Frame(plot_frame1)
        self.plot_frame.grid(row=0, column=0, sticky='nsew')

        # define labels and entries
        line_seq = np.linspace(0, 50, dtype=int)
        line = iter(line_seq)
        ttk.Label(main_frame1, text="Geometry", font=("Helvetica", 16, "bold")) \
            .grid(row=next(line), column=0, columnspan=2, pady=(10, 0), sticky='w')
        add_label_and_entry(main_frame1, next(line), "Side length", self.geom, bold=False)
        add_label_and_entry(main_frame1, next(line), "Max. strain", self.eps_tot, bold=False)
        add_label_and_entry(main_frame1, next(line), "Lateral BC", self.sides, entry_type="combobox",
                            options=["force", "disp"], bold=False)
        add_label_and_entry(main_frame1, next(line), "Matrix Young's modulus", self.E1, bold=False)
        add_label_and_entry(main_frame1, next(line), "Filler Young's modulus", self.E2, bold=False)

        # create buttons
        button_frame1 = ttk.Frame(main_frame1)
        button_frame1.grid(row=next(line), column=0, columnspan=2, pady=10, sticky='ew')
        button_run = ttk.Button(button_frame1, text="Run", style='TButton',
                                 command=self.run)
        button_run.grid(row=1, column=1, padx=(10, 5), pady=5, sticky='ew')
        button_exit = ttk.Button(button_frame1, text="Exit", style='TButton', command=self.close)
        button_exit.grid(row=1, column=2, padx=(10, 5), pady=5, sticky='ew')

    def close(self):
        self.app.quit()
        self.app.destroy()

    def display_plot(self, fig):
        """ Show image on canvas. """
        self.app.update_idletasks()
        width, height = self.app.winfo_reqwidth(), self.app.winfo_reqheight()
        self.app.geometry(f"{width}x{height}")

        if self.canvas1 is not None:
            self.canvas1.get_tk_widget().destroy()
        self.canvas1 = FigureCanvasTkAgg(fig, master=self.plot_frame)
        self.canvas1.draw()
        self.canvas1.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        self.app.update_idletasks()
        width, height = self.app.winfo_reqwidth(), self.app.winfo_reqheight()
        self.app.geometry(f"{width}x{height}")

    def run(self):
        # setup material definition for regular mesh
        NX = NY = int(self.geom.get())
        sides = self.sides.get()
        E1 = self.E1.get()
        E2 = self.E2.get()
        NXi1 = int(NX / 3)
        NXi2 = 2 * NXi1
        NYi1 = int(NY / 3)
        NYi2 = 2 * NYi1
        el = np.ones((NX, NY), dtype=np.int)
        el[NXi1:NXi2, NYi1:NYi2] = 2

        # define materials
        mat1 = Material(num=1)  # call class to generate material object
        mat1.elasticity(E=E1, nu=0.27)  # define elastic properties
        mat1.plasticity(sy=150., khard=500., sdim=6)  # isotropic plasticity
        mat2 = Material(num=2)  # define second material
        mat2.elasticity(E=E2, nu=0.3)  # material is purely elastic

        # setup model for elongation in y-direction
        fe = Model(dim=2, planestress=False)  # initialize finite element model
        fe.geom(sect=2, LX=4., LY=4.)  # define geometry with two sections
        fe.assign([mat1, mat2])  # assign materials to sections

        # boundary conditions: uniaxial stress in longitudinal direction
        fe.bcbot(0.)  # fix bottom boundary
        fe.bcright(0., sides)  # boundary condition on lateral edges of model
        fe.bcleft(0., sides)
        fe.bctop(self.eps_tot.get() * fe.leny, 'disp')  # strain applied to top nodes

        # meshing and plotting of model
        fe.mesh(elmts=el, NX=NX, NY=NY)  # create regular mesh with sections as defined in el
        if sides == 'force':
            # fix lateral displacements of corner node to prevent rigid body motion
            hh = [no in fe.nobot for no in fe.noleft]
            noc = np.nonzero(hh)[0]  # find corner node
            fe.bcnode(noc, 0., 'disp', 'x')  # fix lateral displacement
        fe.plot('mat', mag=1, shownodes=False)

        # find solution and plot stress and strain fields
        fe.solve()  # calculate mechanical equilibrium under boundary conditions
        fe.plot('stress1', mag=4, shownodes=False)
        fe.plot('stress2', mag=4, shownodes=False)
        fe.plot('seq', mag=4, shownodes=False)
        fe.plot('peeq', mag=4, shownodes=False)


""" Main code section """
app = tk.Tk()
app.title("pyLabFEA")
screen_width = app.winfo_screenwidth()
screen_height = app.winfo_screenheight()
# plt.rcParams['figure.dpi'] = screen_height / 19  # height stats_plot: 9, height voxel_plot: 6, margin: 4
window_width = int(screen_width * 0.9)
window_height = int(screen_height * 0.8)
x_coordinate = int((screen_width / 2) - (window_width / 2))
y_coordinate = 0  # int((screen_height / 2) - (window_height / 2))
app.geometry(f"{window_width}x{window_height}+{x_coordinate}+{y_coordinate}")

notebook = ttk.Notebook(app)
notebook.pack(fill='both', expand=True)
style = ttk.Style(app)
default_font = tkFont.Font(family="Helvetica", size=12, weight="bold")
style.configure('TNotebook.Tab', font=('Helvetica', '12', "bold"))
style.configure('TButton', font=default_font)

""" Start main loop """
gui = UserInterface(app, notebook)
app.mainloop()
