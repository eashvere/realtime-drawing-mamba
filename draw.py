# From Matiiss: https://stackoverflow.com/questions/70403360/drawing-with-mouse-on-tkinter-canvas

import tkinter as tk

lines = []
line_id = None
curr_line = []
line_options = {}


def draw_line(event):
    global line_id
    curr_line.append((event.x, event.y))
    if line_id is not None:
        canvas.delete(line_id)
    line_id = canvas.create_line(curr_line, **line_options)


def set_start(event):
    curr_line.append((event.x, event.y))


def end_line(event=None):
    global line_id
    lines.append(curr_line)
    print(lines)
    curr_line.clear()
    line_id = None


root = tk.Tk()

canvas = tk.Canvas(background='white', width=500, height=500)
canvas.pack()

canvas.bind('<Button-1>', set_start)
canvas.bind('<B1-Motion>', draw_line)
canvas.bind('<ButtonRelease-1>', end_line)

root.mainloop()
