# From Matiiss: https://stackoverflow.com/questions/70403360/drawing-with-mouse-on-tkinter-canvas

import tkinter as tk
import rdppy
import numpy as np

lines = []
simplified_lines = []
line_id = None
curr_line = []
line_options = {}
stroke5 = []


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
    global stroke5
    lines.append(curr_line)
    # print(lines)
    simplified_lines.append(simplifyLine(curr_line))

    prevx, prevy = 0,0
    if len(simplified_lines) > 1:
        prevx, prevy = simplified_lines[-2][-1][0], simplified_lines[-2][-1][1]
    converted = convert5stroke(prevx, prevy, simplified_lines[-1])
    stroke5 += converted
    print(stroke5, len(stroke5))

    curr_line.clear()
    line_id = None
    # print(curr_line)

def simplifyLine(line):
    # print(line)
    npline = np.array(line)
    mask = rdppy.filter(line, 2)
    output = []
    for i in range(len(mask)):
        if mask[i]:
            output.append(line[i])
    # print('line', line, 'simplified', output)
    return output

def convert5stroke(prevx, prevy, line):
    results = []
    if prevx - line[0][0] != 0 or prevy - line[0][1] != 0:
        stroke = [0] * 5
        stroke[0] = prevx - line[0][0]
        stroke[1] = prevy - line[0][1]
        stroke[2] = 0
        stroke[3] = 1
        stroke[4] = 0
        prevx = line[0][0]
        prevy = line[0][1]
        results.append(stroke)
    for i in range(1, len(line)):
        stroke = [0] * 5
        stroke[0] = prevx - line[i][0]
        stroke[1] = prevy - line[i][1]
        stroke[2] = 1
        stroke[3] = 0
        stroke[4] = 0
        prevx = line[i][0]
        prevy = line[i][1]
        results.append(stroke)
    return results


root = tk.Tk()

canvas = tk.Canvas(background='white', width=256, height=256)
canvas.pack()

canvas.bind('<Button-1>', set_start)
canvas.bind('<B1-Motion>', draw_line)
canvas.bind('<ButtonRelease-1>', end_line)

root.mainloop()
