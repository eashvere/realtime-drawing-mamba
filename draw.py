# From Matiiss: https://stackoverflow.com/questions/70403360/drawing-with-mouse-on-tkinter-canvas

import tkinter as tk
import rdppy
import numpy as np
from customModel import customModel
# from torch import nn
import torch

# test_drawing = [[  16,   -5,    1,    0,    0], [  19,  -12,    1,    0,    0], [  80,  -56,    1,    0,    0], [  22,  -33,    1,    0,    0], [  11,  -29,    1,    0,    0], [  17,  -66,    1,    0,    0], [   5,  -38,    1,    0,    0], [   2, -394,    1,    0,    0], [   8,  -37,    1,    0,    0], [  13,  -26,    1,    0,    0], [  11,  -32,    1,    0,    0], [  57,  -99,    1,    0,    0], [   2,   10,    1,    0,    0], [  -4,   33,    1,    0,    0], [  -9,   40,    1,    0,    0], [ -10,   84,    1,    0,    0], [   0,  125,    1,    0,    0], [  12,   53,    1,    0,    0], [  24,   80,    1,    0,    0], [  50,  114,    1,    0,    0], [  27,   75,    1,    0,    0], [  12,   25,    1,    0,    0], [  69,   51,    1,    0,    0], [  27,   35,    1,    0,    0], [  30,   58,    1,    0,    0], [  17,   22,    1,    0,    0], [  14,   15,    1,    0,    0], [  31,   20,    0,    1,    0], [   0,    0,    0,    0,    1], [   0,    0,    0,    0,    1], [   0,    0,    0,    0,    1], [   0,    0,    0,    0,    1], [   0,    0,    0,    0,    1], [   0,    0,    0,    0,    1], [   0,    0,    0,    0,    1], [   0,    0,    0,    0,    1], [   0,    0,    0,    0,    1], [   0,    0,    0,    0,    1], [   0,    0,    0,    0,    1], [   0,    0,    0,    0,    1], [   0,    0,    0,    0,    1], [   0,    0,    0,    0,    1], [   0,    0,    0,    0,    1], [   0,    0,    0,    0,    1], [   0,    0,    0,    0,    1], [   0,    0,    0,    0,    1], [   0,    0,    0,    0,    1], [   0,    0,    0,    0,    1], [   0,    0,    0,    0,    1], [   0,    0,    0,    0,    1], [   0,    0,    0,    0,    1], [   0,    0,    0,    0,    1], [   0,    0,    0,    0,    1], [   0,    0,    0,    0,    1], [   0,    0,    0,    0,    1], [   0,    0,    0,    0,    1], [   0,    0,    0,    0,    1], [   0,    0,    0,    0,    1], [   0,    0,    0,    0,    1], [   0,    0,    0,    0,    1], [   0,    0,    0,    0,    1], [   0,    0,    0,    0,    1], [   0,    0,    0,    0,    1], [   0,    0,    0,    0,    1], [   0,    0,    0,    0,    1], [   0,    0,    0,    0,    1], [   0,    0,    0,    0,    1], [   0,    0,    0,    0,    1], [   0,    0,    0,    0,    1], [   0,    0,    0,    0,    1], [   0,    0,    0,    0,    1], [   0,    0,    0,    0,    1], [   0,    0,    0,    0,    1], [   0,    0,    0,    0,    1], [   0,    0,    0,    0,    1], [   0,    0,    0,    0,    1], [   0,    0,    0,    0,    1], [   0,    0,    0,    0,    1], [   0,    0,    0,    0,    1], [   0,    0,    0,    0,    1], [   0,    0,    0,    0,    1], [   0,    0,    0,    0,    1], [   0,    0,    0,    0,    1], [   0,    0,    0,    0,    1], [   0,    0,    0,    0,    1], [   0,    0,    0,    0,    1], [   0,    0,    0,    0,    1], [   0,    0,    0,    0,    1], [   0,    0,    0,    0,    1], [   0,    0,    0,    0,    1], [   0,    0,    0,    0,    1], [   0,    0,    0,    0,    1], [   0,    0,    0,    0,    1], [   0,    0,    0,    0,    1], [   0,    0,    0,    0,    1], [   0,    0,    0,    0,    1], [   0,    0,    0,    0,    1], [   0,    0,    0,    0,    1], [   0,    0,    0,    0,    1]]), tensor([[  19,  -12,    1,    0,    0], [  80,  -56,    1,    0,    0], [  22,  -33,    1,    0,    0], [  11,  -29,    1,    0,    0], [  17,  -66,    1,    0,    0], [   5,  -38,    1,    0,    0], [   2, -394,    1,    0,    0], [   8,  -37,    1,    0,    0], [  13,  -26,    1,    0,    0], [  11,  -32,    1,    0,    0], [  57,  -99,    1,    0,    0], [   2,   10,    1,    0,    0], [  -4,   33,    1,    0,    0], [  -9,   40,    1,    0,    0], [ -10,   84,    1,    0,    0], [   0,  125,    1,    0,    0], [  12,   53,    1,    0,    0], [  24,   80,    1,    0,    0], [  50,  114,    1,    0,    0], [  27,   75,    1,    0,    0], [  12,   25,    1,    0,    0], [  69,   51,    1,    0,    0], [  27,   35,    1,    0,    0], [  30,   58,    1,    0,    0], [  17,   22,    1,    0,    0], [  14,   15,    1,    0,    0], [  31,   20,    0,    1,    0], [   0,    0,    0,    0,    1], [   0,    0,    0,    0,    1], [   0,    0,    0,    0,    1], [   0,    0,    0,    0,    1], [   0,    0,    0,    0,    1], [   0,    0,    0,    0,    1], [   0,    0,    0,    0,    1], [   0,    0,    0,    0,    1], [   0,    0,    0,    0,    1], [   0,    0,    0,    0,    1], [   0,    0,    0,    0,    1], [   0,    0,    0,    0,    1], [   0,    0,    0,    0,    1], [   0,    0,    0,    0,    1], [   0,    0,    0,    0,    1], [   0,    0,    0,    0,    1], [   0,    0,    0,    0,    1], [   0,    0,    0,    0,    1], [   0,    0,    0,    0,    1], [   0,    0,    0,    0,    1], [   0,    0,    0,    0,    1], [   0,    0,    0,    0,    1], [   0,    0,    0,    0,    1], [   0,    0,    0,    0,    1], [   0,    0,    0,    0,    1], [   0,    0,    0,    0,    1], [   0,    0,    0,    0,    1], [   0,    0,    0,    0,    1], [   0,    0,    0,    0,    1], [   0,    0,    0,    0,    1], [   0,    0,    0,    0,    1], [   0,    0,    0,    0,    1], [   0,    0,    0,    0,    1], [   0,    0,    0,    0,    1], [   0,    0,    0,    0,    1], [   0,    0,    0,    0,    1], [   0,    0,    0,    0,    1], [   0,    0,    0,    0,    1], [   0,    0,    0,    0,    1], [   0,    0,    0,    0,    1], [   0,    0,    0,    0,    1], [   0,    0,    0,    0,    1], [   0,    0,    0,    0,    1], [   0,    0,    0,    0,    1], [   0,    0,    0,    0,    1], [   0,    0,    0,    0,    1], [   0,    0,    0,    0,    1], [   0,    0,    0,    0,    1], [   0,    0,    0,    0,    1], [   0,    0,    0,    0,    1], [   0,    0,    0,    0,    1], [   0,    0,    0,    0,    1], [   0,    0,    0,    0,    1], [   0,    0,    0,    0,    1], [   0,    0,    0,    0,    1], [   0,    0,    0,    0,    1], [   0,    0,    0,    0,    1], [   0,    0,    0,    0,    1], [   0,    0,    0,    0,    1], [   0,    0,    0,    0,    1], [   0,    0,    0,    0,    1], [   0,    0,    0,    0,    1], [   0,    0,    0,    0,    1], [   0,    0,    0,    0,    1], [   0,    0,    0,    0,    1], [   0,    0,    0,    0,    1], [   0,    0,    0,    0,    1], [   0,    0,    0,    0,    1], [   0,    0,    0,    0,    1], [   0,    0,    0,    0,    1], [   0,    0,    0,    0,    1], [   0,    0,    0,    0,    1]]


class Canvas:
    def __init__(self, canvas, isPredictions=False, completionCanvas=None):
        self.canvas = canvas
        self.canvas.pack()
        if not isPredictions:
            self.canvas.bind('<Button-1>', self.set_start)
            self.canvas.bind('<B1-Motion>', self.draw_line)
            self.canvas.bind('<ButtonRelease-1>', self.end_line)
            self.completionCanvas = completionCanvas
            self.canvas.bind('<KeyPress>', self.reset)
        else:
            # self.canvas.bind('<Button-1>', self.generate_drawing)
            #self.canvas.bind('<KeyPress>', test)
            self.canvas.bind('<Button-1>', self.focus)
            self.canvas.bind('<KeyPress>', self.reset)
            # self.drawingCanvas = drawingCanvas
            # self.canvas.bind('<Button-2>', self.gener)

        self.lines = []
        self.simplified_lines = []
        self.line_id = None
        self.curr_line = []
        self.line_options = {}
        self.stroke5 = []
        self.prevx = 0  
        self.prevy = 0
        self.busy = False

    def focus(self, event):
        self.canvas.focus_set()
    
    def reset_busy(self):
        self.busy = False
    
    def generate_drawing(self, drawing):
        start = torch.tensor(drawing, dtype=torch.float).unsqueeze(0).to('cuda')
        generated = loaded_model.generate(start)
        gen_list = generated.squeeze(0).cpu().numpy().astype('int').tolist()

        self.full_drawing(gen_list)
        # self.full_drawing(drawing)

    def reset(self, event):
        self.lines = []
        self.simplified_lines = []
        self.line_id = None
        self.curr_line = []
        self.stroke5 = []
        self.prevx = 0  
        self.prevy = 0
        self.canvas.delete('all')

    def get_drawing_size(self, drawing):
        minx, miny = 0, 0
        maxx, maxy = 0, 0
        for stroke in drawing:
            minx = min(minx, stroke[0])
            miny = min(miny, stroke[1])
            maxx = max(maxx, stroke[0])
            maxy = max(maxy, stroke[1])
        return minx, miny, maxx, maxy, maxx - minx, maxy - miny

    def full_drawing(self, test_drawing):
        self.reset(None)
        # test_drawing = [[  16,   -5,    1,    0,    0], [  19,  -12,    1,    0,    0], [  80,  -56,    1,    0,    0], [  22,  -33,    1,    0,    0], [  11,  -29,    1,    0,    0], [  17,  -66,    1,    0,    0], [   5,  -38,    1,    0,    0], [   2, -394,    1,    0,    0], [   8,  -37,    1,    0,    0], [  13,  -26,    1,    0,    0], [  11,  -32,    1,    0,    0], [  57,  -99,    1,    0,    0], [   2,   10,    1,    0,    0], [  -4,   33,    1,    0,    0], [  -9,   40,    1,    0,    0], [ -10,   84,    1,    0,    0], [   0,  125,    1,    0,    0], [  12,   53,    1,    0,    0], [  24,   80,    1,    0,    0], [  50,  114,    1,    0,    0], [  27,   75,    1,    0,    0], [  12,   25,    1,    0,    0], [  69,   51,    1,    0,    0], [  27,   35,    1,    0,    0], [  30,   58,    1,    0,    0], [  17,   22,    1,    0,    0], [  14,   15,    1,    0,    0], [  31,   20,    0,    1,    0], [   0,    0,    0,    0,    1], [   0,    0,    0,    0,    1], [   0,    0,    0,    0,    1], [   0,    0,    0,    0,    1], [   0,    0,    0,    0,    1], [   0,    0,    0,    0,    1], [   0,    0,    0,    0,    1], [   0,    0,    0,    0,    1], [   0,    0,    0,    0,    1], [   0,    0,    0,    0,    1], [   0,    0,    0,    0,    1], [   0,    0,    0,    0,    1], [   0,    0,    0,    0,    1], [   0,    0,    0,    0,    1], [   0,    0,    0,    0,    1], [   0,    0,    0,    0,    1], [   0,    0,    0,    0,    1], [   0,    0,    0,    0,    1], [   0,    0,    0,    0,    1], [   0,    0,    0,    0,    1], [   0,    0,    0,    0,    1], [   0,    0,    0,    0,    1], [   0,    0,    0,    0,    1], [   0,    0,    0,    0,    1], [   0,    0,    0,    0,    1], [   0,    0,    0,    0,    1], [   0,    0,    0,    0,    1], [   0,    0,    0,    0,    1], [   0,    0,    0,    0,    1], [   0,    0,    0,    0,    1], [   0,    0,    0,    0,    1], [   0,    0,    0,    0,    1], [   0,    0,    0,    0,    1], [   0,    0,    0,    0,    1], [   0,    0,    0,    0,    1], [   0,    0,    0,    0,    1], [   0,    0,    0,    0,    1], [   0,    0,    0,    0,    1], [   0,    0,    0,    0,    1], [   0,    0,    0,    0,    1], [   0,    0,    0,    0,    1], [   0,    0,    0,    0,    1], [   0,    0,    0,    0,    1], [   0,    0,    0,    0,    1], [   0,    0,    0,    0,    1], [   0,    0,    0,    0,    1], [   0,    0,    0,    0,    1], [   0,    0,    0,    0,    1], [   0,    0,    0,    0,    1], [   0,    0,    0,    0,    1], [   0,    0,    0,    0,    1], [   0,    0,    0,    0,    1], [   0,    0,    0,    0,    1], [   0,    0,    0,    0,    1], [   0,    0,    0,    0,    1], [   0,    0,    0,    0,    1], [   0,    0,    0,    0,    1], [   0,    0,    0,    0,    1], [   0,    0,    0,    0,    1], [   0,    0,    0,    0,    1], [   0,    0,    0,    0,    1], [   0,    0,    0,    0,    1], [   0,    0,    0,    0,    1], [   0,    0,    0,    0,    1], [   0,    0,    0,    0,    1], [   0,    0,    0,    0,    1], [   0,    0,    0,    0,    1], [   0,    0,    0,    0,    1], [   0,    0,    0,    0,    1], [   0,    0,    0,    0,    1], [   0,    0,    0,    0,    1]]
        # test_drawing = self.stroke5.copy()
        # print(test_drawing, self.stroke5)
        # print(self.get_drawing_size(test_drawing))
        currx, curry = 256, 256
        for i,stroke in enumerate(test_drawing):
            # print(stroke)
            if stroke[4] == 1:
                break
            nextx, nexty = currx + test_drawing[i][0], curry + test_drawing[i][1]
            if stroke[2] == 1:
                self.canvas.create_line(currx, curry, nextx, nexty, fill='black', width=2)
            currx, curry = nextx, nexty


    def draw_line(self,event):
        if not self.busy:
            self.busy = True
        
            self.curr_line.append((event.x, event.y))
            if self.line_id is not None:
                self.canvas.delete(self.line_id)
            self.line_id = self.canvas.create_line(self.curr_line, **self.line_options)
            
            temp = self.simplified_lines
            temp.append(self.simplifyLine(self.curr_line))
            
            temp_stroke5 = self.stroke5
            temp_stroke5 += self.convert5stroke(self.simplified_lines[-1])
            self.completionCanvas.generate_drawing(temp_stroke5)
            self.canvas.after(100, self.reset_busy)


    def set_start(self, event):
        self.focus("")
        self.curr_line.append((event.x, event.y))


    def end_line(self, event=None):
        self.lines.append(self.curr_line)
        # print(lines)
        self.simplified_lines.append(self.simplifyLine(self.curr_line))

        # prevx, prevy = 0,0
        # if len(self.simplified_lines) > 1:
        #     prevx, prevy = self.simplified_lines[-2][-1][0], self.simplified_lines[-2][-1][1]
        converted = self.convert5stroke(self.simplified_lines[-1])
        self.stroke5 += converted
        print(self.stroke5, len(self.stroke5))

        self.curr_line.clear()
        self.line_id = None
        #self.completionCanvas.generate_drawing(self.stroke5)
        # print(curr_line)

    def simplifyLine(self, line):
        # print(line)
        # npline = np.array(line)
        mask = rdppy.filter(line, 4)
        output = []
        for i in range(len(mask)):
            if mask[i]:
                output.append(line[i])
        print('line', line, 'simplified', output)
        return output

    def convert5stroke(self, line):
        results = []
        if self.prevx - line[0][0] != 0 or self.prevy - line[0][1] != 0:
            stroke = [0] * 5
            stroke[0] = line[0][0] - self.prevx
            stroke[1] = line[0][1] - self.prevy
            stroke[2] = 0
            stroke[3] = 1
            stroke[4] = 0
            self.prevx = line[0][0]
            self.prevy = line[0][1]
            results.append(stroke)
        for i in range(1, len(line)):
            stroke = [0] * 5
            stroke[0] = line[i][0] - self.prevx
            stroke[1] = line[i][1] - self.prevy
            stroke[2] = 1
            stroke[3] = 0
            stroke[4] = 0
            self.prevx = line[i][0]
            self.prevy = line[i][1]
            results.append(stroke)
        return results


root = tk.Tk()

loaded_model = customModel(nb=4, no=2, ns=2, embed_dim=256).to("cuda")
loaded_model.load('./saved/base.pth')

completionCanvas = Canvas(tk.Canvas(background='white', width=1024, height=1024), True)
drawingCanvas = Canvas(tk.Canvas(background='white', width=256, height=256), False, completionCanvas)

root.mainloop()
