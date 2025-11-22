import tkinter as tk
from tkinter import ttk
from constants import *
from PIL import Image, ImageDraw, ImageTk


class MapEditor:
    def __init__(self, parent, map_image=None, on_save_callback=None):
        self.top = tk.Toplevel(parent)
        self.top.title("Map Editor")
        self.top.geometry(f"{MAP_WIDTH + 50}x{MAP_HEIGHT + 100}")

        self.on_save_callback = on_save_callback

        # Map Data
        if map_image:
            self.image = map_image.copy()
        else:
            self.image = Image.new("RGB", (MAP_WIDTH, MAP_HEIGHT), "white")

        self.draw = ImageDraw.Draw(self.image)
        self.brush_size = 10
        self.color = "black"  # Default to drawing walls

        self.create_ui()

    def create_ui(self):
        # Toolbar
        toolbar = tk.Frame(self.top, height=50)
        toolbar.pack(side=tk.TOP, fill=tk.X)

        # Brush Button
        btn_brush = tk.Button(
            toolbar, text="Brush (Wall)", command=self.use_brush)
        btn_brush.pack(side=tk.LEFT, padx=5, pady=5)

        # Eraser Button
        btn_eraser = tk.Button(
            toolbar, text="Eraser (Free)", command=self.use_eraser)
        btn_eraser.pack(side=tk.LEFT, padx=5, pady=5)

        # Clear Map Button
        btn_clear = tk.Button(
            toolbar, text="Clear Map", command=self.clear_map)
        btn_clear.pack(side=tk.LEFT, padx=5, pady=5)

        # Save Button
        btn_save = tk.Button(
            toolbar, text="Save & Close", command=self.save_map)
        btn_save.pack(side=tk.RIGHT, padx=5, pady=5)

        # Canvas
        self.canvas_frame = tk.Frame(self.top)
        self.canvas_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.canvas = tk.Canvas(
            self.canvas_frame, width=MAP_WIDTH, height=MAP_HEIGHT, bg="white",
            highlightthickness=0)
        self.canvas.pack()

        self.update_canvas()

        # Bindings
        self.canvas.bind("<B1-Motion>", self.paint)
        self.canvas.bind("<Button-1>", self.paint)

    def use_brush(self):
        self.color = "black"

    def use_eraser(self):
        self.color = "white"

    def paint(self, event):
        x1, y1 = (event.x - self.brush_size), (event.y - self.brush_size)
        x2, y2 = (event.x + self.brush_size), (event.y + self.brush_size)

        # Draw on PIL Image
        self.draw.ellipse(
            [x1, y1, x2, y2],
            fill=self.color, outline=self.color)

        # Draw on Canvas
        self.canvas.create_oval(
            x1, y1, x2, y2, fill=self.color, outline=self.color)

    def update_canvas(self):
        self.tk_image = ImageTk.PhotoImage(self.image)
        self.canvas.create_image(0, 0, image=self.tk_image, anchor="nw")

    def clear_map(self):
        """Clear the entire map to white"""
        self.image = Image.new("RGB", (MAP_WIDTH, MAP_HEIGHT), "white")
        self.draw = ImageDraw.Draw(self.image)
        self.canvas.delete("all")
        self.update_canvas()

    def save_map(self):
        if self.on_save_callback:
            self.on_save_callback(self.image)
        self.top.destroy()
