import tkinter as tk
from tkinter import ttk, colorchooser
from .constants import *
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
        self.color_rgb = (0, 0, 0)  # RGB tuple for PIL
        self.tk_image = None  # Will be set when canvas is updated

        self.create_ui()

    def create_ui(self):
        # Toolbar
        toolbar = ttk.Frame(self.top)
        toolbar.pack(side=tk.TOP, fill=tk.X, padx=10, pady=10)

        # Brush Button
        btn_brush = ttk.Button(
            toolbar, text="Brush (Wall)", command=self.use_brush)
        btn_brush.pack(side=tk.LEFT, padx=5)

        # Eraser Button
        btn_eraser = ttk.Button(
            toolbar, text="Eraser (Free)", command=self.use_eraser)
        btn_eraser.pack(side=tk.LEFT, padx=5)

        # Color Picker Button (show current color using a styled ttk.Label)
        self.style = ttk.Style()
        # Configure a named style for the color display; will be updated on color change
        self.style.configure('Color.TLabel', background=self.color)
        self.color_display = ttk.Label(
            toolbar, text="  ", width=3, style='Color.TLabel')
        self.color_display.pack(side=tk.LEFT, padx=5)

        btn_color = ttk.Button(
            toolbar, text="Pick Color", command=self.pick_color)
        btn_color.pack(side=tk.LEFT, padx=5)

        # Clear Map Button
        btn_clear = ttk.Button(
            toolbar, text="Clear Map", command=self.clear_map)
        btn_clear.pack(side=tk.LEFT, padx=5)

        # Save Button
        btn_save = ttk.Button(
            toolbar, text="Save & Close", command=self.save_map)
        btn_save.pack(side=tk.RIGHT, padx=5)

        # Canvas Container with proper responsive layout
        self.canvas_frame = ttk.Frame(self.top)
        self.canvas_frame.pack(fill=tk.BOTH, expand=True,
                               padx=10, pady=(0, 10))

        # Configure grid for responsive layout
        self.canvas_frame.grid_rowconfigure(0, weight=1)
        self.canvas_frame.grid_columnconfigure(0, weight=1)

        self.canvas = tk.Canvas(
            self.canvas_frame, bg="white",
            highlightthickness=1, highlightbackground="#cccccc")
        self.canvas.grid(row=0, column=0, sticky="nsew")

        # Scaling parameters
        self.scale_factor = 1.0
        self.offset_x = 0
        self.offset_y = 0

        self.update_canvas()

        # Bindings
        self.canvas.bind("<B1-Motion>", self.paint)
        self.canvas.bind("<Button-1>", self.paint)
        self.canvas.bind("<Configure>", self.on_canvas_resize)

    def use_brush(self):
        self.color = "black"
        self.color_rgb = (0, 0, 0)
        # Update style background to reflect color
        self.style.configure('Color.TLabel', background=self.color)

    def use_eraser(self):
        self.color = "white"
        self.color_rgb = (255, 255, 255)
        # Update style background to reflect color
        self.style.configure('Color.TLabel', background=self.color)

    def pick_color(self):
        """Open color picker dialog and set the selected color"""
        color = colorchooser.askcolor(
            title="Choose wall color",
            initialcolor=self.color
        )
        if color[1]:  # color[1] is the hex color string
            self.color = color[1]
            self.color_rgb = color[0]  # color[0] is the RGB tuple
            # Convert float RGB values to integers if needed
            self.color_rgb = tuple(int(c) for c in self.color_rgb)
            # Update style background to reflect selected color
            self.style.configure('Color.TLabel', background=self.color)

    def paint(self, event):
        # Convert canvas coordinates to image coordinates
        img_x = (event.x - self.offset_x) / self.scale_factor
        img_y = (event.y - self.offset_y) / self.scale_factor

        # Check if within image bounds
        if img_x < 0 or img_x >= MAP_WIDTH or img_y < 0 or img_y >= MAP_HEIGHT:
            return

        x1, y1 = (img_x - self.brush_size), (img_y - self.brush_size)
        x2, y2 = (img_x + self.brush_size), (img_y + self.brush_size)

        # Draw on PIL Image (use RGB tuple)
        self.draw.ellipse(
            [x1, y1, x2, y2],
            fill=self.color_rgb, outline=self.color_rgb)

        # Draw on Canvas (use hex color string) - scaled coordinates
        canvas_x1 = x1 * self.scale_factor + self.offset_x
        canvas_y1 = y1 * self.scale_factor + self.offset_y
        canvas_x2 = x2 * self.scale_factor + self.offset_x
        canvas_y2 = y2 * self.scale_factor + self.offset_y

        self.canvas.create_oval(
            canvas_x1, canvas_y1, canvas_x2, canvas_y2,
            fill=self.color, outline=self.color)

    def on_canvas_resize(self, event):
        """Handle canvas resize and update image display"""
        self.update_canvas()

    def update_canvas(self):
        """Update canvas with scaled image to fit width while maintaining aspect ratio"""
        # Get canvas dimensions
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()

        # If canvas not yet sized, skip
        if canvas_width <= 1 or canvas_height <= 1:
            return

        # Calculate scale factor to fit width while maintaining aspect ratio
        img_width, img_height = self.image.size
        width_scale = canvas_width / img_width
        height_scale = canvas_height / img_height

        # Use the smaller scale to ensure image fits in both dimensions
        self.scale_factor = min(width_scale, height_scale)

        # Calculate new dimensions
        new_width = int(img_width * self.scale_factor)
        new_height = int(img_height * self.scale_factor)

        # Calculate offset to center the image
        self.offset_x = (canvas_width - new_width) / 2
        self.offset_y = (canvas_height - new_height) / 2

        # Scale the image
        scaled_image = self.image.resize(
            (new_width, new_height),
            Image.Resampling.LANCZOS)
        self.tk_image = ImageTk.PhotoImage(scaled_image)

        # Clear and redraw
        self.canvas.delete("all")
        self.canvas.create_image(
            self.offset_x, self.offset_y, image=self.tk_image, anchor="nw")

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
