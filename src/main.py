import tkinter as tk
from tkinter import ttk
from constants import *
from PIL import Image, ImageTk
from editor import MapEditor


class App:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Robot Localization via Modern Hopfield Networks")
        self.root.geometry(f"{SCREEN_WIDTH}x{SCREEN_HEIGHT}")

        # Map Data
        self.current_map_image = Image.new(
            "RGB", (MAP_WIDTH, MAP_HEIGHT), "white")
        self.tk_map_image = None

        # Configure styles
        self.style = ttk.Style()
        self.style.theme_use('clam')

        self.create_layout()
        self.create_toolbar()
        self.create_left_panel()
        self.create_right_panel()

    def create_layout(self):
        # Main container
        self.main_container = ttk.Frame(self.root)
        self.main_container.pack(fill=tk.BOTH, expand=True)

        # Toolbar area
        self.toolbar_frame = tk.Frame(
            self.main_container, height=TOOLBAR_HEIGHT)
        self.toolbar_frame.pack(side=tk.TOP, fill=tk.X)
        self.toolbar_frame.pack_propagate(False)  # Prevent shrinking

        # Content area
        self.content_frame = ttk.Frame(self.main_container)
        self.content_frame.pack(
            side=tk.TOP, fill=tk.BOTH, expand=True, padx=PANEL_PADDING,
            pady=PANEL_PADDING)

    def create_toolbar(self):
        buttons = {
            "Edit Map": self.open_map_editor,
            "Import Map": lambda: print("Import Map"),
            "Export Map": lambda: print("Export Map"),
            "Auto Sample": lambda: print("Auto Sample")
        }

        for btn_text, command in buttons.items():
            btn = tk.Button(
                self.toolbar_frame,
                text=btn_text,
                command=command
            )
            btn.pack(side=tk.LEFT, padx=5, pady=5)

    def create_left_panel(self):
        # Left Panel Container
        self.left_panel = tk.Frame(
            self.content_frame, width=MAP_WIDTH,
            height=MAP_HEIGHT)
        self.left_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=False)
        self.left_panel.pack_propagate(False)

        # Label
        lbl = tk.Label(
            self.left_panel, text="World View (God Mode)")
        lbl.pack(side=tk.TOP, pady=5)

        # Canvas for Map
        self.map_canvas = tk.Canvas(
            self.left_panel, highlightthickness=1)
        self.map_canvas.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Placeholder text
        self.map_canvas.create_text(
            MAP_WIDTH // 2, MAP_HEIGHT // 2, text="Map Area (800x450)",
            fill="gray")

    def create_right_panel(self):
        # Right Panel Container
        self.right_panel = tk.Frame(self.content_frame)
        self.right_panel.pack(side=tk.LEFT, fill=tk.BOTH,
                              expand=True, padx=(PANEL_PADDING, 0))

        # Label
        lbl = tk.Label(self.right_panel, text="Robot Perception",
                       font=("Arial", 12, "bold"))
        lbl.pack(side=tk.TOP, pady=10)

        # 1. Current Input
        input_frame = tk.Frame(self.right_panel)
        input_frame.pack(fill=tk.X, padx=10, pady=10)

        tk.Label(
            input_frame, text="Current Input").pack(
            anchor="w")
        self.input_canvas = tk.Canvas(
            input_frame, height=100, highlightthickness=0)
        self.input_canvas.pack(fill=tk.X, pady=(5, 0))

        # 2. Retrieved Memory
        memory_frame = tk.Frame(self.right_panel)
        memory_frame.pack(fill=tk.X, padx=10, pady=10)

        tk.Label(
            memory_frame, text="Retrieved Memory").pack(
            anchor="w")
        self.memory_canvas = tk.Canvas(
            memory_frame, height=100, highlightthickness=0)
        self.memory_canvas.pack(fill=tk.X, pady=(5, 0))

        # 3. Similarity Metric
        sim_frame = tk.Frame(self.right_panel)
        sim_frame.pack(fill=tk.X, padx=10, pady=10)

        tk.Label(
            sim_frame, text="Similarity Metric").pack(
            anchor="w")
        self.sim_canvas = tk.Canvas(
            sim_frame, height=40, highlightthickness=0)
        self.sim_canvas.pack(fill=tk.X, pady=(5, 0))

    def open_map_editor(self):
        MapEditor(self.root, self.current_map_image, self.on_map_saved)

    def on_map_saved(self, new_map_image):
        self.current_map_image = new_map_image
        self.update_map_display()

    def update_map_display(self):
        self.tk_map_image = ImageTk.PhotoImage(self.current_map_image)
        self.map_canvas.delete("all")
        self.map_canvas.create_image(
            0, 0, image=self.tk_map_image, anchor="nw")

    def run(self):
        self.update_map_display()
        self.root.mainloop()


if __name__ == "__main__":
    app = App()
    app.run()
