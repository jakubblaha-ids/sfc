"""
Entry point for running the package as a module with `python -m src`
"""

from .app import App

if __name__ == "__main__":
    app = App()
    app.run()
