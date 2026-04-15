__all__ = ["main", "launch_gui", "DeepSliceMainWindow"]


def __getattr__(name):
	if name == "main":
		from .app import main

		return main

	if name in {"launch_gui", "DeepSliceMainWindow"}:
		from .main_window import DeepSliceMainWindow, launch_gui

		if name == "launch_gui":
			return launch_gui
		return DeepSliceMainWindow

	raise AttributeError(f"module 'DeepSlice.gui' has no attribute '{name}'")
