__all__ = ["DSModel", "launch_gui"]


def __getattr__(name):
	if name == "DSModel":
		from .main import DSModel

		return DSModel

	if name == "launch_gui":
		try:
			from .gui import launch_gui
		except Exception:
			launch_gui = None
		return launch_gui

	raise AttributeError(f"module 'DeepSlice' has no attribute '{name}'")
