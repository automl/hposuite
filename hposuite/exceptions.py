from __future__ import annotations


class OptBenchNotInstalledError(ImportError):
    """Raised when a benchmark or optimizer is not installed."""
    def __init__(self, module: str, msg: str) -> None:
        """Initialize the exception."""
        import re
        not_installed_str = re.search(r"['\"]([^'\"]+)['\"]", msg)
        not_installed_str = not_installed_str.group(1) if not_installed_str else "Unknown"
        match not_installed_str:
            case "Unknown":
                super()._init_(msg)
            case _:
                super().__init__(
                    f"{not_installed_str} is not installed in module {module}. "
                    "Please install it first."
                )
        self.msg = msg
        self.name = not_installed_str
        self.module = module

    def __str__(self) -> str:
        if self.name == "Unknown":
            return self.msg
        return (
            f"{self.name} is not installed in module {self.module}. Please install it first."
        )

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.name})"