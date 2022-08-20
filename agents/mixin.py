from abc import ABC
from inspect import getmro
from typing import NoReturn


class ABCMixin(ABC):
    def __getattr__(self, name) -> NoReturn:
        """Fallback when __getattribute__ failed. Will raise AttributeError anyway, but
        with a more verbose message.

        Args:
            name (_type_): Object member name.

        Raises:
            AttributeError: Can't find member in object instance.

        Returns:
            NoReturn: Never returns, can only raise.
        """
        parent_classes_string = "`, `".join(
            e.__name__ for e in getmro(self.__class__) if issubclass(e, ABCMixin)
        )
        raise AttributeError(
            f"Can't find member `{name}` in object instance of type "
            f"`{parent_classes_string}`."
        )
