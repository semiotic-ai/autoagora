import functools
import logging


def async_exit_on_exception(exit_code: int = -1):
    """Returns decorator that logs any exception and exits the program immediately.

    The goal of this function is to easily trigger an immediate program abort from any
    asynchronous function.

    Args:
        exit_code (int, optional): Self explanatory. Defaults to -1.
    """

    def decorator(func):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            try:
                return await func(*args, **kwargs)
            except:
                logging.exception("exit_on_exception triggered")
                exit(exit_code)

        return wrapper

    return decorator
