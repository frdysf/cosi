# TODO: for every_n_epoch
def conditional_decorator(condition):
    def decorator(func):
        def wrapper(*args, **kwargs):
            if condition:
                return func(*args, **kwargs)
            else:
                # Optionally, return None or another default value
                return
        return wrapper
    return decorator