import time


def retry(func, max_retries=3, delay=1):
    def wrapper(*args, **kwargs):
        for i in range(max_retries):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                print(f"Retry {i+1} failed: {e}")
                time.sleep(delay)
        raise Exception(f"All {max_retries} retries failed.")

    return wrapper
