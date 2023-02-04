from threading import Lock

running = True
running_lock = Lock()
counter = 0
counter_lock = Lock()