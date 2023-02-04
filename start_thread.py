import threading
import time
from draw_network import *
import global_vars

def counter():
    for i in range(100):
        time.sleep(0.1)
        with global_vars.counter_lock:
            global_vars.counter = i
            print(global_vars.counter)
    with global_vars.running_lock:
        global_vars.running = False

# thrd1 = threading.Thread(target=counter, args=('Thread-1',))
# thrd2 = threading.Thread(target=counter, args=('Thread-2',))

# thrd1.start()
# thrd2.start()
# print(threading.active_count())

brain_analyser_thread = threading.Thread(target=brain_analyser)
brain_analyser_thread.start()

brain_thread = threading.Thread(target=counter, daemon=True)
brain_thread.start()

brain_analyser_thread.join()