import time

def terminate(*processes, force=True):
    """
    Terminates all given processes. If a process does not finish within a second, it is terminated
    (signal 15).

    Parameters
    ----------
    processes: varargs
        The processes to terminate.
    force: bool
        Whether to kill processes (with signal 9) in case they do terminate on signal 15 after 0.05
        seconds.
    """
    deadline = time.time() + 1
    for p in processes:
        p.join(max(deadline - time.time(), 0.1))
        if p.is_alive():
            p.terminate()
            if force:
                time.sleep(0.05)
                if p.is_alive():
                    p.kill()
