import os
import sys
import time
import datetime
import math

_ERASE_LINE = '\x1b[2K'

class ProgressBar:
    """
    The progress bar can be used to print progress for a specific task where either a specified
    number of work items or an iterable has to be processed. The progress bar also measures the
    execution time and provides an estimated remaining time for the operation. A common use case
    for the progress bar are for-loops where one work item is completed per iteration.

    Examples
    --------

    .. code-block:: python
        :caption: Iterate over a range of integers

        for i in ProgressBar(4): # equivalently initialized as ProgressBar(range(4))
            time.sleep(2.5)

    .. code-block:: python
        :caption: Iterate over an iterable of known length

        l = [1, 5, 7]
        for i in ProgressBar(l):
            time.sleep(2.5)

    .. code-block:: python
        :caption: Iterate over an iterable of unknown size

        it = (x + 1 for x in range(3))
        for i in ProgressBar(it):
            time.sleep(1.5)

    .. code-block:: python
        :caption: Visualize some complex manual progress

        with ProgressBar() as p:
            time.sleep(3)
            p.step()
            for _ in range(10):
                time.sleep(0.1)
                p.step()
            for _ in range(5):
                time.sleep(0.4)
                p.step()
    """

    ########################################################################################
    ### INITIALIZATION
    ########################################################################################
    def __init__(self, iterable=None, denom=None, file=None, verbose=True):
        """
        Initializes a new progress bar with the given number of work items.

        Parameters
        ----------
        iterable: int or iterable, default: None
            Either the number of work items to be processed or an iterable whose values are returned
            when iterating over this progress bar. If no value is given, this iterable can not be
            used within for-loops.
        denom: int, default: None
            If the first parameter is an integer, this value may also be given. In that case, the
            first parameter acts as the numerator and the second parameter as the denominator. The
            rounded up division of these two values is used as the number of work items.
        file: str, default: None
            If given, defines the file where the progress bar should write to instead of the
            command line. Intermediate directories are created automatically.
        verbose: bool, default: True
            Whether to actually log anything. This is useful in cases where logging should be
            turned of dynamically without introducing additional control flow.
        """
        if not (denom is None or (isinstance(denom, int) and isinstance(iterable, int))):
            raise ValueError("If second parameter is given, first parameter must be integer.")

        self.verbose = verbose
        if file is not None:
            os.makedirs(file, exist_ok=True)
            self.stream = open(file, 'a+')
        else:
            self.stream = sys.stdout

        if iterable is None:
            self.iterable = None
        elif hasattr(iterable, '__iter__'):
            self.iterable = iterable
        elif isinstance(iterable, int) and denom is not None:
            self.iterable = range(math.ceil(iterable / denom))
        elif isinstance(iterable, int):
            self.iterable = range(iterable)
        else:
            raise ValueError(
                f"First parameter must be iterable or integer but found {type(iterable)}."
            )

        self.haslength = hasattr(self.iterable, '__len__')
        if self.haslength:
            self.iteration_max = len(self.iterable)

        self._iteration_count = 0
        self._start_time = None
        self._latest_print_length = None

    ########################################################################################
    ### INSTANCE METHODS
    ########################################################################################
    def start(self):
        """
        Starts to record the progress of the operation. Time measuring is initiated and the
        beginning of the operation is indicated on the command line.

        Note
        ----
        This method should usually not be called explicitly. It is implicitly called at the
        beginning of a :code:`with` statement.
        """
        self._iteration_count = 0
        self._start_time = time.time()
        self._latest_print_length = 0
        self._print_progress(compute_eta=False)

    def step(self):
        """
        Tells the progress bar that one work item has been processed. The command line output is
        updated as well as the estimated finishing time of the operation.

        If used from within a with statement, this method must be called explicitly, otherwise, it
        should not be called.
        """
        self._iteration_count += 1
        self._print_progress()

    def finish(self, kv=None):
        """
        Stops the progress bar and prints the total duration of the operation. If metrics are given,
        these will be printed along with the elapsed time and the number of iterations per second.
        Metrics may be provided in the form :code:`<name>__<format specifier>` to e.g. format
        floating point numbers with a fixed number of decimal points.

        If key value pairs are given, this method must be called explicitly, otherwise, it is
        implicitly called at the end of a with statement or for loop.

        Note
        ----
        If this method is called mutliple times with not calls to :meth:`start` in between, all but
        the first call are no-ops.

        Parameters
        ----------
        metrics: dict, default: None
            The metrics to print as key-value pairs. Usually, they provide more information about
            the operation whose progress has been tracked.
        """
        if self._start_time is None:
            return
        if kv is None:
            kv = {}
        self._print_done(kv)
        self._start_time = None
        self._latest_print_length = None

    ########################################################################################
    ### SPECIAL METHODS
    ########################################################################################
    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.finish()
        return False

    def __iter__(self):
        if self.iterable is None:
            raise ValueError("ProgressBar must be given an iterable if used within a for-loop.")

        self.start()
        it = iter(self.iterable)
        try:
            while True:
                yield next(it)
                self.step()
        except StopIteration:
            self.finish()

    ########################################################################################
    ### PRIVATE METHODS
    ########################################################################################
    def _print_progress(self, compute_eta=True):
        if not self.verbose:
            return

        elapsed = time.time() - self._start_time
        elapsed_time = datetime.timedelta(0, int(elapsed))
        it_per_sec = self._iteration_count / elapsed

        if self.haslength:
            perc = self._iteration_count / self.iteration_max
            p = int(round(perc * 30))
            pbar = "" if p == 0 else "=" * (p - 1) + ">" if p < 30 else "=" * 30
            whitespace = " " * (30 - p)
            progress = f"[{pbar}{whitespace}] ({perc:02.1%})"
            if compute_eta:
                eta = datetime.timedelta(0, int((1 - perc) / perc * elapsed))
            else:
                eta = "n/a"
        else:
            progress = f"[{self._iteration_count:,} iterations]"
            eta = "n/a"

        text = " {} ETA {} [Elapsed {} | {:,.2f} it/s]".format(
            progress, eta, elapsed_time, it_per_sec
        )
        print(f"{_ERASE_LINE}{self._pad_whitespace(text)}", end='\r', file=self.stream)
        self.stream.flush()

    def _print_done(self, metrics):
        if not self.verbose:
            return

        elapsed = time.time() - self._start_time
        elapsed_time = datetime.timedelta(0, int(elapsed))
        it_per_sec = self._iteration_count / elapsed

        m_strings = []
        for k, v in sorted(metrics.items(), key=lambda k: k[0]):
            split = k.split('__')
            if len(split) > 1:
                f = split[1]
            else:
                f = '{:.5f}'
            string = f'{split[0]}: {f}'.format(v)
            m_strings += [string]

        text = " [Elapsed {} | {:,.2f} it/s] {}".format(
            elapsed_time, it_per_sec, ", ".join(m_strings)
        )
        print(f"{_ERASE_LINE}{self._pad_whitespace(text)}", file=self.stream)
        self._latest_print_length = 0
        self.stream.flush()

    def _pad_whitespace(self, text):
        diff = self._latest_print_length - len(text)
        self._latest_print_length = max(len(text), self._latest_print_length)
        if diff > 0:
            return text + " " * diff
        return text

    def __del__(self):
        if self.stream != sys.stdout:
            self.stream.close()


def ensure_valid_directories(file):
    """
    Creates intermediate directories to ensure that files can be saved to their specified directory.

    Parameters
    ----------
    file: str
        The absolute path to the file whose directory structure to ensure.
    """
    directory = "/".join(file.split('/')[:-1])
    os.makedirs(directory, exist_ok=True)
