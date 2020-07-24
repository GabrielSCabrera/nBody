from time import time
import numpy as np

class Counter:

    def __init__(self, tot_iter):
        self.counter = 0
        self.t0 = time()
        self.perc = 0
        self.tot_iter = tot_iter
        self.times = np.zeros(tot_iter)
        self.dt = 0
        print(f"\tStatus\t\t\tIn Progress {0:>3d}%", end = "")

    def __call__(self):
        self.counter += 1
        new_perc = int(100*self.counter/self.tot_iter)
        self.times[self.counter-1] = time()
        if int(time() - self.t0) > self.dt and self.counter > 1:
            self.perc = new_perc
            t_avg = np.mean(np.diff(self.times[:self.counter]))
            eta = t_avg*(self.tot_iter - self.counter)
            dd = int((eta//86400))
            hh = int((eta//3600)%24)
            mm = int((eta//60)%60)
            ss = int(eta%60)
            msg = f"\r\tStatus\t\t\tIn Progress {self.perc:>3d}% – "
            if dd > 0:
                msg += f"{dd:d} day(s) + "
            msg += f"{hh:02d}:{mm:02d}:{ss:02d}"
            print(msg, end = "")
        self.dt = time() - self.t0

    def close(self):
        dt = time() - self.t0
        dd = int((dt//86400))
        hh = int(dt//3600)%24
        mm = int((dt//60)%60)
        ss = int(dt%60)
        msg = ""
        if dd > 0:
            msg += f"{dd:d} day(s) + "
        msg += f"{hh:02d}:{mm:02d}:{ss:02d}"
        print(f"\r\tStatus\t\t\tComplete – Total Time Elapsed {msg}")
