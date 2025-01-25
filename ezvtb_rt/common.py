from abc import ABC, abstractmethod
import numpy as np
import collections
import time

class Core(ABC):
    @abstractmethod
    def setImage(self, img:np.ndarray):
        pass  # Call at initialization

    @abstractmethod
    def syncFetchRes(self) -> np.ndarray:
        pass  # sync to get ONE OF the results 
    
    @abstractmethod
    def finishedFetch(self) -> bool:
        pass  # if fetch finished

    @abstractmethod
    def infer(self, pose:np.ndarray):
        pass  # put inference into async running, and reload result

    intervals:collections.deque = collections.deque(maxlen=50)
    intervalSum:float = 0
    startTimestamp:float = time.time()

    def startCount(self):
        self.startTimestamp = time.time()

    def endCount(self):
        if len(self.intervals) >= self.intervals.maxlen:
            self.intervalSum -= self.intervals[0]
        self.intervals.append(time.time() - self.startTimestamp)
        self.intervalSum += self.intervals[-1]

    @abstractmethod
    def averageInterval(self) -> float:
        pass
        # if len(self.intervals) >= 1:
        #     return self.intervalSum / len(self.intervals) 
        # else:
        #     return 0.001