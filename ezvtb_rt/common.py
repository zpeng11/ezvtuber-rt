from abc import ABC, abstractmethod
import numpy as np
from typing import List

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