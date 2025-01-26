from abc import ABC, abstractmethod
import numpy as np
from typing import List

class Core(ABC):
    @abstractmethod
    def setImage(self, img:np.ndarray):
        pass  # Call at initialization

    @abstractmethod
    def inference(self, pose:np.ndarray) -> List[np.ndarray]:
        pass  # sync to get ONE OF the results 
    