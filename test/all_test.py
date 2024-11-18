from core_test import CorePerf, CoreTestShow
from rife_test import RIFETestPerf, RIFETestShow
from tha_test import THATestPerf, THATestShow
import os


if __name__ == "__main__":
    os.makedirs('./test/data/rife', exist_ok=True)
    RIFETestPerf()
    RIFETestShow()
    os.makedirs('./test/data/tha', exist_ok=True)
    THATestPerf()
    THATestShow()
    os.makedirs('./test/data/core', exist_ok=True)
    CorePerf()
    CoreTestShow()