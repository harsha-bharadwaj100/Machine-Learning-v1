import sys
import inspect
from sklearn import *

g = globals().copy()


# recursively get all modules, classes and functions
def explorer():
    for name, obj in g.items():
        if inspect.ismodule(obj):
            print(obj.__name__)
            # print all classes and submodules within the module
            for name, obj in obj.__dict__.items():
                if inspect.ismodule(obj):
                    print(obj.__name__)
                if inspect.isclass(obj) and not inspect.isabstract(obj):
                    print(obj.__name__)
                if inspect.isfunction(obj) and not inspect.isabstract(obj):
                    print(obj.__name__)
        if inspect.isclass(obj):
            print(obj.__name__)
        if inspect.isfunction(obj):
            print(obj.__name__)


explorer()
