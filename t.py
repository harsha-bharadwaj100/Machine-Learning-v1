import sklearn
import inspect

import sklearn.model_selection

# from sklearn import
# base = sklearn.__dict__.items()

# # print(base)

# # iterate through the base items and print all their contents
# for name, obj in sklearn.__dict__.items():
#     if inspect.ismodule(obj):
#         print(obj.__name__)

# print(dir(sklearn)[-10])

from inspect import getmembers

# list = getmembers(sklearn, inspect.ismodule)

# print(list[-7])

# print(list[-7][1].__dict__["__all__"])

# sub = getmembers(list[-7][1])

# print(sub)

# print(sklearn.__dict__["__all__"])


def explorer(module, indent=""):
    members = getmembers(module)
    for member in members:
        try:
            if member[0] in module.__dict__["__all__"]:
                print(indent + member[1].__name__, type(member[1]))
                explorer(member[1], indent=indent + "\t")
        except KeyError:
            continue


explorer(sklearn)
