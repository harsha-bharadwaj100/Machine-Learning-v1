from inspect import getmembers, ismodule


def explorer(module, indent="", explored=[]):
    members = getmembers(module, ismodule)
    for member in members:
        try:
            # if member[0] in module.__dict__["__all__"]:
            # check if it not in explored
            if member[1] not in explored and str(module.__name__) in str(
                member[1].__name__
            ):
                print(indent + str(member[1].__name__), type(member[1]))
                explored.append(member[1])
                explorer(member[1], indent=indent + "\t", explored=explored)
        except KeyError:
            continue


import sklearn

explorer(sklearn)
