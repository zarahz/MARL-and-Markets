class DictList(dict):
    """A dictionnary of lists of same size. Dictionnary items can be
    accessed using `.` notation and list items using `[]` notation.
    Example:
        >>> d = DictList({"a": [[1, 2], [3, 4]], "b": [[5], [6]]})
        >>> d.a
        [[1, 2], [3, 4]]
        >>> d[0]
        DictList({"a": [1, 2], "b": [5]})
    """

    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__

    def __len__(self):
        return len(next(iter(dict.values(self))))

    def __getitem__(self, index):
        return DictList({key: value[index] for key, value in dict.items(self)})

    def _getagentitem__(self, index, agents):
        returnObj = {}
        for key, _ in dict.items(self):
            returnObj[key] = {}
            if key == 'action' or key == 'actions':
                for agent in range(agents):
                    agent_value = DictList()
                    agent_value.action = dict.__getitem__(self, key)[agent]
                    returnObj[key][agent] = agent_value.__getitem__(index)
            elif key == 'log_prob' or key == 'log_probs':
                for agent in range(agents):
                    agent_value = DictList()
                    agent_value.log_prob = dict.__getitem__(self, key)[agent]
                    returnObj[key][agent] = agent_value.__getitem__(index)
            else:
                returnObj[key] = dict.__getitem__(
                    self, key).__getitem__(index)
        return DictList(returnObj)

    def __setitem__(self, index, d):
        for key, value in d.items():
            dict.__getitem__(self, key)[index] = value
