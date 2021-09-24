class LookupTable:
    def __init__(self, keys, allow_unk=True):
        """
        Args:
            keys: the tokens.
            allow_unk: if True, #tokens will increased by one as unk is appended. OOV will not raise an error.
        """
        self.allow_unk = allow_unk
        keys = sorted(set(keys))
        self.table = {key: i for i, key in enumerate(keys)}
        unk = {"<unk>": len(list(self.table.keys()))} if self.allow_unk else {}
        blank = {'blank': len(list(self.table.keys())) + 1}
        self.table = {**self.table, **unk, **blank}
        #self.table['blank'] = len(self)

    def __call__(self, key):
        if key in self.table:
            return self.table[key]
        elif self.allow_unk:
            return self.table['<unk>']
        raise KeyError(key)

    def __len__(self):
        return len(self.table)  #blank

    def __str__(self):
        # unk = {"unk": len(self) - 2} if self.allow_unk else {}
        # blank = {'blank': len(self) - 1}
        return str({**self.table})  #blank is always the last token
