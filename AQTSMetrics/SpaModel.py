class SPA:
    def __init__(self, data):
        set_ = set()
        self.data = data

        for state in self.data.keys():
            set_.add(state)
        self.states = sorted(list(set_))

        set_.clear()
        for state, value in self.data.items():
            if isinstance(value, dict):
                set_.update(value.keys())
        self.labels = list(set_)

    def pretty_print(self):
        for key, value in self.data.items():
            print(key, value)

    def squiggly_l(self, s, a):
        return self.data[s].get(a, [])