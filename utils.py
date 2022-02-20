def compose(xs):
    def f(x):
        for g in xs:
            x = g(x)
        return x
    return f
