def encode_symbol(o, h, l, c):
    if c > o:
        return 'B'
    elif c < o:
        return 'D'
    else:
        return 'X'
