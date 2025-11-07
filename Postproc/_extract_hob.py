

def mf_read_comment(file, comment='#'):
    c = []
    while True:
        l = file.readline()
        if l.startswith(comment):
            c.append(l)
        else:
            return(l, ''.join(c))

def parse_line_items(line, totype=str, n=9999):
    return [totype(t) for t in line.replace(",", " ").replace("\t", " ").split()[:n]]


class mf_hob:
    """
    read MNW package
    """
    def __init__(self, hobfile=None):
        self.file=hobfile
        self.comments = None
        if hobfile is not None:
            self.read_package()

    def read_package(self, ):

        with open(self.file, 'r') as f:
            l, self.comments = mf_read_comment(f)
            lines = f.readlines()
            lines.insert(0, l)

        nline = -1
        # Data Set 1
        lit = parse_line_items(lines[nline:=nline+1], int, n=4)
        for v, s in zip('NH MOBS MAXM IUHOBSV HOBDRY'.split(), lit[:j]):
            setattr(self, v, float(s) if v.startswith('H') else int(s))
