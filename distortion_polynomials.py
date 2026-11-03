"""Polynomial score functions for an integer draw distribution."""
from fractions import Fraction as F
from intervals import A, Jet, ZERO, ONE

class Law:
    def __init__(self, parameters):
        self.degrees = list(map(F, parameters['degrees']))
        self.weights = list(map(F, parameters['weights']))
        assert len(self.degrees) == len(self.weights) > 0
        assert all(d >= 1 and d.denominator == 1 for d in self.degrees)
        assert all(w > 0 for w in self.weights) and sum(self.weights) == 1
        self.mu = sum((w / (d + 1) for w, d in zip(self.weights, self.degrees)), F(0))
        self.den = 1 - self.mu
        self.ad = [int(d) for d in self.degrees]
        self.aw = list(map(A, self.weights))
        self.am = A(self.mu)
        self.an = A(self.den)

    def bH(self,x):
        x=x.nonnegative_part().min(ONE).nonnegative_part()
        return (self.am*x-sum((w*x**(d+1)/(d+1) for w,d in zip(self.aw,self.ad)),ZERO))/self.an
    def bpsi(self,x):
        x=x.nonnegative_part().min(ONE).nonnegative_part()
        return (self.am-sum((w*x**d for w,d in zip(self.aw,self.ad)),ZERO))/self.an
    def bgp(self,x):
        x=x.nonnegative_part().min(ONE).nonnegative_part()
        return sum((w*d*x**(d-1) for w,d in zip(self.aw,self.ad)),ZERO)/self.an

def jH(law,x):
    value=law.bH(x.v);derivative=law.bpsi(x.v)
    return Jet(value,[derivative*v for v in x.g])
def jpsi(law,x):
    value=law.bpsi(x.v);derivative=-law.bgp(x.v)
    return Jet(value,[derivative*v for v in x.g])

def jHd(law,x,y,width):
    # H(y)-H(x), with the geometrically known nonnegative width y-x.
    # These mean-value enclosures retain cancellation for short intervals.
    xx=x.v.nonnegative_part().min(ONE).nonnegative_part();yy=y.v.nonnegative_part().min(ONE).nonnegative_part()
    ww=width.v.nonnegative_part().min(ONE).nonnegative_part();hull=xx.union(yy).nonnegative_part().min(ONE).nonnegative_part()
    value=(law.bH(yy)-law.bH(xx)).intersection(ww*law.bpsi(hull))
    px=law.bpsi(xx);difference=(law.bpsi(yy)-px).intersection(-ww*law.bgp(hull))
    return Jet(value,[px*(b-a)+difference*b for a,b in zip(x.g,y.g)])
