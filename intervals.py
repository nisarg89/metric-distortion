"""Proof-independent, outward interval bounds for differentiable functions.

All acceptance decisions use Arb enclosures. Floating point is used only to
choose a subdivision coordinate and to print diagnostic statistics.
"""
from fractions import Fraction as F
import math
from flint import arb, fmpq, ctx

ctx.prec = 160
ZERO = arb(0)
ONE = arb(1)

def Q(x):
    if isinstance(x,F):return x
    return F(str(x))
def A(x):
    x=Q(x);return arb(fmpq(x.numerator,x.denominator))
def bounds(x):
    return Q(x.lower().fmpq()),Q(x.upper().fmpq())
def interval(l,u):
    return A(l).union(A(u))
def upper(x):return float(x.upper())
def lower(x):return float(x.lower())
def magnitude(x):return x.abs_upper()
class Jet:
    def __init__(self,v,g=None):self.v=v if isinstance(v,arb) else A(v);self.g=[ZERO]*4 if g is None else g
    @staticmethod
    def var(v,i):
        g=[ZERO]*4;g[i]=ONE;return Jet(v,g)
    def __add__(self,o):
        o=jet(o);return Jet(self.v+o.v,[a+b for a,b in zip(self.g,o.g)])
    __radd__=__add__
    def __neg__(self):return Jet(-self.v,[-x for x in self.g])
    def __sub__(self,o):return self+-jet(o)
    def __rsub__(self,o):return jet(o)+-self
    def __mul__(self,o):
        o=jet(o);return Jet(self.v*o.v,[a*o.v+self.v*b for a,b in zip(self.g,o.g)])
    __rmul__=__mul__
    def __truediv__(self,o):
        o=jet(o);return Jet(self.v/o.v,[(a*o.v-self.v*b)/(o.v*o.v) for a,b in zip(self.g,o.g)])
    def __rtruediv__(self,o):return jet(o)/self

def jet(x):return x if isinstance(x,Jet) else Jet(x)
class Failure(Exception):pass

def certify_box(fun,box,limit=50000,depth=0,stats=None):
    if stats is None:stats={'nodes':0,'leaves':0,'monotone':0,'max_depth':0,'upper':-math.inf}
    stack=[(box,depth)]
    while stack:
        b,d=stack.pop();stats['nodes']+=1;stats['max_depth']=max(stats['max_depth'],d)
        if stats['nodes']>limit:raise Failure(('box_limit',stats))
        z=[Jet.var(interval(l,u),i) for i,(l,u) in enumerate(b)];f=fun(z)
        assert f.v.is_finite(),('nonfinite',b,str(f.v))
        if f.v<=0:
            stats['leaves']+=1;stats['upper']=max(stats['upper'],upper(f.v));continue
        # The interval mean value theorem bounds the entire box using
        # enclosed partial derivatives.
        mid=[(l+u)/2 for l,u in b];m=fun([Jet(A(x)) for x in mid]).v
        if m>0:raise Failure(('positive_midpoint',[[str(l),str(u)] for l,u in b],str(m)))
        hh=[(u-l)/2 for l,u in b]
        ub=m+sum((magnitude(g)*A(h) for g,h in zip(f.g,hh)),ZERO)
        if ub<=0:
            stats['leaves']+=1;stats['upper']=max(stats['upper'],upper(ub));continue
        collapsed=False
        for i,((l,u),g) in enumerate(zip(b,f.g)):
            if l==u:continue
            if g>=0 or g<=0:
                q=u if g>=0 else l;bb=list(b);bb[i]=(q,q);stack.append((tuple(bb),d+1));stats['monotone']+=1;collapsed=True;break
        if collapsed:continue
        scores=[float(magnitude(g))*float(h) for g,h in zip(f.g,hh)]
        i=max(range(len(b)),key=lambda k:scores[k]);l,u=b[i]
        if l==u or d>90:raise Failure(('stalled',b,str(ub),stats))
        q=(l+u)/2;left=list(b);right=list(b);left[i]=(l,q);right[i]=(q,u)
        stack.extend([(tuple(right),d+1),(tuple(left),d+1)])
    return stats
