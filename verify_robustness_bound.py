"""Exact certificate for (1-p) lambda(p) + tau_0(p) <= 16.

Uses only Fraction arithmetic. On every closed mass interval it clears the
positive denominator p(1-p), converts the resulting degree-at-most-three
polynomial to the Bernstein basis, and checks every coefficient is nonnegative.
The separately defined endpoint values p=0,1 are checked independently.
"""
from pathlib import Path
from fractions import Fraction as F
from math import comb
import hashlib,json
from certificate_paths import DATA, OUTPUT, load_data
ROOT=DATA
WITNESS=ROOT/'rank-cut-witness.json'
w=load_data('rank-cut-witness.json')

def add(a,b):
    out=[F(0)]*max(len(a),len(b))
    for i,x in enumerate(a):out[i]+=x
    for i,x in enumerate(b):out[i]+=x
    return out

def mul(a,b):
    out=[F(0)]*(len(a)+len(b)-1)
    for i,x in enumerate(a):
        for j,y in enumerate(b):out[i+j]+=x*y
    return out

def bernstein(poly,lo,hi):
    while len(poly)>1 and poly[-1]==0:poly=poly[:-1]
    n=len(poly)-1
    # Substitute p = lo + (hi-lo) t, then convert powers of t to Bernstein form.
    power=[sum((poly[k]*comb(k,j)*lo**(k-j)*(hi-lo)**j
                for k in range(j,n+1)),F(0)) for j in range(n+1)]
    return [sum((power[j]*F(comb(k,j),comb(n,j)) for j in range(k+1)),F(0))
            for k in range(n+1)]

assert list(map(int,w['g_degrees']))==[1,2,32]
assert list(map(F,w['g_weights']))==[F(19,73),F(35,146),F(1,2)]
rows=[]
last=F(0)
for index,cell in enumerate(w['mass_cells']):
    lo,hi=F(cell['p0']),F(cell['p1'])
    assert lo==last and lo<hi<=1
    last=hi
    lam=list(map(F,cell['lambda_numerator']))
    tau=list(map(F,cell['charge_numerators'][0]))
    # 16 p(1-p) - (1-p) L(p) - T_0(p).
    poly=add([F(0),F(16),F(-16)],[-x for x in add(mul(lam,[F(1),F(-1)]),tau)])
    coefficients=bernstein(poly,lo,hi)
    assert min(coefficients)>=0,(index,lo,hi)
    rows.append({'index':index,'p0':str(lo),'p1':str(hi),
                 'bernstein_coefficients':[str(x) for x in coefficients]})
assert last==1
# The manuscript specifies lambda(0)=lambda(1)=tau_j(0)=0, tau_j(1)=eta_j.
assert w['endpoints']=={'lambda(0)':'0','lambda(1)':'0','t_j(0)':'0','t_j(1)':'eta_j'}
assert 0<=F(w['eta'][0])<=16
result={'status':'verified','bound':16,'domain':'0 <= p <= 1',
        'witness_sha256':hashlib.sha256(WITNESS.read_bytes()).hexdigest(),
        'checker_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'polynomial_intervals':len(rows),'maximum_degree':max(len(x['bernstein_coefficients'])-1 for x in rows),
        'subdivision_needed':False,'endpoint_values_checked':True,'rows':rows}
(OUTPUT/'robustness-bound-verification.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k!='rows'}))
