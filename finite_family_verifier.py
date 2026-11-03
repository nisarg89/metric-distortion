"""Direct polynomial checks for a finite charge family; no scalar-criterion reductions.
Every residual is evaluated on its entire listed interval box.  This file has no
root-monotonicity, E-breakpoint, small-mass, or large-mass analytical implication.
"""
from pathlib import Path
from fractions import Fraction as F
from math import comb
import json, time, math, argparse

from certificate_paths import CODE, DATA, OUTPUT, load_data
from intervals import A, arb, ctx, Jet, ZERO, ONE, certify_box as check
from distortion_polynomials import Law, jH, jpsi, jHd

ROOT=OUTPUT
data=load_data('parameters.json')
assert data['schema']=='metric-distortion-v4-parameters-v1'
law=Law(data)
k=F(data['kappa']);C=F(data['C']);s=F(data['s']);ell=F(data['l']);b=F(data['b'])
assert law.degrees==list(map(F,[1,2,32]))
assert law.weights==list(map(F,['19/73','35/146','1/2']))
assert law.mu==F(data['mu'])==F(1085,4818)
assert 1+2*C==F(data['distortion'])==F(213609,100000)
assert C==F(113609,200000) and b==F(2840189,5000000)
assert 0<s<b<ell<1 and k>=1
bands=[tuple(map(F,row)) for row in data['bands']];es=sorted(set(E for r0,r1,E in bands))
prices={F(p):(F(beta),F(T)) for p,beta,T in data['prices']}
assert len(prices)==len(data['prices'])
assert min(prices)==s and max(prices)==ell
assert all(beta>=0 and theta>=k for beta,theta in prices.values())

def jpow(x,n):
    if not n:return Jet(1)
    xv=x.v.nonnegative_part().min(ONE).nonnegative_part()
    return Jet(xv**n,[n*xv**(n-1)*g for g in x.g])

def jdH_increment(x,h):
    # Exact polynomial (H(x+h)-H(x))/h-psi(x), including h=0.
    value=Jet(0)
    for wf,df in zip(law.weights,law.degrees):
        d=int(df)
        for j in range(2,d+2):
            value-=wf*F(comb(d+1,j),d+1)/law.den*jpow(x,d+1-j)*jpow(h,j-1)
    return value

def jdH(x,h):
    return jpsi(law,x)+jdH_increment(x,h)

def mix(x,y,t):
    x=x if isinstance(x,Jet) else Jet(x);y=y if isinstance(y,Jet) else Jet(y)
    out=x+(y-x)*t
    # Convex combination enclosure, with the same polynomial derivatives.
    out.v=out.v.intersection(x.v.union(y.v))
    return out

def jF0(x):
    return law.mu/law.den-sum((wf/(df+1)/law.den*jpow(x,int(df)) for wf,df in zip(law.weights,law.degrees)),Jet(0))

def jpdiff(x,y,width):
    # Exact polynomial psi(y)-psi(x), factored by the nonnegative width.
    out=Jet(0)
    width=Jet(width.v.nonnegative_part(),width.g)
    for wf,df in zip(law.weights,law.degrees):
        d=int(df)
        out-=wf/law.den*width*sum((jpow(y,d-1-i)*jpow(x,i) for i in range(d)),Jet(0))
    return out

def jJ(x,y):
    # [average(psi,[x,y])-psi(y)]/(y-x), extended polynomially.
    out=Jet(0)
    for wf,df in zip(law.weights,law.degrees):
        d=int(df)
        out+=wf/(df+1)/law.den*sum(((d-i)*jpow(y,d-1-i)*jpow(x,i) for i in range(d)),Jet(0))
    return out

def jB(x):
    """Exact polynomial H(x)/(1-x), including x=1."""
    out=Jet(0)
    for wf,df in zip(law.weights,law.degrees):
        out+=wf/(df+1)/law.den*sum((jpow(x,i) for i in range(1,int(df)+1)),Jet(0))
    return out

baseknots=sorted({F(0),F(1),s,ell,b}|set(prices)|{r for lo,hi,E in bands for r in (lo,hi)})

def rawprice(p0,p1):
    if p1<=s:return ('small',None,None,None,None)
    if p0>=ell:return ('large',None,None,None,None)
    keys=sorted(prices)
    i=max(i for i,x in enumerate(keys) if x<=p0)
    x0,x1=keys[i:i+2];beta0,T0=prices[x0];beta1,T1=prices[x1]
    bp=(beta1-beta0)/(x1-x0);tp=(T1-T0)/(x1-x0)
    return ('middle',beta0-bp*x0,bp,T0-tp*x0,tp)

# At every E, insert all rational p where theta=E or c=E.
# Verify the selected branch is the maximum on each resulting closed interval.
def cells(E):
    out=[]
    for p0,p1 in zip(baseknots,baseknots[1:]):
        mode,bc,bp,tc,tp=rawprice(p0,p1)
        knots={p0,p1}
        if mode=='middle':
            if tp:
                cut=(k*E-tc)/tp
                if p0<cut<p1:knots.add(cut)
            if bp!=k*E:
                cut=-bc/(bp-k*E)
                if p0<cut<p1:knots.add(cut)
        elif mode=='small':
            cut=(E-1)/4
            if p0<cut<p1:knots.add(cut)
        cuts=sorted(knots)
        for lo,hi in zip(cuts,cuts[1:]):
            mid=(lo+hi)/2
            if mode=='small':branches=[F(0),k*(1+4*mid-E),k*(E-4*(1-mid))]
            elif mode=='large':branches=[F(0),k*(1-E),k*(E-1)]
            else:branches=[F(0),tc+tp*mid-k*E,k*E-(bc+bp*mid)/mid]
            branch=max(range(3),key=branches.__getitem__)
            # All three branch differences have at most quadratic numerators;
            # verify their signs exactly on the full mass interval, including vertices.
            def get_coeff(j):
                if mode=='small':return ([F(0)],[k*(1-E),4*k],[k*(E-4),4*k])[j]
                if mode=='large':return ([F(0)],[k*(1-E)],[k*(E-1)])[j]
                return ([F(0)],[F(0),tc-k*E,tp],[-bc,k*E-bp])[j] # multiply by p
            cj=get_coeff(branch)
            for jj in range(3):
                cc=get_coeff(jj);degree=max(len(cj),len(cc));coef=[(cj[i] if i<len(cj) else 0)-(cc[i] if i<len(cc) else 0) for i in range(degree)]
                sample=[lo,hi]
                if len(coef)==3 and coef[2]>0:
                    stationary=-coef[1]/(2*coef[2])
                    if lo<stationary<hi:sample.append(stationary)
                assert all(sum(c*x**i for i,c in enumerate(coef))>=0 for x in sample),(E,lo,hi,branch,jj,coef)
            out.append((lo,hi,mode,bc,bp,tc,tp,branch))
    return out

def fun(kind,cell,E,r0=None,r1=None):
    p0,p1,mode,bc,bp,tc,tp,branch=cell
    def f(z):
        p,u,v,w=z
        if kind=='low-first-bottom':
            a=(p if p1<=r0 else r0)*u;r=mix(r0,r1,v);first=True;wa=Jet(0)
        elif kind=='low-first-top':
            a=r0+((p if p1<=r1 else r1)-r0)*u;r=mix(a,r1,v);first=True;wa=Jet(0)
        elif kind=='low-internal':
            r=mix(r0,r1,w) if p1<=r0 else mix(p,r1,w)
            a=p*u;cap=mix(p,r,v);first=False;wa=wcap=Jet(0)
        elif kind=='high-first-bottom':
            a=(p if p1<=b else b)*u;r=mix(b,1,v);first=True;wa=Jet(0)
        elif kind=='high-first-top':
            a=b+(p-b)*u;r=mix(a,1,v);first=True;wa=C-a-k*E*jpsi(law,a)
        elif kind=='high-internal-bottom':
            a=p*u;cap=mix(p,b,v);r=mix(b,1,w);first=False;wa=wcap=Jet(0)
        elif kind=='high-internal-cross':
            a=(p if p1<=b else b)*u
            start=p if p0>=b else Jet(b);cap=mix(start,1,v);r=mix(cap,1,w)
            first=False;wa=Jet(0);wcap=C-cap-k*E*jpsi(law,cap)
        elif kind=='high-internal-top':
            a=b+(p-b)*u;cap=mix(p,1,v);r=mix(cap,1,w);first=False
            wa=C-a-k*E*jpsi(law,a);wcap=C-cap-k*E*jpsi(law,cap)
        else:raise ValueError(kind)
        rhs=C if first else wcap
        if mode=='large':
            epsilon=1-p
            if kind=='high-first-top':
                return k*jdH_increment(a,epsilon)+k*(E-1)*jpdiff(a,r,((1-p)+(p-b)*(1-u))*v)
            if first:val=a+k*jdH(a,epsilon)
            else:
                # All large-mass caps have cap=p+(1-p)*v.
                val=a+k*(v*jdH(a,epsilon*v)-(1-v)*jB(cap))
            return val+k*(E-1)*jpsi(law,r)+wa-rhs
        if first:D=jHd(law,a,a+1-p,1-p)
        else:D=-jH(law,a)-jHd(law,a+cap-p,cap,p-a)
        if mode=='small':
            ch=[Jet(0),k*(1+4*p-E),k*(E-4*(1-p))][branch]
            if not first and kind in ('low-internal','high-internal-bottom'):
                # Cancel the positive factor p; evaluate the remaining polynomial.
                aa=p*u
                left=aa+((r-p)*v if kind=='low-internal' else (b-p)*v)
                gap=(r-p)*(1-v) if kind=='low-internal' else (b-p)*(1-v)+(1-b)*w
                jj=jJ(left,cap)
                if E==1:
                    return u*(1-4*k*jF0(aa)+4*k*jpsi(law,r))+4*k*(1-u)*jpdiff(cap,r,gap)-4*k*p*jpow(1-u,2)*jj
                assert branch==0
                return u*(1-4*k*jF0(aa))-4*k*(1-u)*jpsi(law,cap)-4*k*p*jpow(1-u,2)*jj
            return a+4*k*D+ch*jpsi(law,r)+wa-rhs
        beta=bc+bp*p;T=tc+tp*p;den=p*(1-p)
        chn=[Jet(0),(T-k*E)*den,(k*E*p-beta)*(1-p)][branch]
        return (a+wa-rhs)*den+beta*D+chn*jpsi(law,r)
    return f

def tasks(E,cell,group='all'):
    p0,p1,*_=cell
    out=[]
    if group in ('all','low'):
        for r0,r1,ee in bands:
            if E!=ee:continue
            out.append(('low-first-bottom',r0,r1))
            if p0>=r0:out.append(('low-first-top',r0,r1))
            if p1<=r1:out.append(('low-internal',r0,r1))
    if group in ('all','high'):
        out.extend([('high-first-bottom',None,None),('high-internal-cross',None,None)])
        if p1<=b:out.append(('high-internal-bottom',None,None))
        if p0>=b:out.extend([('high-first-top',None,None),('high-internal-top',None,None)])
    return out

def signed_large_first_certificate():
    import sympy as sp
    aa,hh,rr,dd,ee=sp.symbols('a h r d e')
    cv=lambda q:sp.Rational(q.numerator,q.denominator)
    ps=lambda x:(cv(law.mu)-sum(cv(w)*x**int(d) for w,d in zip(law.weights,law.degrees)))/cv(law.den)
    hp=lambda x:(cv(law.mu)*x-sum(cv(w)*x**(int(d)+1)/(int(d)+1) for w,d in zip(law.weights,law.degrees)))/cv(law.den)
    inc=-sum(cv(w)/((int(d)+1)*cv(law.den))*sum(comb(int(d)+1,j)*aa**(int(d)+1-j)*hh**(j-1) for j in range(2,int(d)+2)) for w,d in zip(law.weights,law.degrees))
    dif=-dd*sum(cv(w)/cv(law.den)*sum(rr**(int(d)-1-i)*aa**i for i in range(int(d))) for w,d in zip(law.weights,law.degrees))
    poly=sp.Poly(cv(k)*(inc+ee*dif),aa,hh,rr,dd,ee)
    original=cv(k)*(sp.cancel((hp(aa+hh)-hp(aa))/hh)-(1+ee)*ps(aa)+ee*ps(rr))
    assert sp.Poly(sp.expand(original-poly.as_expr().subs(dd,rr-aa)),aa,hh,rr,ee).is_zero
    assert all(coef<=0 for coef in poly.coeffs())
    out={'status':'verified','identity':'original first residual = P(a,1-p,r,r-a,E-1)',
         'box':'a,h,r,d,e in [0,1]','method':'all exact monomial coefficients nonpositive',
         'terms':[{'powers':list(powers),'coefficient':str(coef)} for powers,coef in poly.terms()]}
    (ROOT/'large-first-polynomial-certificate.json').write_text(json.dumps(out,indent=2)+'\n')
    return {'nodes':1,'leaves':1,'monotone':0,'max_depth':0,'upper':0,'exact_polynomial_identity':True,'nonpositive_coefficients':len(poly.terms())}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--group',default='all',choices=['all','low','high']);ap.add_argument('--e');ap.add_argument('--limit',type=int,default=500000);ap.add_argument('--start',type=int,default=0);ap.add_argument('--count',type=int,default=100000)
    args=ap.parse_args();start=time.monotonic();total=0;rows=[];done=0;large_first=signed_large_first_certificate()
    chosen=es if args.e is None else [F(args.e)]
    for E in chosen:
        cc=cells(E)
        for i,cell in enumerate(cc):
            if i<args.start or i>=args.start+args.count:continue
            p0,p1,*_=cell
            records=[]
            for kind,r0,r1 in tasks(E,cell,args.group):
                box=((p0,p1),(F(0),F(1)),(F(0),F(1)),(F(0),F(1)))
                if 'first' in kind:box=box[:-1]+((F(0),F(0)),)
                try:ans=large_first if kind=='high-first-top' and cell[2]=='large' else check(fun(kind,cell,E,r0,r1),box,limit=args.limit)
                except Exception as er:
                    print(json.dumps({'FAIL':True,'E':str(E),'cell':i,'p0':str(p0),'p1':str(p1),'kind':kind,'error':repr(er)}),flush=True);raise
                total+=ans['nodes'];records.append({'kind':kind,'root_interval':[str(r0),str(r1)],**ans})
            rows.append({'E':str(E),'cell':i,'p0':str(p0),'p1':str(p1),'checks':records});done+=1
            if done%10==0:print(json.dumps({'E':str(E),'cells':done,'nodes':total,'seconds':time.monotonic()-start}),flush=True)
    out={'status':'partial_direct_bounds_verified','all_analytical_reductions_removed_from_checker':True,'group':args.group,'coefficients':list(map(str,chosen)),'nodes':total,'seconds':time.monotonic()-start,'rows':rows}
    filename='finite-family-'+args.group+'-'+('-'.join(str(e).replace('/','_') for e in chosen))+'-'+str(args.start)+'-'+str(args.count)+'.json'
    (ROOT/filename).write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({kk:vv for kk,vv in out.items() if kk!='rows'}),flush=True)
if __name__=='__main__':main()
