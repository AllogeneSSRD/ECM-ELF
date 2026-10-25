\\ Independent PARI/GP model of GKL Table 1/2 and production Atkin-Morain.
\\ family: 1=GKL(iv), 2=Atkin-Morain, 3=generic GKL row-4 with t=seed.
ecmf_model(family,m,p=0) = {
  my(E,Q,s,r,t,e,a,d,x,y,alpha,beta,h,den,A,B,u,P);
  if(m<1,error("seed must be positive"));
  if(family==4 || family==5,
    if(family==4,a=-1;x=825/2752;y=1521/1504;d=(-x^2+y^2-1)/(x^2*y^2),
      a=1;d=25921/83521;x=13/7;y=289/49);
    if(p,d=Mod(d,p);x=Mod(x,p);y=Mod(y,p));e=0,
  if(family==2,
    E=ellinit([0,0,0,-8,-32],p);
    Q=ellmul(E,[12,40],m);if(Q==[0],return([]));
    s=Q[1];r=Q[2];den=r+s+16;if(den==0,return([]));
    alpha=(s-9)/den;den=8*alpha^2-1;if(den==0,return([]));
    beta=2*alpha*(4*alpha+1)/den;h=2*beta-1;if(h==0,return([]));
    a=1;d=(2*h^2-1)/h^4;den=6*beta-5;if(den==0,return([]));
    x=h*(4*beta-3)/den;den=(r+3*s-2)*(r+s+16);if(den==0,return([]));
    y=h*(r^2+50*r-2*s^3+27*s^2-104)/den;e=0,
    if(family==1,
      E=ellinit([0,-1,0,-9,9],p);Q=ellmul(E,[5,8],m);
      if(Q==[0],return([]));s=Q[1];r=Q[2];if(r==4,return([]));
      t=(4*s+4)/(r-4),
      if(family!=3,error("unknown curve family"));t=if(p,Mod(m,p),m)
    );
    den=t^2-4;if(den==0,return([]));e=(t^2+4*t)/den;a=-1;d=-e^4;
    den=t^4+6*t^3+12*t^2+16*t;if(den==0,return([]));
    x=(2*t^3+2*t^2-8*t-8)/den;
    den=t^6+6*t^5+10*t^4+16*t^3+48*t^2+64*t;if(den==0,return([]));
    y=(t^6+6*t^5+10*t^4-16*t^3-48*t^2-32*t-32)/den
  ));
  if(d==0 || d==a,return([]));
  if(a*x^2+y^2 != 1+d*x^2*y^2,error("off-curve construction"));
  A=2*(a+d)/(a-d);B=4/(a-d);
  E=ellinit([0,A*B,0,B^2,0]);
  if(y==1,P=[0],if(y==-1,P=[0,0],u=(1+y)/(1-y);P=[B*u,B^2*u/x]));
  if(!ellisoncurve(E,P),error("invalid birational map"));
  [E,P,[a,d,x,y,e]];
};

ecmf_order(family,m,p) = {
  my(c=ecmf_model(family,m,p),E,P,n,r);
  if(#c==0,print(family,"|",m,"|",p,"|bad");return());
  E=c[1];P=c[2];n=ellcard(E);r=ellorder(E,P,[n,factor(n)]);
  if(n%r || (n-p-1)^2>4*p,error("invalid group or point order"));
  if(family==1 && n%16,error("family (iv) lost guaranteed factor 16"));
  if(family==2 && n%16,error("Atkin-Morain lost torsion subgroup"));
  if(family==3 && n%8,error("generic family lost torsion subgroup"));
  if(family==4 && n%6,error("C6 lost rational torsion"));
  if(family==5 && n%16,error("fixed Z2xZ8 lost rational torsion"));
  print(family,"|",m,"|",p,"|",n,"|",r,"|",valuation(n,2),"|",valuation(n,3),"|",valuation(r,2));
};
