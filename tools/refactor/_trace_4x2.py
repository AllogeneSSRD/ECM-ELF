"""4x2 sliced CIOS close-to-C trace, 2^61-1."""
import random; random.seed(42)
DLIMBS=2;LANES=4;LIMBS=8;B32=2**32
N_int=2**61-1;R_val=B32**LIMBS;R_inv=pow(R_val,-1,N_int);np0=pow(-N_int,-1,B32)
def L(x):return[(x>>(32*i))&0xFFFFFFFF for i in range(LIMBS)]
def U(L):return sum(l<<(32*i) for i,l in enumerate(L))

A_raw=random.randint(1,N_int-1);B_raw=random.randint(1,N_int-1)
A_mont=(A_raw*R_val)%N_int;B_mont=(B_raw*R_val)%N_int
expected=(A_mont*B_mont*R_inv)%N_int
A=L(A_mont);B=L(B_mont);NN=L(N_int)

print(f"N={N_int:#x} np0={np0:#x}")
print(f"A={[f'{x:#010x}' for x in A]}")
print(f"NN={[f'{x:#010x}' for x in NN]}")
print(f"Expected: {U(L(expected)):#018x}")

def bp(src,arr):return arr[src%LANES]

my_T=[[0,0],[0,0],[0,0],[0,0]]
my_A=[[A[0],A[1]],[A[2],A[3]],[A[4],A[5]],[A[6],A[7]]]
my_B=[[B[0],B[1]],[B[2],B[3]],[B[4],B[5]],[B[6],B[7]]]
my_N=[[NN[0],NN[1]],[NN[2],NN[3]],[NN[4],NN[5]],[NN[6],NN[7]]]
tN=tN1=0

for i in range(LIMBS):
    i_lane=i//DLIMBS;i_off=i%DLIMBS;A_i=my_A[i_lane][i_off]

    # P1: product per lane (inner dlimb loop)
    carry=[0]*LANES
    for lid in range(LANES):
        c=0
        for d in range(DLIMBS):
            uv=my_T[lid][d]+A_i*my_B[lid][d]+c;my_T[lid][d]=uv&0xFFFFFFFF;c=uv>>32
        carry[lid]=c
    # Lane carry: lid>0 gets from lid-1 via ds_bpermute
    for lid in range(LANES):
        ci=carry[(lid-1)%LANES] if lid>0 else 0
        for d in range(DLIMBS):
            uv=my_T[lid][d]+ci;my_T[lid][d]=uv&0xFFFFFFFF;ci=uv>>32
            if ci==0:break
    C31=carry[LANES-1]
    t_lo=tN+C31;tN=t_lo&0xFFFFFFFF;tN1+=t_lo>>32

    # P2: reduce
    m_val=(my_T[0][0]*np0)&0xFFFFFFFF
    carry=[0]*LANES
    for lid in range(LANES):
        c=0
        for d in range(DLIMBS):
            uv=my_T[lid][d]+m_val*my_N[lid][d]+c;my_T[lid][d]=uv&0xFFFFFFFF;c=uv>>32
        carry[lid]=c
    for lid in range(LANES):
        ci=carry[(lid-1)%LANES] if lid>0 else 0
        for d in range(DLIMBS):
            uv=my_T[lid][d]+ci;my_T[lid][d]=uv&0xFFFFFFFF;ci=uv>>32
            if ci==0:break
    C31=carry[LANES-1]
    t_lo=tN+C31;tN=t_lo&0xFFFFFFFF;tN1+=t_lo>>32

    # P3: shift [T0,T1,T2,T3,T0[1],...] right by 1, last=tN
    Tf=[my_T[0][0],my_T[0][1],my_T[1][0],my_T[1][1],my_T[2][0],my_T[2][1],my_T[3][0],my_T[3][1]]
    Tf=[Tf[1],Tf[2],Tf[3],Tf[4],Tf[5],Tf[6],Tf[7],tN]
    my_T[0]=[Tf[0],Tf[1]];my_T[1]=[Tf[2],Tf[3]];my_T[2]=[Tf[4],Tf[5]];my_T[3]=[Tf[6],Tf[7]]
    tN=tN1;tN1=0

Tf=[my_T[0][0],my_T[0][1],my_T[1][0],my_T[1][1],my_T[2][0],my_T[2][1],my_T[3][0],my_T[3][1]]
print(f"After {LIMBS} iters: T={[f'{x:#x}' for x in Tf]} tN={tN:#x}")

# Cond-sub — EXACT C mirror: per-lane dlimb, ds_bpermute, final borrow check
# FIX: lane-level ds_bpermute borrow is SEPARATE from per-lane dlimb borrow
D=[[0,0],[0,0],[0,0],[0,0]]
borrow=[0]*LANES
for lid in range(LANES):
    b=0
    for d in range(DLIMBS):
        tv=my_T[lid][d];nv=my_N[lid][d];w=tv-nv-b;D[lid][d]=w&0xFFFFFFFF;b=1 if tv<nv+b else 0
    borrow[lid]=b

# Lane-level: ds_bpermute reads left neighbor's borrow, subtract ONLY from D[0]
# The left neighbor's borrow is the ORIGINAL borrow (before dlimb loop consumed it)
# lane 0 gets borrow[3] via ds_bpermute(lid-1) which reads from lane 3.
# Actually in C, lid==0 reads from lid==3 via ds_bpermute((0-1)&3,...).
# But the C code does: bi = (lid==0)?0:ds_bpermute(lid-1, borrow).
# So lane 0 gets 0.
for lid in range(LANES):
    bi = borrow[(lid-1)%LANES] if lid > 0 else 0
    if bi:
        w = D[lid][0] - bi; D[lid][0] = w & 0xFFFFFFFF
        bi_new = 1 if (w >> 63) & 1 else 0
        # Only propagate to dlimb[1] if borrow remains
        if bi_new:
            w = D[lid][1] - bi_new; D[lid][1] = w & 0xFFFFFFFF

fb = borrow[LANES-1]  # lane 3's outgoing borrow 
ns = 1 if fb == 0 else 0; mk = 0xFFFFFFFF if ns else 0
Df=[D[0][0],D[0][1],D[1][0],D[1][1],D[2][0],D[2][1],D[3][0],D[3][1]]
result=[(Df[i]&mk)|(Tf[i]&~mk) for i in range(LIMBS)]
r_val=U(result)

print(f"Borrow: {borrow} fb={fb} ns={ns} mk={mk:#x}")
print(f"Result:  {r_val:#018x}")
print(f"Expected:{expected:#018x}")
print(f"Match: {r_val==expected}")

if r_val!=expected:
    EL=L(expected)
    for i in range(LIMBS):
        if result[i]!=EL[i]:
            print(f"  diff[{i}]: out={result[i]:#010x} exp={EL[i]:#010x}")

# Also compare each iteration against standard CIOS
print("\n--- CIOS iteration comparison ---")
def cios_iter(t,i):
    a=A;b=B;n=NN
    c0=0
    for j in range(LIMBS):uv=t[j]+a[i]*b[j]+c0;t[j]=uv&0xFFFFFFFF;c0=uv>>32
    top=t[LIMBS]+c0;t[LIMBS]=top&0xFFFFFFFF;t[LIMBS+1]=top>>32
    m=(t[0]*np0)&0xFFFFFFFF;c0=0
    for j in range(LIMBS):uv=t[j]+m*n[j]+c0
    if j>0:t[j-1]=uv&0xFFFFFFFF  # BROKEN line, use proper
    pass
    return t

# Direct GMP verification
ref_t=(A_mont*B_mont)%N_int
ref_t=(ref_t*R_inv)%N_int
print(f"GMP ref: {U(L(ref_t)):#018x} match={ref_t==r_val}")
