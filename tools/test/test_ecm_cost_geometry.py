"""Exact product-tree schedule and G=1 feature regression fixtures."""
from pathlib import Path
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'bench'))
import ecm_cost_model as model
from calibrate_stage2_d import phi,inverse,unit


def enumerate_tree(n):
    if n<2:return (0,0,0,0)
    pad=1<<(n-1).bit_length();front=[1]*n+[0]*(pad-n)
    pairs=groups=copies=coeffs=0
    while len(front)>1:
        shapes=set();next=[]
        for a,b in zip(front[::2],front[1::2]):
            next.append(a+b)
            if a and b:pairs+=1;shapes.add(tuple(sorted((a+1,b+1))))
            elif a or b:copies+=1;coeffs+=a+b+1
        groups+=len(shapes);front=next
    return pairs,groups,copies,coeffs


class GeometryTests(unittest.TestCase):
    def test_exact_schedule(self):
        for n in [*range(259),2880,5759,5760,5761,11520]:
            f=model.tree_stats(n,2203)
            self.assertEqual((f['pairs'],f['groups'],f['copies'],f['copy_coeffs']),enumerate_tree(n),n)
            self.assertEqual(f['pairs'],max(0,n-1))
    def test_g1_inverse_moves_into_descent(self):
        d=120120;p=phi(d)//2
        f=model.features(d,d*(p//2-2),4423)
        self.assertEqual(f['G'],1);self.assertEqual(f['inverse'],0);self.assertEqual(f['fold'],0)
        self.assertEqual(f['local_inverse'],inverse(p,4423));self.assertEqual(f['root_reduction'],0)
        boundary=model.features(d,d*(p-2),4423)
        self.assertEqual(boundary['root_reduction'],unit(1,4423)+unit(p+1,4423))
        multiple=model.features(d,d*(p-1),4423)
        self.assertEqual(multiple['G'],2);self.assertEqual(multiple['local_inverse'],0)
        self.assertEqual(multiple['inverse'],inverse(p+1,4423))


if __name__=='__main__':unittest.main()
