"""Async ownership, process-qualified correlations and non-additive waits."""
import sqlite3
import sys
from pathlib import Path
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'bench'))
from analyze_stage2_tree_trace import analyze,kernel_group,union_ns


class TreeTraceTest(unittest.TestCase):
    def test_async_and_process_ownership(self):
        c=sqlite3.connect(':memory:')
        c.executescript('''
          create table StringIds(id integer,value text);
          insert into StringIds values(1,'cudaLaunchKernel'),(2,'cudaEventSynchronize'),
            (3,'cudaMemcpy'),(4,'void tile_kernel<false>()'),(5,'void s4_warp_reduce_kernel<256,true>()');
          create table NVTX_EVENTS(start integer,end integer,text text,textId integer,globalTid integer);
          create table CUPTI_ACTIVITY_KIND_RUNTIME(start integer,end integer,nameId integer,globalTid integer,correlationId integer);
          create table CUPTI_ACTIVITY_KIND_KERNEL(start integer,end integer,globalPid integer,correlationId integer,deviceId integer,demangledName integer);
          create table CUPTI_ACTIVITY_KIND_MEMCPY(start integer,end integer,globalPid integer,correlationId integer,deviceId integer);
        ''')
        pid=7<<24;tid=pid+1;other=8<<24
        c.executemany('insert into NVTX_EVENTS values(?,?,?,?,?)',[(0,100,'Tree.host_driver',None,tid),(110,200,'GTree.build',None,tid),
            (9,15,'NTT.shape 1048576 4',None,tid)])
        c.executemany('insert into CUPTI_ACTIVITY_KIND_RUNTIME values(?,?,?,?,?)',
            [(10,12,1,tid,11),(120,121,1,other+1,11),(130,132,1,tid,12),
             (150,180,2,tid,14),(170,195,3,tid,13)])
        c.executemany('insert into CUPTI_ACTIVITY_KIND_KERNEL values(?,?,?,?,?,?)',
            [(120,150,pid,11,1,4),(175,180,other,11,1,4),(210,250,pid,12,1,5)])
        c.execute('insert into CUPTI_ACTIVITY_KIND_MEMCPY values(177,189,?,?,1)',(pid,13))
        r=analyze(c,1);f=r['trees']['Tree.host_driver'];g=r['trees']['GTree.build']
        self.assertAlmostEqual(f['gpu_seconds']['NTT_transform_pointwise'],30e-9)
        self.assertEqual(len(r['ntt_shapes']),1)
        shape=r['ntt_shapes'][0]
        self.assertEqual((shape['phase'],shape['length'],shape['batch'],shape['kernel']),('Tree.host_driver',1048576,4,'tile_kernel'))
        self.assertAlmostEqual(shape['seconds'],30e-9)
        self.assertNotIn('NTT_transform_pointwise',g['gpu_seconds'])
        self.assertAlmostEqual(g['gpu_seconds']['S4'],40e-9)
        self.assertAlmostEqual(g['gpu_seconds']['Memcpy'],12e-9)
        self.assertAlmostEqual(g['synchronous_api_seconds'],45e-9)
        self.assertAlmostEqual(g['host_outside_cuda_api_seconds'],43e-9)
        self.assertEqual(r['outside_tree_activity']['CUPTI_ACTIVITY_KIND_KERNEL'],1)
        self.assertFalse(r['unmatched_activity']);c.close()

    def test_union_and_kernel_scope(self):
        self.assertEqual(union_ns([(10,20),(15,30),(40,50),(50,50)]),30)
        self.assertEqual(kernel_group('void outer_coop_kernel<8,false>()'),'NTT_transform_pointwise')
        self.assertEqual(kernel_group('void build_tile_table_kernel()'),'Other_kernel')
        self.assertEqual(kernel_group('void carry_extract_kernel()'),'Other_kernel')


if __name__=='__main__':unittest.main()
