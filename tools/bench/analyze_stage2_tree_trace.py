"""Attribute GPU work to its CPU tree submission range, even across async edges.

CPU range wall, API residence and GPU duration sums overlap. None are additive
across CPU/GPU. Correlation is process-qualified; GPU start time is not ownership.
"""
import argparse
import bisect
import collections
import json
from pathlib import Path
import re
import sqlite3

from bench_stage2_point_routes import sha

TREE_NAMES={'Tree.host_driver','GTree.build'}
NTT=re.compile(r'\b(?:tile_kernel|outer_fwd_kernel|outer_inv_kernel|outer_coop_kernel|pointwise_kernel|pointwise_twiddle_kernel|dit_stage_kernel|ntt_stage_kernel|bitrev_swap_kernel|scale_kernel)\b')
S4=re.compile(r'\bs4_(?:warp_)?reduce_kernel\b')
SYNCHRONOUS=re.compile(r'^(?:cuda(?:Event|Stream|Device)Synchronize|cudaMemcpy(?:_v\d+)?$)')


def union_ns(intervals):
    end=None;total=0
    for left,right in sorted(intervals):
        if right<left:raise ValueError('Negative interval')
        if end is None or left>end:total+=right-left;end=right
        elif right>end:total+=right-end;end=right
    return total


def kernel_group(name):
    if S4.search(name):return 'S4'
    if NTT.search(name):return 'NTT_transform_pointwise'
    return 'Other_kernel'


def analyze(connection,device):
    connection.row_factory=sqlite3.Row
    connection.text_factory=lambda b:b.decode('utf-8','replace')
    names=dict(connection.execute('select id,value from StringIds'))
    ranges=collections.defaultdict(list)
    for raw in connection.execute('select * from NVTX_EVENTS where end is not null'):
        r=dict(raw);name=r.get('text') or names.get(r.get('textId'),'')
        if name in TREE_NAMES and r['end']>=r['start']:
            r['name']=name;ranges[r['globalTid']].append(r)
    if not ranges:raise ValueError('No instrumented CPU tree ranges')
    for values in ranges.values():values.sort(key=lambda r:r['start'])
    starts={tid:[r['start'] for r in v] for tid,v in ranges.items()}
    def owners(tid,time):
        values=ranges.get(tid,());begin=bisect.bisect_right(starts.get(tid,()),time)-1
        # Tree ranges are few; walking backwards also handles nested host trees.
        found={}
        for r in reversed(values[:begin+1]):
            if r['start']<=time<r['end']:found.setdefault(r['name'],r)
        return list(found.values())
    result={name:dict(host_range_seconds=0,host_api_seconds=0,synchronous_api_seconds=0,
        host_outside_cuda_api_seconds=0,host_range_count=0,kernel_counts=collections.Counter(),
        gpu_seconds=collections.Counter(),gpu_active_union_seconds=0,api_by_name={},kernel_by_name={}) for name in TREE_NAMES}
    host_spans=collections.defaultdict(list);api_spans=collections.defaultdict(list);sync_spans=collections.defaultdict(list)
    api_duration=collections.defaultdict(collections.Counter);kernel_duration=collections.defaultdict(collections.Counter)
    gpu_spans=collections.defaultdict(list);api={}
    for tid,values in ranges.items():
        for r in values:
            host_spans[r['name'],tid].append((r['start'],r['end']));result[r['name']]['host_range_count']+=1
    for row in connection.execute('select * from CUPTI_ACTIVITY_KIND_RUNTIME order by start'):
        r=dict(row);pid=r['globalTid']&~((1<<24)-1);key=(pid,r['correlationId'])
        # Exporters can include an outer runtime entry and an inner one with the
        # same correlation. Ambiguity must not silently reassign a kernel.
        if key in api and api[key]['globalTid']!=r['globalTid']:raise ValueError('Ambiguous API process/correlation')
        r['trees']=owners(r['globalTid'],r['start']);api[key]=r
        for tree in r['trees']:
            name=tree['name'];interval=(r['start'],min(r['end'],tree['end']))
            if interval[1]<interval[0]:raise ValueError('Invalid API interval')
            api_spans[name,r['globalTid']].append(interval)
            label=names.get(r['nameId'],'unknown');api_duration[name][label]+=interval[1]-interval[0]
            if SYNCHRONOUS.search(label):sync_spans[name,r['globalTid']].append(interval)
    unmatched=collections.Counter();outside=collections.Counter()
    tables={r[0] for r in connection.execute('select name from sqlite_master where type="table"')}
    for table in ('CUPTI_ACTIVITY_KIND_KERNEL','CUPTI_ACTIVITY_KIND_MEMCPY','CUPTI_ACTIVITY_KIND_MEMSET'):
        if table not in tables:continue
        for raw in connection.execute('select * from '+table+' where deviceId=?',(device,)):
            r=dict(raw);launch=api.get((r['globalPid'],r['correlationId']))
            if launch is None:unmatched[table]+=1;continue
            if not launch['trees']:outside[table]+=1;continue
            for tree in launch['trees']:
                name=tree['name'];duration=r['end']-r['start'];gpu_spans[name].append((r['start'],r['end']))
                if table.endswith('KERNEL'):
                    label=names.get(r['demangledName'],'unknown');group=kernel_group(label)
                    result[name]['kernel_counts'][group]+=1;kernel_duration[name][label]+=duration
                else:group='Memcpy' if table.endswith('MEMCPY') else 'Memset'
                result[name]['gpu_seconds'][group]+=duration/1e9
    for name,row in result.items():
        for (label,tid),spans in host_spans.items():
            if label==name:
                row['host_range_seconds']+=union_ns(spans)/1e9
                row['host_api_seconds']+=union_ns(api_spans[label,tid])/1e9
                row['synchronous_api_seconds']+=union_ns(sync_spans[label,tid])/1e9
        row['host_outside_cuda_api_seconds']=row['host_range_seconds']-row['host_api_seconds']
        row['gpu_active_union_seconds']=union_ns(gpu_spans[name])/1e9
        row['api_by_name']={k:v/1e9 for k,v in api_duration[name].most_common()}
        row['kernel_by_name']={k:v/1e9 for k,v in kernel_duration[name].most_common()}
    return dict(trees=result,unmatched_activity=dict(unmatched),outside_tree_activity=dict(outside),
        scope='CPU tree scope; GPU work belongs to the process-qualified launch API, not execution timestamp. '
              'NTT includes transform and pointwise kernels, excludes carry/pack/table construction. '
              'API residence includes CPU/driver work and waits; synchronous API residence is not pure GPU wait. '
              'Host outside CUDA APIs includes preparation, scheduling and possible OS waits, not measured CPU cycles. '
              'CPU/GPU sums overlap; nested range totals must not be added.')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--trace',type=Path,required=True)
    p.add_argument('--device',type=int,default=1);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    with sqlite3.connect(a.trace.resolve().as_uri()+'?mode=ro',uri=True) as c:result=analyze(c,a.device)
    result.update(trace=str(a.trace.resolve()),trace_sha256=sha(a.trace),analyzer_sha256=sha(__file__),device=a.device)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open('x',encoding='utf-8') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps({name:{k:v for k,v in row.items() if k not in ('api_by_name','kernel_by_name')}
                      for name,row in result['trees'].items()},ensure_ascii=False))


if __name__=='__main__':main()
