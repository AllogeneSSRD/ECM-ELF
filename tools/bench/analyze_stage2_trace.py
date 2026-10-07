"""Read-only Nsight trace audit: allocation lifetimes and correlated API waits.

Tracked cudaMalloc payload and CUDA pinned RAM are separate from process/driver
memory. API duration, GPU execution and DMA duration are overlapping measures.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sqlite3


def event_returns(c, device):
    """Match event recordings, not just reusable event handles.

    A completed event may intentionally be consumed much later. Only time
    inside the successful sync API after completion is a return-delay probe.
    Synchronization's deviceId is not reliable on this Windows exporter: the
    completion event and the already-audited kernel device identify the GPU.
    """
    tables={r[0] for r in c.execute('select name from sqlite_master where type="table"')}
    needed={'CUPTI_ACTIVITY_KIND_CUDA_EVENT','CUPTI_ACTIVITY_KIND_SYNCHRONIZATION'}
    if not needed<=tables:return dict(available=False,reason='No CUDA completion/synchronization tables')
    completed={}
    for row in c.execute('select * from CUPTI_ACTIVITY_KIND_CUDA_EVENT where deviceId=?',(device,)):
        key=tuple(row[k] for k in ('globalPid','contextId','eventId','eventSyncId'))
        if key in completed:raise ValueError('Ambiguous re-recorded CUDA event: '+str(key))
        completed[key]=dict(row)
    if not completed:return dict(available=False,reason='No recorded CUDA completion events for device')
    records=[];unmatched=[]
    sql='''select r.*,s.value name from CUPTI_ACTIVITY_KIND_RUNTIME r
           join StringIds s on s.id=r.nameId
           where s.value like 'cudaEventSynchronize%' and r.returnValue=0'''
    for row in c.execute(sql):
        pid=row['globalTid']&~((1<<24)-1)
        sync=list(c.execute('select * from CUPTI_ACTIVITY_KIND_SYNCHRONIZATION where globalPid=? and correlationId=? and syncType=1',(pid,row['correlationId'])))
        if len(sync)>1:raise ValueError('Ambiguous event synchronization correlation')
        if not sync:
            unmatched.append(dict(correlationId=row['correlationId'],reason='No synchronization'));continue
        wait=sync[0];key=tuple(wait[k] for k in ('globalPid','contextId','eventId','eventSyncId'))
        event=completed.get(key)
        if event is None:
            unmatched.append(dict(correlationId=row['correlationId'],reason='No exact recording completion'));continue
        timestamp=event['timestamp'];start,end=row['start'],row['end']
        records.append(dict(globalPid=pid,globalTid=row['globalTid'],contextId=wait['contextId'],
            eventId=wait['eventId'],eventSyncId=wait['eventSyncId'],device=event['deviceId'],
            reported_sync_device=wait['deviceId'],streamId=event['streamId'],
            correlationId=row['correlationId'],record_correlationId=event['correlationId'],
            api_start_ns=start,api_end_ns=end,completion_ns=timestamp,
            api_seconds=(end-start)/1e9,
            completion_before_api_seconds=max(0,start-timestamp)/1e9,
            api_before_completion_seconds=max(0,timestamp-start)/1e9,
            in_api_after_completion_seconds=(end-max(start,timestamp))/1e9))
    return dict(available=True,completion_records=len(completed),matched_sync_apis=len(records),
        unmatched_sync_apis=unmatched,
        max_in_api_after_completion_seconds=max((r['in_api_after_completion_seconds'] for r in records),default=None),
        max_api_seconds=max((r['api_seconds'] for r in records),default=None),
        records=sorted(records,key=lambda r:r['in_api_after_completion_seconds'],reverse=True),
        scope='Exact process/context/event/recording match; clocks have tracing precision. Completion before API entry is intentional/CPU work, not API return latency. Only captured event-sync APIs are covered.')


def memory_peak(c,kind,device):
    # Pinned host events have no useful physical-device attribution.
    where='memKind=?'+(' and deviceId=?' if kind==2 else '')
    args=(kind,device) if kind==2 else (kind,)
    live={};total=peak=0;peak_time=None;allocations=deallocations=0;snapshot=[]
    for r in c.execute('select * from CUDA_GPU_MEMORY_USAGE_EVENTS where '+where+' order by start',args):
        key=(r['globalPid'],r['contextId'],r['address'])
        if r['memoryOperationType']==0:
            if key in live:raise ValueError('Overlapping allocation at the same address')
            live[key]=dict(r);total+=r['bytes'];allocations+=1
        elif r['memoryOperationType']==1:
            old=live.pop(key,None)
            if old is None or old['bytes']!=r['bytes']:raise ValueError('Unmatched free in allocation trace')
            total-=r['bytes'];deallocations+=1
        else:raise ValueError('Unknown memory operation')
        if total>peak:peak=total;peak_time=r['start'];snapshot=list(live.values())
    return dict(peak_bytes=peak,peak_time_ns=peak_time,end_live_bytes=total,
                allocations=allocations,deallocations=deallocations,
                peak_allocations=sorted(snapshot,key=lambda r:r['bytes'],reverse=True))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--trace',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--device',type=int,default=1);p.add_argument('--long-api-ms',type=float,default=5)
    a=p.parse_args()
    if a.output.resolve()==a.trace.resolve() or a.device!=1 or a.long_api_ms<=0:raise ValueError('Use GPU1 and a separate output')
    raw_sha=hashlib.sha256(a.trace.read_bytes()).hexdigest()
    c=sqlite3.connect('file:'+a.trace.resolve().as_posix()+'?mode=ro',uri=True);c.row_factory=sqlite3.Row
    c.text_factory=lambda b:b.decode('utf-8','replace')
    devices=[dict(r) for r in c.execute('select deviceId,count(*) count from CUPTI_ACTIVITY_KIND_KERNEL group by deviceId')]
    if len(devices)!=1 or devices[0]['deviceId']!=a.device:raise ValueError('Unexpected GPU activity')
    events=[]
    for table in ('CUPTI_ACTIVITY_KIND_KERNEL','CUPTI_ACTIVITY_KIND_MEMCPY','CUPTI_ACTIVITY_KIND_MEMSET'):
        events.extend((r[0],r[1]) for r in c.execute('select start,end from '+table+' where deviceId=?',(a.device,)))
    merged=[]
    for left,right in sorted(events):
        if not merged or left>merged[-1][1]:merged.append([left,right])
        else:merged[-1][1]=max(right,merged[-1][1])
    if not merged:raise ValueError('Empty activity window')
    gaps=sorted(((merged[i-1][1],merged[i][0]) for i in range(1,len(merged))),key=lambda g:g[1]-g[0],reverse=True)[:12]
    gap_rows=[]
    for left,right in gaps:
        # Include APIs crossing gap boundaries; contained-only queries miss waits.
        api=[dict(r) for r in c.execute('select s.value name,r.*,(r.end-r.start)/1e9 seconds from CUPTI_ACTIVITY_KIND_RUNTIME r join StringIds s on s.id=r.nameId where r.start<? and r.end>? order by r.start',(right,left))]
        gap_rows.append(dict(start_ns=left,end_ns=right,seconds=(right-left)/1e9,overlapping_runtime=api))
    long=[]
    for r in c.execute('select s.value name,r.*,(r.end-r.start)/1e9 seconds from CUPTI_ACTIVITY_KIND_RUNTIME r join StringIds s on s.id=r.nameId where r.end-r.start>=? order by r.end-r.start desc',(int(a.long_api_ms*1e6),)):
        row=dict(r)
        copies=[dict(m) for m in c.execute('select * from CUPTI_ACTIVITY_KIND_MEMCPY where correlationId=? and globalPid=? and deviceId=?',(r['correlationId'],r['globalTid']&~((1<<24)-1),a.device))]
        row['correlated_copies']=copies
        if copies:
            row['first_copy_delay_seconds']=(min(m['start'] for m in copies)-r['start'])/1e9
            row['return_after_last_copy_seconds']=(r['end']-max(m['end'] for m in copies))/1e9
            row['dma_seconds']=sum(m['end']-m['start'] for m in copies)/1e9
        long.append(row)
    span=merged[-1][1]-merged[0][0];union=sum(r-l for l,r in merged)
    result=dict(schema=1,trace=str(a.trace.resolve()),trace_sha256=raw_sha,tool_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        device=a.device,devices=devices,device_allocations=memory_peak(c,2,a.device),pinned_host_allocations=memory_peak(c,1,a.device),
        window=dict(span_seconds=span/1e9,own_activity_union_seconds=union/1e9,no_own_activity_seconds=(span-union)/1e9),
        largest_gaps=gap_rows,long_apis=long,
        event_completion_returns=event_returns(c,a.device),
        scope='Own process tree. Device payload excludes static/module/driver/context allocations; pinned RAM is not process private RAM. End-live pinned allocations can be reclaimed at process exit. No own GPU event is not proof the whole card is idle; correlated DMA delays include queued earlier work.')
    c.close();a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps({k:result[k] for k in ('window',)}|{'device_payload_peak_bytes':result['device_allocations']['peak_bytes'],
        'pinned_host_peak_bytes':result['pinned_host_allocations']['peak_bytes'],'long_apis':len(long)}))


if __name__=='__main__':main()
