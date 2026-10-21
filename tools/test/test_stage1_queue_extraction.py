"""Check queue ownership, input normalization and the complete completion protocol."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
from test_result_publication_extraction import body
from test_stage1_host_modules import normalize

ROOT=Path(__file__).resolve().parents[2]


def masked(text):
    return re.sub(r'"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\'|/\*.*?\*/|//[^\n]*',lambda m:' '*len(m[0]),text,flags=re.S)


def end(text,at,left,right):
    depth=0
    for i in range(at,len(text)):
        depth+=(text[i]==left)-(text[i]==right)
        if not depth:return i
    raise AssertionError((left,at))


def raw_body(text,name):
    m=masked(text)
    for hit in re.finditer(r'\b'+name+r'\s*\(',m):
        at=m.index('(',hit.start());i=end(m,at,'(',')')+1
        while m[i].isspace():i+=1
        if m[i]=='{':return text[i+1:end(m,i,'{','}')]
    raise AssertionError(name)


def arguments(text,at):
    m=masked(text);start=m.index('(',at);last=end(m,start,'(',')');parts=[];begin=start+1;stack=[]
    for i in range(start+1,last):
        if m[i] in '([{':stack.append(m[i])
        elif m[i] in ')]}':stack.pop()
        elif m[i]==',' and not stack:parts.append(text[begin:i].strip());begin=i+1
    parts.append(text[begin:last].strip());return parts,last


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    previous=subprocess.check_output(['git','show','6fad9f6:src/ecm/application/cli/ecm_driver.cpp'],cwd=ROOT).decode('utf-8-sig')
    read=lambda name:(ROOT/name).read_text(encoding='utf-8-sig')
    driver=read('src/ecm/application/cli/ecm_driver.cpp');source=read('src/ecm/application/cli/queue_source.cpp');workflow=read('src/ecm/service/work/queue_workflow.cpp');inputs=read('src/ecm/application/cli/stage1_input.cpp')
    input_names=['is_absolute_path_local','normalize_path_local','resolve_rel_local','get_exe_dir_local','stage1_save_construction','stage1_selection_error','stage1_task_error','parse_affinity_spec','stage1_result_settings']
    for name in input_names:assert body(previous,name)==body(inputs,name),name
    for name in ['current_epoch_seconds','build_n_expr','p95_dir_of','p95_join_path','edwards_stage2_line','stage2_save_record_count']:assert body(previous,name)==body(source,name),name
    assert body(previous,'p95_quote')==body(read('src/ecm/storage/transfer/p95_transfer.cpp'),'p95_transfer_quote')
    one=raw_body(previous,'queue_run_one');expected=one[one.index('    auto result=run_stage1_task'):]
    expected=expected.replace('},stage1_tasks);','},tasks);').replace('p95_quote(', 'p95_transfer_quote(')
    current=raw_body(workflow,'run_one');current=current[current.index('    auto result=run_stage1_task'):]
    assert normalize(expected)==normalize(current),'completion/failure/factor/park/finished/source/sync/flush/delivery ordering changed'
    manager=raw_body(previous,'run_queue_manager');app=raw_body(source,'run_queue_manager')
    startup=manager[:manager.index('    int processed = 0;')].replace('p95_quote(', 'p95_transfer_quote(')
    actual=app[:app.index('    bool saw_task=false;')]
    assert normalize(startup)==normalize(actual),'configuration/startup/lazy Prime95 protocol changed'
    loop_at=manager.index('while (true)');m=masked(manager);brace=m.index('{',loop_at)
    old_parse=manager[manager.index('        std::string line;',brace):end(m,brace,'{','}')]
    at=source.index('const auto next_task=');m=masked(source);brace=m.index('{',at);new_parse=source[brace+1:end(m,brace,'{','}')]
    # Check every moved call input, then compare the remaining source parser in full.
    assignments=[('b1',1),('b2',2),('curves',3),('save_file',6),('expression',7),('aid',8),('ticket',9),('options',13),('marker',18),('p95',20),('published_line',21),('stage2',22),('stage2_line',23),('auto_stage2_line',24)]
    for k in range(2):
        at=old_parse.index('const bool cont = queue_run_one(');args,last=arguments(old_parse,at);assert len(args)==25
        tail=old_parse.index('if (!cont)',last);m=masked(old_parse);brace=m.index('{',tail);finish=end(m,brace,'{','}')+1
        if k==0:finish+=re.match(r'\s*continue;',old_parse[finish:]).end()
        old_parse=old_parse[:at]+'READY'+str(k)+';'+old_parse[finish:]
        at=new_parse.index('item.b1=');finalize=new_parse.index('return finalize_queue_input(',at);values,finish=arguments(new_parse,finalize)
        for field,index in assignments:
            assert normalize('item.'+field+'='+args[index]+';') in normalize(new_parse[at:finalize]),(k,field)
        assert list(map(normalize,values))==['item',normalize(args[4]),normalize(args[5])]
        assert list(map(normalize,args[10:13]))==['worktodo_path','finished_path','worker']
        assert list(map(normalize,args[14:20]))==['exe_dir','sync1','sync2','full_sync','marker','&processed']
        assert new_parse[finish+1]==';'
        new_parse=new_parse[:at]+'READY'+str(k)+';'+new_parse[finish+2:]
    old_parse=normalize(old_parse);new_parse=normalize(new_parse)
    old_parse=old_parse.replace('if(logf){ecm_log_set_mirror(nullptr);fclose(logf);}','')
    old_parse=old_parse.replace('mpz_tN;mpz_init(N);','').replace('mpz_clear(N);','').replace('return1;','returnQueueReadStatus::Error;')
    old_parse=old_parse.replace('if(!saw_task)p95_for_task(opt);break;','if(!saw_task)p95_for_task(opt);returnQueueReadStatus::Empty;')
    old_parse=old_parse.replace('if(!ecm_worktodo_advance(worktodo_path,worker,line,WorktodoAction::MarkError))break;continue;','returnQueueReadStatus::MarkError;')
    new_parse=new_parse.replace('mpz_ptrN=item.modulus;','').replace('saw_task=true;item.ticket=line;','saw_task=true;')
    (a.output/'normalized_before.txt').write_text(old_parse,encoding='utf8')
    (a.output/'normalized_after.txt').write_text(new_parse,encoding='utf8')
    assert old_parse==new_parse,'source parsing/typed-save/skip/lazy-handoff order changed'
    assert 'run_stage1_task' not in source and 'ecm_worktodo_advance(' not in source
    assert 'application/' not in workflow and 'ecm_queue_config' not in workflow
    assert workflow.count('run_stage1_task(')==1 and 'run_queue_manager' not in workflow
    assert 'queue_run_one' not in driver and 'ecm_worktodo_read_next_line' not in driver and 'while (true)' not in driver
    assert normalize('QueueItem::QueueItem(){mpz_init(modulus);}') in normalize(workflow)
    assert normalize('QueueItem::~QueueItem(){mpz_clear(modulus);}') in normalize(workflow)
    assert 'run_queue_manager(ini_path, worker_index,stage1_tasks)' in driver
    result=dict(complete=True,baseline='6fad9f6',input_functions_unchanged=9,source_helpers_unchanged=7,source_formats=2,source_parser_unchanged=True,startup_and_prime95_lazy_order_unchanged=True,complete_protocol_sha256=hashlib.sha256(normalize(current).encode()).hexdigest(),complete_protocol_unchanged=True,queue_item_owns_modulus=True,public_queue_runner_classes=0,workflow_application_includes=0,driver_lines=len(driver.splitlines()),cuda_runs=0)
    (a.output/'result.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf8');print(json.dumps(result))


if __name__=='__main__':main()
