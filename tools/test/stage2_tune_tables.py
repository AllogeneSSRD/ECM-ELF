"""Read portable tune matrices for assertions and analysis."""
def summaries(document, kind='sample'):
    if document.get('profile',{}).get('format') not in range(8,15):
        raise ValueError('This reader requires portable profile format 8–14')
    result=[]
    tables=document.get('width' if kind=='sample' else 'ntt',{})
    for key, table in ([(None,tables)] if 'matrix' in tables else tables.items()):
        columns,*rows=table['matrix']
        assert len(columns)==len(set(columns))
        for values in rows:
            row={k:v for k,v in zip(columns,values) if v!=''}
            assert len(values)==len(columns)
            if kind=='sample':
                row['target_bits']=int(key)
                row['arithmetic_bits']=row.get('carrier_exponent') or int(key)
                row['execution_path']='resident' if row['fold_resident'] and row['frontier_resident'] else 'host_frontier' if row['fold_resident'] else 'host_fold_frontier'
            else:
                if key is not None:row['condition']=int(key)
                if row.get('ntt_type','stage2_convolution')=='stage2_convolution' and 'mean_seconds' not in row:
                    row['mean_seconds']=row['batch']/row['iter_per_s']
                    row['batch_iter_per_s']=1/row['mean_seconds']
            if 'program_version' not in row and document['condition'][str(row['condition'])].get('program_version'):
                row['program_version']=document['condition'][str(row['condition'])]['program_version']
            result.append(row)
    return result
