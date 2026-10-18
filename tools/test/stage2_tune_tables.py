"""Read format-8 mean-time matrices for assertions and analysis."""
def summaries(document, kind='sample'):
    if document.get('profile',{}).get('format')!=8:
        raise ValueError('This reader requires a format-8 mean-time profile')
    result=[]
    for key, table in document.get('width' if kind=='sample' else 'ntt',{}).items():
        columns,*rows=table['matrix']
        assert len(columns)==len(set(columns))
        for values in rows:
            row={k:v for k,v in zip(columns,values) if v!=''}
            assert len(values)==len(columns)
            if kind=='sample':
                row['target_bits']=int(key)
                row['arithmetic_bits']=row.get('carrier_exponent') or int(key)
                row['execution_path']='resident' if row['fold_resident'] and row['frontier_resident'] else 'host_frontier' if row['fold_resident'] else 'host_fold_frontier'
            else:row['condition']=int(key)
            row['program_version']=document['condition'][str(row['condition'])]['program_version']
            result.append(row)
    return result
