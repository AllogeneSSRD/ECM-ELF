"""Read format-7 final matrices for test assertions (not a cost model)."""
def summaries(document, kind='sample'):
    result=[]
    group='width' if kind=='sample' else 'ntt'
    for node in document.get(group,{}).values():
        for table in node.get('table',{}).values():
            common={k:v for k,v in table.items() if k != 'matrix'}
            columns,*rows=table['matrix']
            assert len(columns)==len(set(columns)) and all(isinstance(k,str) for k in columns)
            for row in rows:
                assert len(row)==len(columns)
                assert not common.keys() & set(columns)
                result.append(dict(common,**dict(zip(columns,row))))
    return result
