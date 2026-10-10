"""Read format-6 final parameter tables for test assertions (not a cost model)."""
def summaries(document, kind='sample'):
    result=[]
    group='width' if kind=='sample' else 'ntt'
    for node in document.get(group,{}).values():
        for table in node.get('table',{}).values():
            common={k:v for k,v in table.items() if k not in ('columns','rows')}
            for row in table['rows']:
                assert len(row)==len(table['columns'])
                result.append(dict(common,**dict(zip(table['columns'],row))))
    return result
