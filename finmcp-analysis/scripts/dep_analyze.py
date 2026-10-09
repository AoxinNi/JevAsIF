import json, sys, collections, re

def leaves(obj, key=None, out=None, parent_list=False):
    if out is None: out=[]
    if isinstance(obj, dict):
        for k,v in obj.items(): leaves(v, k, out)
    elif isinstance(obj, list):
        for v in obj: leaves(v, key, out)
    else:
        out.append((key, obj))
    return out

def parse(s):
    try: return json.loads(s)
    except Exception: return None

def norm(v): return str(v).strip()

def analyze(samples, label):
    cats=collections.Counter(); cand_sizes=[]; per_sample=[]
    turns_list=[]; calls_list=[]; resp_chars=[]
    examples=collections.defaultdict(list)
    for si,s in enumerate(samples):
        msgs=s['messages']
        question=' '.join(m.get('content') or '' for m in msgs if m['role']=='user')
        seen=[]  # list of (tool, leaves)
        sample_cats=collections.Counter(); turns=0; calls=0
        for m in msgs:
            if m['role']=='user':
                question += ' ' + (m.get('content') or '')
            if m['role']=='assistant' and m.get('tool_calls'):
                turns+=1
                for tc in m['tool_calls']:
                    calls+=1
                    args=parse(tc['function']['arguments']) or {}
                    for an,av in args.items():
                        vals = av if isinstance(av,list) else [av]
                        for v in vals:
                            if isinstance(v,(dict,)) or v is None or v=='' : continue
                            if isinstance(v,bool): cats['bool/flag']+=1; sample_cats['bool/flag']+=1; continue
                            nv=norm(v)
                            # search upstream
                            found=None
                            for tool,lv in reversed(seen):
                                keys=[k for k,x in lv if norm(x)==nv]
                                if keys:
                                    k=keys[0]
                                    sib=set(norm(x) for kk,x in lv if kk==k)
                                    found=(tool,k,len(sib)); break
                            inq = nv in question
                            if found and found[2]>1 and not inq:
                                c='selection(upstream,>1 candidates)'; cand_sizes.append(found[2])
                                if len(examples[c])<12: examples[c].append((si,tc['function']['name'],an,nv,found))
                            elif found and not inq:
                                c='direct_ref(upstream,unique)'
                                if len(examples[c])<6: examples[c].append((si,tc['function']['name'],an,nv,found))
                            elif inq:
                                c='from_user_query'
                            else:
                                c='other(const/enum/generated)'
                                if len(examples[c])<15: examples[c].append((si,tc['function']['name'],an,nv))
                            cats[c]+=1; sample_cats[c]+=1
            if m['role']=='tool':
                p=parse(m.get('content') or '')
                resp_chars.append(len(m.get('content') or ''))
                if p is not None: seen.append(('?',leaves(p)))
        per_sample.append(sample_cats); turns_list.append(turns); calls_list.append(calls)
    n=len(samples)
    print(f'===== {label}: {n} samples')
    tot=sum(cats.values())
    for c,v in cats.most_common(): print(f'  {c:38s} {v:5d}  {v/tot:6.1%}')
    has_sel=sum(1 for c in per_sample if c['selection(upstream,>1 candidates)']>0)
    has_dep=sum(1 for c in per_sample if c['selection(upstream,>1 candidates)']+c['direct_ref(upstream,unique)']>0)
    print(f'  samples with any upstream dependency: {has_dep} ({has_dep/n:.0%}); with selection-type: {has_sel} ({has_sel/n:.0%})')
    import statistics as st
    if cand_sizes: print('  candidate set size: median',st.median(cand_sizes),'max',max(cand_sizes), 'p90', sorted(cand_sizes)[int(.9*len(cand_sizes))])
    print('  LLM tool-call turns per sample: mean %.2f median %s max %d'%(st.mean(turns_list),st.median(turns_list),max(turns_list)))
    print('  tool calls per sample: mean %.2f max %d'%(st.mean(calls_list),max(calls_list)))
    print('  tool response chars: median %d mean %d p90 %d'%(st.median(resp_chars),st.mean(resp_chars),sorted(resp_chars)[int(.9*len(resp_chars))]))
    return examples

if __name__=='__main__':
    d=json.load(open(sys.argv[1]))
    ex=analyze(d, sys.argv[1])
    if len(sys.argv)>2:
        for c,l in ex.items():
            print('\n##',c)
            for e in l: print('  ',e)
