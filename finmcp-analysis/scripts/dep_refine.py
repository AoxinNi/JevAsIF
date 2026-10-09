import json, sys, collections, statistics as st
from dep_analyze import leaves, parse, norm
d=json.load(open(sys.argv[1]))
ev=collections.Counter(); sizes=collections.defaultdict(list); samp=collections.defaultdict(set)
ex=collections.defaultdict(list)
for si,s in enumerate(d):
    msgs=s['messages']; question=' '.join(m.get('content') or '' for m in msgs if m['role']=='user'); seen=[]
    for m in msgs:
        if m['role']=='assistant' and m.get('tool_calls'):
            for tc in m['tool_calls']:
                args=parse(tc['function']['arguments']) or {}
                for an,av in args.items():
                    vals=av if isinstance(av,list) else [av]
                    vals=[norm(v) for v in vals if not isinstance(v,(dict,bool)) and v not in (None,'')]
                    # group values by the upstream key/response they came from
                    groups=collections.defaultdict(list)
                    for v in vals:
                        if v in question: continue
                        for ri,lv in reversed(list(enumerate(seen))):
                            keys=[k for k,x in lv if norm(x)==v]
                            if keys:
                                k=keys[0]; sib=set(norm(x) for kk,x in lv if kk==k)
                                if len(sib)>1: groups[(ri,k)].append((v,len(sib)))
                                break
                    for (ri,k),g in groups.items():
                        n=g[0][1]; kk=len(set(v for v,_ in g))
                        c='pick 1 of n' if kk==1 else ('pick all n' if kk>=n else 'pick subset k of n')
                        ev[c]+=1; sizes[c].append(n); samp[c].add(si)
                        if len(ex[c])<8: ex[c].append((si, s['question'][:40].replace('\n',' '), tc['function']['name'], an, f'{kk}/{n}', k))
        if m['role']=='tool':
            p=parse(m.get('content') or '')
            if p is not None: seen.append(leaves(p))
n=len(d); print('samples',n)
for c,v in ev.most_common():
    print(f'  {c:20s} events {v:5d} | samples {len(samp[c]):4d} ({len(samp[c])/n:.0%}) | n median {st.median(sizes[c])} p90 {sorted(sizes[c])[int(.9*len(sizes[c]))]} max {max(sizes[c])}')
judg=samp['pick 1 of n']|samp['pick subset k of n']
print(f'  samples needing a real judgment (pick 1 or subset): {len(judg)} ({len(judg)/n:.0%})')
for c,l in ex.items():
    print('\n##',c)
    for e in l: print('  ',e)
