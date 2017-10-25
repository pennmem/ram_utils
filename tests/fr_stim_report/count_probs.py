def prob_counter(probs,events,fname,thresh):
    with open(fname,'w') as _file:
        print>>_file, 'Word #\tProb\tPre-stim\Below Thresh'
        k=11
        for i,event in enumerate(events):
            if event.type=='WORD':
                k+=1
                print>>_file, '{0}\t{1:.3}\t{2}\t{3}'.format(k-12,probs[k],events[i+1].type=='STIM_ON',probs[k]<thresh)
