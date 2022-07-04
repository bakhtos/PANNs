import os.path



def create_blacklist():
    """Create black list. Black list is a list of audio ids that will be 
    skipped in training. 
    """
    
    # Black list (eval split)
    eval_weak_csv = 'audioset_eval_weak_top110.tsv'
    eval_weak_csv = os.path.join('metadata', eval_weak_csv)

    file = open(eval_weak_csv, 'r')
    ids_set = set() 
    for line in file:
        line = line.split('\t')
        if line[0] == 'segment_id': continue
        ids_set.add(line[0])
    file.close()
        
    
    # Write black list
    black_list_csv = os.path.join('metadata', 'black_list.csv')
    fw = open(black_list_csv, 'w')
    
    for id in ids_set:
        fw.write('{}\n'.format(id))
    fw.close()
    

if __name__ == '__main__':
    create_blacklist()
