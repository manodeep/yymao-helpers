__all__ = ['findLagrangianVolume']
import numpy as np
from readGadgetSnapshot import readGadgetSnapshot
from findSpheresInSnapshots import getParticlesWithinSphere
from mvee import mvee

def findLagrangianVolume(c, r, snapshot_prefix, ic_prefix, snapshot_edges=None,
        id_int64=None, rec_only=False):
    ids_all = getParticlesWithinSphere(c, r, snapshot_prefix, snapshot_edges, \
            output_dtype=np.dtype([('id', np.uint64)]))['id']
    id_min = ids_all.min()
    id_max = ids_all.max()
    
    header = readGadgetSnapshot(ic_prefix+'.0')
    ic_subregion_count = header.num_files
    L = header.BoxSize
    
    #for Gadget, particle id starts at 1
    current_ic_id_start = 1
    current_ic_id_end = 1
    total_count = 0
    for x in xrange(ic_subregion_count):
        ic_snapshot_file = '%s.%d'%(ic_prefix, x)
        header = readGadgetSnapshot(ic_snapshot_file)
        current_ic_id_start = current_ic_id_end
        current_ic_id_end += sum(header.npart)
        if(id_max < current_ic_id_start or id_min >= current_ic_id_end): 
            continue

        find = ids_all[(ids_all >= current_ic_id_start) & \
                (ids_all < current_ic_id_end)]
        ic_ids = np.arange(current_ic_id_start, current_ic_id_end, \
                dtype=np.uint64)
        find = np.searchsorted(ic_ids, ids_all)
        find[find>=len(ic_ids)] = -1
        find = find[ic_ids[find]==ids_all]
        if(len(find)==0): continue

        header, ic_pos = readGadgetSnapshot(ic_snapshot_file, read_pos=True)
        pos_selected = ic_pos[find]
        if(total_count == 0):
            pos_all = np.zeros((len(pos_selected), 3), np.float32)
        else:
            pos_all.resize((total_count+len(pos_selected), 3))         
        pos_all[total_count:] = pos_selected
        total_count += len(pos_selected)
    
    if total_count != len(ids_all):
        raise ValueError("Something went wrong!")
    
    for pos in pos_all.T:
        p = np.sort(pos)
        gaps = np.ediff1d(p)
        j = np.argmax(gaps)
        max_gap = gaps[j]
        gap_start = p[j]
        pos_range = p[-1] - p[0]
        if L - max_gap < pos_range:
            pos[pos <= gap_start] += L
    
    pos_min = pos_all.min(axis=0)
    pos_max = pos_all.max(axis=0)
    if rec_only: return pos_min, pos_max-pos_min

    A, c = mvee(pos_all)
    return pos_min, pos_max-pos_min, c, A

