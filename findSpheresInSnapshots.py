__all__ = ['getSnapshotEdges', 'getParticlesWithinSphere']
import os
import numpy as np
from fast3tree import fast3tree
from readGadgetSnapshot import readGadgetSnapshot

def getSnapshotEdges(snapshot_prefix, output_file=None, single_type=-1, lgadget=False):
    print "[Info] Calculating the edges of each snapshot file..."
    num_files = readGadgetSnapshot('{0}.0'.format(snapshot_prefix)).num_files
    edges = np.zeros((num_files, 6))
    for i, x in enumerate(edges):
        __, pos = readGadgetSnapshot('{0}.{1}'.format(snapshot_prefix, i), \
                read_pos=True, single_type=single_type, lgadget=lgadget)
        if len(pos):
            x[:3] = pos.min(axis=0)
            x[3:] = pos.max(axis=0)
        else:
            x[:3] = np.inf
            x[3:] = -np.inf
    if output_file is not None:
        np.save(output_file, edges)
    return edges

def _yield_periodic_points(center, radius, box_size):
    cc = np.array(center)
    flag = (cc-radius < 0).astype(int) - (cc+radius >= box_size).astype(int)
    cp = cc + flag*box_size
    a = range(len(cc))
    for j in xrange(1 << len(cc)):
        for i in a:
            if j >> i & 1 == 0:
                cc[i] = center[i]
            elif flag[i]:
                cc[i] = cp[i]
            else:
                break
        else:
            yield cc

def _check_within(c, r, edges):
    return all((c+r >= edges[:3]) & (c-r <= edges[3:]))

def _find_intersected_regions(center, radius, box_size, edges):
    num_subregions = len(edges)
    region_list = []
    for ci in _yield_periodic_points(center, radius, box_size):
        region_list.extend(np.where([_check_within(ci, radius, edges[i]) \
                for i in xrange(num_subregions)])[0])
    region_list = list(set(region_list))
    region_list.sort()
    return region_list

_valid_names = ('r', 'v', 'vr', 'x', 'y', 'z', 'vx', 'vy', 'vz', 'id')
_pos_names = ('r', 'vr', 'x', 'y', 'z')

def _is_string_like(obj):
    'Return True if *obj* looks like a string'
    try:
        obj + ''
    except:
        return False
    return True

def getParticlesWithinSphere(center, radius, snapshot_prefix, \
        snapshot_edges=None, output_dtype=np.dtype([('r', float)]), \
        vel_center=(0,0,0), single_type=-1, lgadget=False):
    #check output fields
    output_names = output_dtype.names
    if not all((x in _valid_names for x in output_names)):
        raise ValueError('Unknown names in output_dtype.')
    if not any((x in output_names for x in _valid_names)):
        raise ValueError('You do not need any output??')
    need_pos = any((x in _pos_names for x in output_names))
    need_vel = any((x.startswith('v') for x in output_names))
    need_id = ('id' in output_names)

    #load one header to get box_size and num_files
    header = readGadgetSnapshot('{0}.0'.format(snapshot_prefix))
    box_size = header.BoxSize
    num_files = header.num_files

    #load edges file
    if _is_string_like(snapshot_edges):
        if os.path.isfile(snapshot_edges):
            edges = np.load(snapshot_edges)
        else:
            edges = getSnapshotEdges(snapshot_prefix, snapshot_edges, \
                    single_type=single_type, lgadget=lgadget)
    elif snapshot_edges is None:
        edges = getSnapshotEdges(snapshot_prefix, \
                single_type=single_type, lgadget=lgadget)
    else:
        edges = np.asarray(snapshot_edges).reshape((num_files, 6))
    
    #actually load particles
    region_list = _find_intersected_regions(center, radius, box_size, edges)
    npart = 0
    for region in region_list:
        snapshot_data = readGadgetSnapshot('{0}.{1}'.format(snapshot_prefix, region), \
                read_pos=True, read_vel=need_vel, read_id=need_id, \
                single_type=single_type, lgadget=lgadget)
        s_pos = snapshot_data[1]
        if need_vel: s_vel = snapshot_data[2]
        if need_id: s_id = snapshot_data[-1]

        with fast3tree(s_pos) as tree:
            tree.set_boundaries(0, box_size)
            p = tree.query_radius(center, radius, True)

        if not len(p):
            continue

        if npart:
            out.resize(npart+len(p))
        else:
            out = np.empty(len(p), output_dtype)
        for x in ('r', 'v', 'vr'):
            if x in output_names:
                out[x][npart:] = 0

        if need_pos or need_vel:
            for i, ax in enumerate('xyz'):
                vax = 'v'+ax
                if need_pos: dx = s_pos[p,i] - center[i]
                if need_vel: dvx = s_vel[p,i] - vel_center[i]
                if ax   in output_names: out[ax][npart:] = dx
                if vax  in output_names: out[vax][npart:] = dvx
                if 'r'  in output_names: out['r'][npart:] += dx*dx
                if 'v'  in output_names: out['v'][npart:] += dvx*dvx
                if 'vr' in output_names: out['vr'][npart:] += dx*dvx
        
        if need_id:
            out['id'][npart:] = s_id[p]
        
        npart += len(p)

    if not npart:
        return np.empty(0, output_dtype)

    for x in ('r', 'v'):
        if x in output_names:
            out[x] **= 0.5

    return out

