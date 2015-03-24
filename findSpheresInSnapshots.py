__all__ = ['getSnapshotEdges', 'getParticlesWithinSphere']
import os
import numpy as np
from fast3tree import fast3tree
from readGadgetSnapshot import readGadgetSnapshot

def getSnapshotEdges(snapshot_prefix, output_file=None):
    print "[Info] Calculating the edges of each snapshot file..."
    num_subregions = readGadgetSnapshot('%s.0'%(snapshot_prefix)).num_files
    edges = np.zeros((num_subregions, 6))
    for i in xrange(num_subregions):
        snapshot_file = '%s.%d'%(snapshot_prefix, i)
        header, pos = readGadgetSnapshot(snapshot_file, read_pos=True)
        if len(pos):
            edges[i, :3] = np.min(pos, axis=0)
            edges[i, 3:] = np.max(pos, axis=0)
        else:
            edges[i, :3] = np.array([np.inf]*3)
            edges[i, 3:] = np.array([-np.inf]*3)
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

def getParticlesWithinSphere(center, radius, snapshot_prefix, \
        snapshot_edges=None, output_dtype=np.dtype([('r', float)]), \
        vel_center=(0,0,0), id_int64=None):
    #
    valid_names = ['r', 'v', 'vr', 'x', 'y', 'z', 'vx', 'vy', 'vz', 'id']
    pos_names = ['r', 'vr', 'x', 'y', 'z']
    output_names = output_dtype.names
    if not all(map(lambda x: x in valid_names, output_names)):
        raise ValueError('Unknown names in output_dtype.')
    need_pos = any(map(lambda x: x in pos_names, output_names))
    need_vel = any(map(lambda x: x[0]=='v', output_names))
    need_id = ('id' in output_names)
    if not any([need_pos, need_vel, need_id]):
        raise ValueError('You do not need any output??')
    #
    if isinstance(snapshot_edges, basestring):
        if os.path.isfile(snapshot_edges):
            edges = np.load(snapshot_edges)
        else:
            edges = getSnapshotEdges(snapshot_prefix, snapshot_edges)
    else:
        edges = getSnapshotEdges(snapshot_prefix)
    box_size = readGadgetSnapshot('%s.0'%(snapshot_prefix)).BoxSize
    region_list = _find_intersected_regions(center, radius, box_size, edges)
    #
    npart = 0
    for ir, region in enumerate(region_list):
        snapshot_file = '%s.%d'%(snapshot_prefix, region)
        snapshot_data = readGadgetSnapshot(snapshot_file, read_pos=True, \
                read_vel=need_vel, read_id=need_id)
        s_pos = snapshot_data[1]
        if need_vel: s_vel = snapshot_data[2]
        if need_id: s_id = snapshot_data[-1]
        #
        with fast3tree(s_pos) as tree:
            tree.set_boundaries(0, box_size)
            p = tree.query_radius(center, radius, True)
        #
        if ir == 0:
            out = np.zeros(len(p), output_dtype)
        else:
            out.resize(npart+len(p))
        #
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
    if 'r' in output_names: out['r'] **= 0.5
    if 'v' in output_names: out['v'] **= 0.5
    return out

