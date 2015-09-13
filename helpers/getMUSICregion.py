__all__ = ['getMUSICregion']
from SimulationAnalysis import readHlist
from readGadgetSnapshot import readGadgetSnapshot
from findLagrangianVolume import findLagrangianVolume

_fmt = lambda a: ', '.join(map(str, a))

def getMUSICregion(target_id, rvir_mult, hlist, snapshot_prefix, ic_prefix,\
        edges_file=None):
    halos = readHlist(hlist, ['id', 'rvir', 'x', 'y', 'z'])
    target = halos[(halos['id'] == target_id)][0]
    c = [target[ax] for ax in 'xyz']
    r = rvir_mult * target['rvir'] * 1.e-3
    
    header = readGadgetSnapshot(snapshot_prefix+'.0')
    box_size = header.BoxSize
    
    rec_cor, rec_len = findLagrangianVolume(c, r, \
            snapshot_prefix, ic_prefix, edges_file, rec_only=True)
    print 'region          = box'
    print 'ref_offset      =', _fmt(rec_cor/box_size)
    print 'ref_extent      =', _fmt(rec_len/box_size)

def main():
    from sys import argv
    from json import load
    with open(argv[1], 'r') as fp:
        d = load(fp)
    getMUSICregion(**d)

if __name__ == '__main__':
    main()
