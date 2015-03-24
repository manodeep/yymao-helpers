__all__ = ['distributeMUSICBndryPart']
import struct
import numpy as np

_header_fmt = '6i6dddii6Iiiddddii6Ii60s'
_small = 1024*256 #256KB

def distributeMUSICBndryPart(fname_in, fname_out=None, long_ids=False, \
        coarse_type=5, distribute=(1,1,1,-1)):

    my_dist = map(int, distribute)
    if len(my_dist) != 4:
        raise ValueError('distribute must be an iterable of length 4.')
    neg = filter(lambda x: x<0, my_dist)
    if len(neg) > 1:
        raise ValueError('There can be at most one element in distribute with negative value.')
    elif len(neg) == 1:
        rest = my_dist.index(neg[0])
    else:
        rest = None
    
    with open(fname_in, 'rb') as f:
        f.seek(4, 1)
        header = list(struct.unpack(_header_fmt, f.read(256)))
        f.seek(4, 1)
        #
        types = range(0, 6)
        types.remove(1)
        types.remove(coarse_type)
        if any([header[i] for i in types]) or header[23] != 1:
            raise ValueError('Issues in this IC file.')
        #
        pos_vel_id = (header[1]+header[coarse_type])*(7+int(long_ids))*4 + 24
        f.seek(pos_vel_id, 1)
        #
        f.seek(4, 1)
        m = np.fromfile(f, np.float32, header[coarse_type])
        f.seek(4, 1)
        #
        m_sep = np.where(m[1:]-m[:-1])[0] + 1
        m_sep = [0] + m_sep.tolist() + [len(m)]
        ntype = len(m_sep) - 1
        #
        if rest is None and ntype != sum(my_dist):
            raise ValueError('Sum of distribute must equal the total number of coarse levels. Otherwise use -1 for one of the element in distribute.')
        if rest is not None:
            my_dist[rest] += ntype - sum(my_dist)
            if my_dist[rest] < 0:
                raise ValueError('Sum of distribute larger than the total number of coarse levels.')
        #
        count = 0
        m_slices = []
        m_size = 0
        for i, nt in zip(range(2, 6), my_dist):
            if nt == 0:
                header[i] = 0
                header[i+6] = 0.
            else:
                header[i] = m_sep[count+nt] - m_sep[count]
                if nt > 1:
                    header[i+6] = 0.
                    m_slices.append(slice(m_sep[count], m_sep[count+nt]))
                    m_size += header[i]
                else:
                    header[i+6] = float(m[m_sep[count]])
                count += nt
        #
        for i in range(2,6):
            header[i+16] = header[i]
        #
        f.seek(0, 0)
        if fname_out is None:
            fname_out = fname_in + '.out'
        with open(fname_out, 'wb') as fo:
            fo.write(f.read(4))
            fo.write(struct.pack(_header_fmt, *header))
            f.seek(256, 1)
            fo.write(f.read(4))
            #
            for i in range(pos_vel_id/_small):
                fo.write(f.read(_small))
            fo.write(f.read(pos_vel_id%_small))
            #
            if m_size:
                s = np.array([m_size*4], dtype=np.int32)
                s.tofile(fo)
                for sl in m_slices:
                    m[sl].tofile(fo)
                s.tofile(fo)
#
if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('fname_in')
    parser.add_argument('-o', dest='fname_out')
    parser.add_argument('-l', dest='long_ids', action='store_true')
    parser.add_argument('-t', dest='coarse_type', type=int, default=5)
    parser.add_argument('-d', dest='distribute', default='1,1,1,-1')
    args = parser.parse_args()
    distributeMUSICBndryPart(args.fname_in, args.fname_out, args.long_ids, \
            args.coarse_type, args.distribute.split(','))

