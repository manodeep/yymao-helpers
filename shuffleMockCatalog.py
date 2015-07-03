__all__ = ['shuffleMockCatalog']
import warnings
import numpy as np
from numpy.lib.recfunctions import rename_fields

def _iter_plateau_in_sorted_array(a):
    k = np.where(a[1:] != a[:-1])[0]
    k += 1
    i = 0
    for j in k:
        yield i, j
        i = j
    yield i, len(a)

def _iter_indices_in_bins(bins, a):
    k = np.searchsorted(bins, a)
    s = k.argsort()
    for i, j in _iter_plateau_in_sorted_array(k[s]):
        yield s[i:j]

_axes = list('xyz')

def shuffleMockCatalog(mock_ids, halo_catalog, bins=None, proxy='mvir', \
        box_size=None, apply_rsd=False):
    
    # check necessary fields in halo_catalog
    fields = ['id', 'upid', proxy] + _axes
    if apply_rsd:
        fields.append('vz')
    if not all((f in halo_catalog.dtype.names for f in fields)):
        raise ValueError('`halo_catalog` should have the following fields: '+ \
                ', '.join(fields))

    # check all mock_ids are in halo_catalog
    s = halo_catalog.argsort(order='id')
    idx = np.searchsorted(halos['id'], mock_ids, sorter=s)
    try:
        idx = s[idx]
    except IndexError:
        raise ValueError('`mock_ids` must all present in `halo_catalog`')
    if not (halo_catalog['id'][idx] == mock_ids).all():
        raise ValueError('`mock_ids` must all present in `halo_catalog`')
    mock_idx = np.ones(len(halo_catalog), dtype=int)
    mock_idx *= -1
    mock_idx[idx] = np.arange(len(mock_ids))
    del s, idx
    
    # separate hosts and subs
    host_flag = (halo_catalog['upid'] == -1)
    hosts = rename_fields(halo_catalog[host_flag], {'upid':'mock_idx'})
    hosts['mock_idx'] = mock_idx[host_flag]
    subs = rename_fields(halo_catalog[~host_flag], {'id':'mock_idx'})
    subs['mock_idx'] = mock_idx[~host_flag]
    del host_flag, mock_idx

    # group subhalos
    subs.sort(order='upid')
    idx = np.fromiter(_iter_plateau_in_sorted_array(subs['upid']), \
            np.dtype([('start', int), ('stop', int)]))
    host_ids = subs['upid'][idx['start']]
    subs_idx = np.zeros(len(hosts), dtype=idx.dtype)
    subs_idx[np.in1d(hosts['id'], host_ids, True)] \
            = idx[np.in1d(host_ids, hosts['id'], True)]
    del idx, host_ids

    # check bins
    if bins is None:
        bins = 50
    try:
        bins = int(bins)
    except (ValueError, TypeError):
        bins = np.asarray(bins)
    else:
        bins = np.logspace(np.log10(hosts[proxy].min()*0.9999), \
                np.log10(hosts[proxy].max()), bins+1)

    # create the array for storing results
    pos = np.empty(len(mock_ids), dtype=np.dtype(zip(_axes, [float]*3)))

    # loop of bins of proxy (e.g. mvir)
    for i, indices in enumerate(_iter_indices_in_bins(bins, hosts[proxy])):
        if i==0 or i==len(bins):
            if (hosts['mock_idx'][indices] > -1).any() or \
                    any(((subs['mock_idx'][slice(*subs_idx[j])] > -1).any() \
                    for j in indices)):
                warnings.warn('Some halos associdated with the mock catalog are outside the bin range.', RuntimeWarning)
            continue

        # swap satellites
        choices = indices.copy()
        n_choices = len(choices)
        for j in indices:
            subs_this = subs[slice(*subs_idx[j])]
            subs_this = subs_this[subs_this['mock_idx'] > -1]
            if not len(subs_this):
                continue
            # find new host
            k = np.random.randint(n_choices)
            n_choices -= 1
            host, host_new = hosts[[j, choices[k]]]
            choices[k] = choices[n_choices]
            # actually do the swapping
            mock_idx_this = subs_this['mock_idx']
            pos[mock_idx_this] = subs_this[_axes]
            for ax in _axes:
                pos[ax][mock_idx_this] += (host_new[ax] - host[ax])
            if apply_rsd:
                pos['z'][mock_idx_this] += (subs_this['vz'] \
                        + host_new['vz'] - host['vz'])/100.0

        # swap hosts
        mock_idx_this = hosts['mock_idx'][indices]
        mock_idx_this = mock_idx_this[mock_idx_this > -1]
        if len(mock_idx_this):
            k = np.random.choice(indices, len(mock_idx_this), replace=False)
            pos[mock_idx_this] = hosts[_axes][k]
            if apply_rsd:
                pos['z'][mock_idx_this] += hosts['vz'][k]/100.0

    # wrap box
    if box_size is not None:
        for ax in _axes:
            np.remainder(pos[ax], box_size, pos[ax])

    return pos
