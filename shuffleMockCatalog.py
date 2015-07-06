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
    s = a.argsort()
    k = np.searchsorted(a, bins, 'right', sorter=s)
    i = 0
    for j in k:
        yield s[i:j]
        i = j
    yield s[i:]

_axes = list('xyz')

def shuffleMockCatalog(mock_ids, halo_catalog, bin_width=None, bins=None,
        proxy='mvir', box_size=None, apply_rsd=False):
    """
    Shuffle a mock catalog according to Zentner et al. (2014) [arXiv:1311.1818]

    Parameters
    ----------
    mock_ids : array_like
        Should be a 1-d array of int which contains the corresponding halo IDs
        for the galaxies in the mock catalog to be shuffled.
    halo_catalog : array_like
        Should be a 1-d structrued array which has the following fields:
        id, upid, x, y, z, vz (if `apply_rsd` it True), and the proxy.
    bin_width : float or None, optional
        The width of the bin, in dex.
    bins : int, array_like, or None, optional
        If an integer is provided, it is interpreted as the number of bins.
        If an array is provided, it is interpreted as the edges of the bins.
        The parameter _overwrites_ `bin_width`.
    proxy : string, optional
        The proxy to bin on. Must be present in the fields of `halo_catalog`.
    box_size : float or None, optional
        The side length of the box. Should be in the same unit as x, y, z.
    apply_rsd : bool, optional
        Whether or not to apply redshift space distortions on the z-axis.

    Returns
    -------
    pos : array_like
        A 1-d strutured array that contains x, y, z of the shuffled positions.
    """

    # check necessary fields in halo_catalog
    fields = ['id', 'upid', proxy] + _axes
    if apply_rsd:
        fields.append('vz')
    if not all((f in halo_catalog.dtype.names for f in fields)):
        raise ValueError('`halo_catalog` should have the following fields: '+ \
                ', '.join(fields))

    # check all mock_ids are in halo_catalog
    s = halo_catalog.argsort(order='id')
    idx = np.searchsorted(halo_catalog['id'], mock_ids, sorter=s)
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
    try:
        bin_width = float(bin_width)
    except (ValueError, TypeError):
        bin_width = None
    else:
        if bin_width <= 0:
            bin_width = None
    if bin_width is None:
        bin_width = 0.1
    
    mi = np.log10(hosts[proxy].min()*0.99999)
    ma = np.log10(hosts[proxy].max())

    if bins is None:
        bins = int(np.ceil((ma-mi)/bin_width))
        mi = ma - bin_width*bins
    try:
        bins = int(bins)
    except (ValueError, TypeError):
        bins = np.asarray(bins)
        if len(bins) < 2 or (bins[1:]<bins[:-1]).any():
            raise ValueError('Please specify a valid `bin` parameter.')
    else:
        bins = np.logspace(mi, ma, bins+1)

    # create the array for storing results
    pos = np.empty(len(mock_ids), dtype=np.dtype(zip(_axes, [float]*3)))
    pos.fill(np.nan)

    # loop of bins of proxy (e.g. mvir)
    for i, indices in enumerate(_iter_indices_in_bins(bins, hosts[proxy])):
        if not len(indices):
            continue

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

    if any((np.isnan(pos[ax]).any() for ax in _axes)):
        warnings.warn('Some galaxies in the mock catalog have not been assigned a new position. Maybe the corresponding halo is outside the bin range.', RuntimeWarning)

    # wrap box
    if box_size is not None:
        for ax in _axes:
            np.remainder(pos[ax], box_size, pos[ax])

    return pos

