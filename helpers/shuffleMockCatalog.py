__all__ = ['shuffleMockCatalog', 'generate_upid']
import warnings
from itertools import izip
import numpy as np
from numpy.lib.recfunctions import rename_fields

def _iter_plateau_in_sorted_array(a):
    if len(a):
        k = np.where(a[1:] != a[:-1])[0]
        k += 1
        i = 0
        for j in k:
            yield i, j
            i = j
        yield i, len(a)

def _iter_indices_in_bins(bins, a):
    if len(a) and len(bins):
        s = a.argsort()
        k = np.searchsorted(a, bins, 'right', sorter=s)
        i = 0
        for j in k:
            yield s[i:j]
            i = j
        yield s[i:]

def _apply_rotation(pos, box_size):
    half_box_size = box_size * 0.5
    pos[pos >  half_box_size] -= box_size
    pos[pos < -half_box_size] += box_size
    return np.dot(pos, np.linalg.qr(np.random.randn(3,3))[0])

_axes = list('xyz')

def _get_xyz(a, ax_type=float):
    return np.fromiter((a[ax] for ax in _axes), ax_type, 3)


def generate_upid(pid, id, recursive=True):
    """
    To generate (or to fix) the upid of a halo catalog.

    Parameters
    ----------
    pid : array_like
        An ndarray of integer that contains the parent IDs of each halo.
    id : array_like
        An ndarray of integer that contains the halo IDs.
    recursive : bool, optional
        Whether or not to run this function recursively. Default is True.

    Returns
    -------
    upid : array_like
        The ultimate parent IDs. 

    Examples
    --------
    >>> halos['upid'] = generate_upid(halos['pid'], halos['id'])
    
    """
    pid = np.ravel(pid)
    id = np.ravel(id)
    if len(id) != len(pid):
        raise ValueError('`pid` and `id` must have the same length.')
    if not len(pid):
        raise ValueError('`pid` and `id` must not be empty.')
    s = pid.argsort()
    idx = np.fromiter(_iter_plateau_in_sorted_array(pid[s]), \
            np.dtype([('start', int), ('stop', int)]))
    unique_pid = pid[s[idx['start']]]
    if unique_pid[0] == -1:
        unique_pid = unique_pid[1:]
        idx = idx[1:]
    host_flag = (pid == -1)
    not_found = np.where(np.in1d(unique_pid, id[host_flag], True, True))[0]
    if not len(not_found):
        return pid
    sub_flag = np.where(~host_flag)[0]
    found = sub_flag[np.in1d(id[sub_flag], unique_pid[not_found], True)]
    found = found[id[found].argsort()]
    assert (id[found] == unique_pid[not_found]).all()
    del host_flag, sub_flag, unique_pid
    pid_old = pid.copy()
    for i, j in izip(found, not_found):
        pid[s[slice(*idx[j])]] = pid_old[i]
    del pid_old, idx, s, found, not_found
    return generate_upid(pid, id, True) if recursive else pid


def shuffleMockCatalog(mock_ids, halo_catalog, bin_width=None, bins=None,
        proxy='mvir', box_size=None, apply_rsd=False,
        shuffle_centrals=True, shuffle_satellites=True, rotate_satellites=False,
        return_structured_array=False):
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
        (Default is False)
    shuffle_centrals : bool, optional
        Whether or not to shuffle central galaxies (Default is True)
    shuffle_satellites : bool, optional
        Whether or not to shuffle satellite galaxies (Default is True)
    rotate_satellites : bool, optional
        Whether or not to apply a random rotation to satellite galaxies 
        (Default is False)
    return_structured_array : bool, optional
        Whether to return a structured array that contains x, y, z
        or just a n-by-3 float array.

    Returns
    -------
    pos : array_like
        A ndarray that contains x, y, z of the shuffled positions.
    """

    # check necessary fields in halo_catalog
    fields = ['id', 'upid', proxy] + _axes
    if apply_rsd:
        fields.append('vz')
    if not all((f in halo_catalog.dtype.names for f in fields)):
        raise ValueError('`halo_catalog` should have the following fields: '+ \
                ', '.join(fields))

    # check dtype
    ax_type = halo_catalog['x'].dtype.type
    if any((halo_catalog[ax].dtype.type != ax_type for ax in 'yz')):
        raise ValueError('The types of fields x, y, z in `halo_catalog` ' \
                'must all be the same.')

    # check all mock_ids are in halo_catalog
    s = halo_catalog['id'].argsort()
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
    del idx
    
    # separate hosts and subs
    host_flag = (halo_catalog['upid'] == -1)
    subs = rename_fields(halo_catalog[~host_flag], {'id':'mock_idx'})
    subs['mock_idx'] = mock_idx[~host_flag]
    subs = subs[subs['mock_idx'] > -1] # only need subs that are mocks 
    host_flag = s[host_flag[s]] # this sorts `hosts` by `id`
    hosts = rename_fields(halo_catalog[host_flag], {'upid':'mock_idx'})
    hosts['mock_idx'] = mock_idx[host_flag]
    del host_flag, mock_idx, s

    # group subhalos
    subs.sort(order='upid')
    idx = np.fromiter(_iter_plateau_in_sorted_array(subs['upid']), \
            np.dtype([('start', int), ('stop', int)]))
    host_ids = subs['upid'][idx['start']]
    if not np.in1d(host_ids, hosts['id'], True).all():
        raise ValueError('Some subhalos associdated with the mock galaxies ' \
                'have no parent halos in `halo_catalog`. Consider using ' \
                '`generate_upid` to fix this.')
    # for the following to work, `hosts` need to be sorted by `id`
    subs_idx = np.zeros(len(hosts), dtype=idx.dtype)
    subs_idx[np.in1d(hosts['id'], host_ids, True)] = idx
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
    pos = np.empty((len(mock_ids), 3), ax_type)
    pos.fill(np.nan)

    # loop of bins of proxy (e.g. mvir)
    for i, indices in enumerate(_iter_indices_in_bins(bins, hosts[proxy])):
        if not len(indices):
            continue

        if i==0 or i==len(bins):
            if (hosts['mock_idx'][indices] > -1).any() or \
                    any((subs_idx['start'][j] < subs_idx['stop'][j] \
                    for j in indices)):
                warnings.warn('Some halos associdated with the mock catalog ' \
                        'are outside the bin range.', RuntimeWarning)
            continue

        # shuffle satellites
        if shuffle_satellites:
            choices = indices.tolist()
        for j in indices:
            subs_this = subs[slice(*subs_idx[j])]
            if not len(subs_this):
                continue
            mock_idx_this = subs_this['mock_idx']
            pos[mock_idx_this] = subs_this[_axes].view((ax_type,3))
            if shuffle_satellites:
                k = choices.pop(np.random.randint(len(choices)))
                pos[mock_idx_this] -= _get_xyz(hosts[j], ax_type)
                if rotate_satellites:
                    pos[mock_idx_this] = \
                            _apply_rotation(pos[mock_idx_this], box_size)
                pos[mock_idx_this] += _get_xyz(hosts[k], ax_type)
                if apply_rsd:
                    pos[mock_idx_this,2] += (subs_this['vz'] \
                            + hosts['vz'][k] - hosts['vz'][j])/100.0
            else:
                if rotate_satellites:
                    host_pos = _get_xyz(hosts[j], ax_type)
                    pos[mock_idx_this] -= host_pos
                    pos[mock_idx_this] = \
                            _apply_rotation(pos[mock_idx_this], box_size)
                    pos[mock_idx_this] += host_pos
                if apply_rsd:
                    pos[mock_idx_this,2] += subs_this['vz']/100.0
            
        # shuffle hosts
        has_mock = indices[hosts['mock_idx'][indices] > -1]
        if not len(has_mock):
            continue
        mock_idx_this = hosts['mock_idx'][has_mock]
        if shuffle_centrals:
            has_mock = np.random.choice(indices, len(has_mock), False)
        pos[mock_idx_this] = hosts[_axes][has_mock].view((ax_type,3))
        if apply_rsd:
            pos[mock_idx_this,2] += hosts['vz'][has_mock]/100.0

    # sanity check
    if np.isnan(pos).any():
        warnings.warn('Some galaxies in the mock catalog have not been ' \
                'assigned a new position. Maybe the corresponding halo is ' \
                'outside the bin range.', RuntimeWarning)

    # wrap box
    if box_size is not None:
        pos = np.remainder(pos, box_size, pos)

    if return_structured_array:
        pos = pos.view(np.dtype(zip(_axes, [ax_type]*3)))

    return pos

