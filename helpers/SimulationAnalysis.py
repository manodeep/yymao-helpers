__all__ = ['getMainBranch', 'a2z', 'z2a', 'readHlist', 'SimulationAnalysis', \
        'TargetHalo', 'getDistance', 'iter_grouped_subhalos_indices']

import os
import re
import math
import gzip
from itertools import izip
from urllib import urlretrieve
from collections import deque
import numpy as np

_islistlike = lambda l: hasattr(l, '__iter__')
a2z = lambda a: 1./a - 1.
z2a = lambda z: 1./(1.+z)

def getMainBranch(iterable, get_num_prog):
    item = iter(iterable)
    q = deque([(item.next(), True)])
    X = []
    while len(q):
        i, i_mb = q.popleft()
        X.append(i_mb)
        n = get_num_prog(i)
        prog_mb = [i_mb] + [False]*(n-1) if n else []
        q.extend([(item.next(), mb) for mb in prog_mb])
    return np.array(X)


class BaseParseFields():
    def __init__(self, header, fields=None):
        if len(header)==0:
            if all([isinstance(f, int) for f in fields]):
                self._usecols = fields
                self._formats = [float]*len(fields)
                self._names = ['f%d'%f for f in fields]
            else:
                raise ValueError('header is empty, so fields must be a list '\
                        'of int.')
        else:
            header_s = map(self._name_strip, header)
            if fields is None or len(fields)==0:
                self._names = header
                names_s = header_s
                self._usecols = range(len(names_s))
            else:
                if not _islistlike(fields):
                    fields = [fields]
                self._names = [header[f] if isinstance(f, int) else str(f) \
                        for f in fields]
                names_s = map(self._name_strip, self._names)
                wrong_fields = filter(bool, [str(f) if s not in header_s \
                        else '' for s, f in zip(names_s, fields)])
                if len(wrong_fields):
                    raise ValueError('The following field(s) are not available'\
                            ': %s.\nAvailable fields: %s.'%(\
                            ', '.join(wrong_fields), ', '.join(header)))
                self._usecols = map(header_s.index, names_s)
            self._formats = map(self._get_format, names_s)

    def parse_line(self, l):
        items = l.split()
        return tuple([c(items[i]) for i, c in \
                zip(self._usecols, self._formats)])

    def pack(self, X):
        return np.array(X, np.dtype({'names':self._names, \
                'formats':self._formats}))

    def _name_strip(self, s):
        return self._re_name_strip.sub('', s).lower()

    def _get_format(self, s):
        return float if self._re_formats.search(s) is None else int

    _re_name_strip = re.compile('\W|_')
    _re_formats = re.compile('^phantom$|^mmp$|id$|^num|num$')


class BaseDirectory:
    def __init__(self, dir_path='.'):   
        self.dir_path = os.path.expanduser(dir_path)

        #get file_index
        files = os.listdir(self.dir_path)
        matches = filter(lambda m: m is not None, \
                map(self._re_filename.match, files))
        if len(matches) == 0:
            raise ValueError('cannot find matching files in this directory: %s.'%(self.dir_path))
        indices = np.array(map(self._get_file_index, matches))
        s = indices.argsort()
        self.files = [matches[i].group() for i in s]
        self.file_indices = indices[s]

        #get header and header_info
        header_info_list = []
        with open('%s/%s'%(self.dir_path, self.files[0]), 'r') as f:
            for l in f:
                if l[0] == '#':
                    header_info_list.append(l)
                else:
                    break
        if len(header_info_list):
            self.header_info = ''.join(header_info_list)
            self.header = [self._re_header_remove.sub('', s) for s in \
                    header_info_list[0][1:].split()]
        else:
            self.header_info = ''
            self.header = []

        self._ParseFields = self._Class_ParseFields(self.header, \
                self._default_fields)

    def _load(self, index, exact_index=False, additional_fields=[]):
        p = self._get_ParseFields(additional_fields)
        fn = '%s/%s'%(self.dir_path, self.get_filename(index, exact_index))
        with open(fn, 'r') as f:
            l = '#'
            while l[0] == '#':
                try:
                    l = f.next()
                except StopIteration:
                    return p.pack([])
            X = [p.parse_line(l)]
            for l in f:
                X.append(p.parse_line(l))
        return p.pack(X)

    def _get_file_index(self, match):
        return match.group()

    def get_filename(self, index, exact_index=False):
        if exact_index:
            i = self.file_indices.searchsorted(index)
            if self.file_indices[i] != index:
                raise ValueError('Cannot find the exact index %s.'%(str(index)))
        else:
            i = np.argmin(np.fabs(self.file_indices - index))
        return self.files[i]

    def _get_ParseFields(self, additional_fields):
        if not _islistlike(additional_fields) or len(additional_fields)==0:
            return self._ParseFields
        else:
            return self._Class_ParseFields(self.header, \
                    self._default_fields + list(additional_fields))

    _re_filename = re.compile('.+')
    _re_header_remove = re.compile('')
    _Class_ParseFields = BaseParseFields
    _default_fields = []
    load = _load


class HlistsDir(BaseDirectory):
    _re_filename = re.compile('^hlist_(\d+\.\d+).list$')
    _re_header_remove = re.compile('\(\d+\)$')
    _default_fields = ['id', 'upid', 'mvir', 'rvir', 'rs', 'x', 'y', 'z', \
            'vmax', 'vpeak']

    def _get_file_index(self, match):
        return math.log10(float(match.groups()[0]))

    def load(self, z=0, exact_index=False, additional_fields=[]):
        return self._load(math.log10(z2a(z)), exact_index, additional_fields)


class RockstarDir(BaseDirectory):
    _re_filename = re.compile('^out_(\d+).list$')
    _re_header_remove = re.compile('')
    _default_fields = ['id', 'mvir', 'rvir', 'rs', 'x', 'y', 'z', 'vmax']

    def _get_file_index(self, match):
        fn = '%s/%s'%(self.dir_path, match.group())
        with open(fn, 'r') as f:
            for l in f:
                if l.startswith('#a '):
                    break
            else:
                raise ValueError('Cannot find the scale factor in this file %s'\
                        %(fn))
        return math.log10(float(l.split()[-1]))

    def load(self, z=0, exact_index=False, additional_fields=[]):
        return self._load(math.log10(z2a(z)), exact_index, additional_fields)


class TreesDir(BaseDirectory):
    _re_filename = re.compile('^tree_\d+_\d+_\d+.dat$')
    _re_header_remove = re.compile('\(\d+\)$')
    _default_fields = ['scale', 'id', 'num_prog', 'upid', 'mvir', 'rvir', \
            'rs', 'x', 'y', 'z', 'vmax']

    def load(self, tree_root_id, additional_fields=[]):
        p = self._get_ParseFields(additional_fields)
        tree_root_id_str = str(tree_root_id)
        location_file = self.dir_path + '/locations.dat'
        if os.path.isfile(location_file):
            with open(location_file, 'r') as f:
                f.readline()
                for l in f:
                    items = l.split()
                    if items[0] == tree_root_id_str:
                        break
                else:
                    raise ValueError("Cannot find this tree_root_id: %d."%(\
                                    tree_root_id))
            tree_file = '%s/%s'%(self.dir_path, items[-1])
            with open(tree_file, 'r') as f:
                f.seek(int(items[2]))
                X = []
                for l in f:
                    if l[0] == '#': break
                    X.append(p.parse_line(l))
        else:
            for fn in self.files:
                tree_file = '%s/%s'%(self.dir_path, fn)
                with open(tree_file, 'r') as f:
                    l = '#'
                    while l[0] == '#':
                        try:
                            l = f.next()
                        except StopIteration:
                            raise ValueError("Cannot find this tree_root_id: %d."%(\
                                    tree_root_id))
                    num_trees = int(l)
                    for l in f:
                        if l[0] == '#' and l.split()[-1] == tree_root_id_str:
                            break #found tree_root_id
                    else:
                        continue #not in this file, check the next one
                    X = []
                    for l in f:
                        if l[0] == '#': break
                        X.append(p.parse_line(l))
                    break #because tree_root_id has found
            else:
                raise ValueError("Cannot find this tree_root_id: %d."%(\
                                    tree_root_id))
        return p.pack(X)


def readHlist(hlist, fields=None, buffering=100000000):
    """
    Read the given fields of a hlist file (also works for tree_*.dat and out_*.list) as a numpy record array.

    Parameters
    ----------
    hlist : str or file obj
        The path to the file (can be an URL) or a file object.
    fields : str, int, array_like, optional
        The desired fields. It can be a list of string or int. If fields is None (default), return all the fields listed in the header. 

    Returns
    -------
    arr : ndarray
        A numpy record array contains the data of the desired fields.

    Example
    -------
    >>> h = readHlist('hlist_1.00000.list', ['id', 'mvir', 'upid'])
    >>> h.dtype
    dtype([('id', '<i8'), ('mvir', '<f8'), ('upid', '<i8')])
    >>> mass_of_hosts = h['mvir'][(h['upid'] == -1)]
    >>> largest_halo_id = h['id'][h['mvir'].argmax()]
    >>> mass_of_subs_of_largest_halo = h['mvir'][(h['upid'] == largest_halo_id)]

    """
    if hasattr(hlist, 'read'):
        f = hlist
    else:
        if re.match(r'(s?ftp|https?)://', hlist, re.I):
            hlist = urlretrieve(hlist)[0]
        if hlist.endswith('.gz'):
            f = gzip.open(hlist, 'r')
        else:
            f = open(hlist, 'r', int(buffering))
    try:
        l = f.next()
        header = l[1:].split()
        header = [re.sub('\(\d+\)$', '', s) for s in header]
        p = BaseParseFields(header, fields)
        while l[0] == '#':
            try:
                l = f.next()
            except StopIteration:
                return p.pack([])
        X = [p.parse_line(l)]
        for l in f:
            X.append(p.parse_line(l))
    finally:
        if not hasattr(hlist, 'read'):
            f.close()
    return p.pack(X)

def readHlistToSqlite3(db, table_name, hlist, fields=None, unique_id=True):
    """
    Read the given fields of a hlist file (also works for tree_*.dat and out_*.list) and save it to a sqlite3 database.

    Parameters
    ----------
    db : sqlite3.Cursor
        A sqlite3.Cursor object.
    hlist : str
        The path to the file.
    fields : str, int, array_like, optional
        The desired fields. It can be a list of string or int. If fields is None (default), return all the fields listed in the header. 

    Returns
    -------
    db : sqlite3.Cursor
        The same cursor object that was given as input.

    """

    with open(hlist, 'r') as f:
        l = f.next()
        header = l[1:].split()
        header = [re.sub('\(\d+\)$', '', s) for s in header]
        p = BaseParseFields(header, fields)
        db_cols = map(lambda s: re.sub(r'\W+', '_', s), \
                map(lambda s: re.sub(r'^\W+|\W+$', '', s), p._names))
        db_create_stmt = 'create table if not exists %s (%s)'%(table_name, \
                ','.join(['%s %s%s'%(name, 'int' if (fmt is int) else 'real', \
                ' unique' if (name == 'id' and unique_id) else '') \
                for name, fmt in zip(db_cols, p._formats)]))
        db_insert_stmt = 'insert or replace into %s values (%s)'%(table_name, \
                ','.join(['?']*len(p._names)))
        empty_file = False
        while l[0] == '#':
            try:
                l = f.next()
            except StopIteration:
                empty_file = True
        db.execute(db_create_stmt)
        if not empty_file:
            db.execute(db_insert_stmt, p.parse_line(l))
            for l in f:
                db.execute(db_insert_stmt, p.parse_line(l))
        db.commit()
    return db

class SimulationAnalysis:

    def __init__(self, hlists_dir=None, trees_dir=None, rockstar_dir=None):
        self._directories = {}
       
        if hlists_dir is not None:
            self.set_hlists_dir(hlists_dir)

        if trees_dir is not None:
            self.set_trees_dir(trees_dir)

        if rockstar_dir is not None:
            self.set_rockstar_dir(rockstar_dir)

        if len(self._directories) == 0:
            raise ValueError('Please specify at least one directory.')

    def set_trees_dir(self, trees_dir):
        self._directories['trees'] = TreesDir(trees_dir)
        self._trees = {}
        self._main_branches = {}

    def set_rockstar_dir(self, rockstar_dir):
        self._directories['olists'] = RockstarDir(rockstar_dir)
        self._olists = {}

    def set_hlists_dir(self, hlists_dir):
        self._directories['hlists'] = HlistsDir(hlists_dir)
        self._hlists = {}

    def load_tree(self, tree_root_id=-1, npy_file=None, additional_fields=[]):
        if 'trees' not in self._directories:
            raise ValueError('You must set trees_dir before using this function.')
        if npy_file is not None and os.path.isfile(npy_file):
            data = np.load(npy_file)
            if tree_root_id < 0:
                tree_root_id = data['id'][0]
            elif tree_root_id != data['id'][0]:
                raise ValueError('tree_root_id does not match.')
            self._trees[tree_root_id] = data
        elif tree_root_id not in self._trees:
            self._trees[tree_root_id] = \
                    self._directories['trees'].load(tree_root_id, \
                    additional_fields=additional_fields)
        if npy_file is not None and not os.path.isfile(npy_file):
            np.save(npy_file, self._trees[tree_root_id])
        return self._trees[tree_root_id]

    def load_main_branch(self, tree_root_id=-1, npy_file=None, keep_tree=False, \
            additional_fields=[]):
        if 'trees' not in self._directories:
            raise ValueError('You must set trees_dir before using this function.')
        if npy_file is not None and os.path.isfile(npy_file):
            data = np.load(npy_file)
            if tree_root_id < 0:
                tree_root_id = data['id'][0]
            elif tree_root_id != data['id'][0]:
                raise ValueError('tree_root_id does not match.')
            self._main_branches[tree_root_id] = data
        elif tree_root_id not in self._main_branches:
            t = self._directories['trees'].load(tree_root_id, \
                    additional_fields=additional_fields)
            mb = getMainBranch(t, lambda s: s['num_prog'])
            if keep_tree:
                self._trees[tree_root_id] = t
            self._main_branches[tree_root_id] = t[mb]
        if npy_file is not None and not os.path.isfile(npy_file):
            np.save(npy_file, self._main_branches[tree_root_id])
        return self._main_branches[tree_root_id]

    def _choose_hlists_or_olists(self, use_rockstar=False):
        if 'hlists' not in self._directories and \
                'olists' not in self._directories:
            raise ValueError('You must set hlists_dir and/or rockstar_dir'\
                    'before using this function.')
        elif 'olists' not in self._directories:
            if use_rockstar:
                print "Warning: ignore use_rockstar"
            return self._directories['hlists'], self._hlists
        elif use_rockstar or 'hlists' not in self._directories:
            return self._directories['olists'], self._olists
        else:
            return self._directories['hlists'], self._hlists

    def load_halos(self, z=0, npy_file=None, use_rockstar=False, \
            additional_fields=[]):
        d, s = self._choose_hlists_or_olists(use_rockstar)
        fn = d.get_filename(math.log10(z2a(z)))
        if npy_file is not None and os.path.isfile(npy_file):
            data = np.load(npy_file) 
            s[fn] = data
        elif fn not in s:
            s[fn] = d.load(z, additional_fields=additional_fields)
        if npy_file is not None and not os.path.isfile(npy_file):
            np.save(npy_file, s[fn])
        return s[fn]

    def del_tree(self, tree_root_id):
        if tree_root_id in self._trees:
            del self._trees[tree_root_id]

    def del_main_branch(self, tree_root_id):
        if tree_root_id in self._main_branches:
            del self._main_branches[tree_root_id]

    def del_halos(self, z, use_rockstar=False):
        d, s = self._choose_hlists_or_olists(use_rockstar)
        fn = d.get_filename(math.log10(z2a(z)))
        if fn in s:
            del s[fn]

    def clear_trees(self):
        self._trees = {}

    def clear_main_branches(self):
        self._main_branches = {}

    def clear_halos(self):
        self._olists = {}
        self._hlists = {}

class TargetHalo:
    def __init__(self, target, halos, box_size=-1):
        self.target = target
        try:
            self.target_id = target['id']
        except KeyError:
            pass
        self.halos = halos
        self.dists = np.zeros(len(halos), float)
        self.box_size = box_size
        half_box_size = 0.5*box_size
        for ax in 'xyz':
            d = halos[ax] - target[ax]
            if box_size > 0:
                d[(d >  half_box_size)] -= box_size
                d[(d < -half_box_size)] += box_size
            self.dists += d*d
        self.dists = np.sqrt(self.dists)

def getDistance(target, halos, box_size=-1):
    t = TargetHalo(target, halos, box_size)
    return t.dists


def iter_grouped_subhalos_indices(host_ids, sub_pids):
    s = sub_pids.argsort()
    k = np.where(sub_pids[s[1:]] != sub_pids[s[:-1]])[0]
    k += 1
    k = np.vstack((np.insert(k, 0, 0), np.append(k, len(s)))).T
    d = np.searchsorted(sub_pids[s[k[:,0]]], host_ids)
    for j, host_id in izip(d, host_ids):
        if j < len(s) and sub_pids[s[k[j,0]]] == host_id:
            yield s[slice(*k[j])]
        else:
            yield np.array([], dtype=int)

