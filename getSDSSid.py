__all__ = ['getSDSSid']
import re
from urllib import urlopen

_re1 = re.compile('http://skyserver\.sdss3\.org/dr8/en/tools/explore/obj\.asp\?(ra=[+-]?\d+\.\d+)&amp;(dec=[+-]?\d+\.\d+)')
_re2 = re.compile('<td align=\'center\' width=\'33%\' class=\'t\'>(\d+)</td>')
_url1 = 'http://www.nsatlas.org/getAtlas.html?search=nsaid&nsaID=%s&submit_form=Submit'
_url2 = 'http://skyserver.sdss3.org/dr10/en/tools/quicklook/quickobj.aspx?%s&%s'

def getSDSSid(nsa_id):
    i = str(nsa_id)
    error_msg = '#Cannot find SDSS id for ' + i
    m = _re1.search(urlopen(_url1%i).read())
    if m is None: 
        return error_msg
    m = _re2.search(urlopen(_url2%m.groups()).read())
    if m is None:
        return error_msg
    return m.groups()[0]

if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser(description='Query SDSS website for object ID, given NSA ID.')
    parser.add_argument('id', type=int, nargs='*', help='List of NSA IDs')
    parser.add_argument('-f', nargs=2, help='Catalog containing NSA IDs in column X')
    args = parser.parse_args()

    ids = map(str, args.id)

    if args.f is not None:
        try:
            i = int(args.f[1]) - 1
            with open(args.f[0]) as f:
                for l in f:
                    ids.append(l.split()[i])
        except OSError:
            parser.error("It seems the file %s cannot be access."%args.f[0])
        except ValueError:
            parser.error("It seems the column %s is not correct."%args.f[1])
        except IndexError:
            parser.error("It seems the column %s is not correct."%args.f[1])

    for i in ids:
        print getSDSSid(i)

