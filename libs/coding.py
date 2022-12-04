import hashlib

md5 = lambda s: hashlib.md5(s).hexdigest()
sha1 = lambda s: hashlib.sha1(s.encode('utf-8')).hexdigest()
sha256 = lambda s: hashlib.sha256(s.encode('utf-8')).hexdigest()


if __name__ == '__main__':
    d = {
        'name': 'ali',
        'ok': 'okkk',
        'age': 16
    }
    d2 = {
        'age': 161,
        'name': 'ali1',
        'ok': 'okkk1',
    }
    s1 = ' | '.join((list(d.keys())))
    s2 = ' | '.join((list(d2.keys())))
    print('{} -> {}'.format(s1, sha1(s1)))
    print('{} -> {}'.format(s2, sha1(s2)))
    print('='*30)
    s1 = ' | '.join(set(list(d.keys())))
    s2 = ' | '.join(set(list(d2.keys())))
    print('{} -> {}'.format(s1, sha1(s1)))
    print('{} -> {}'.format(s2, sha1(s2)))