import hashlib

sha1 = lambda s: hashlib.sha1(s.encode('utf-8')).hexdigest()
sha256 = lambda s: hashlib.sha256(s.encode('utf-8')).hexdigest()
