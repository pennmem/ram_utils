import hashlib

def compute_md5_key(filename):
    try:
        hash_md5 = hashlib.md5()
        with open(filename, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):

                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    except:
        return ''
