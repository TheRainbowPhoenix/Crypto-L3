# -*- coding: utf-8 -*-

from types import *
import base64

# DEFINES

_HEX = '0123456789ABCDEF'
_BASE58 = '123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz'
_RADIX64 = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz0123456789+/'
_ASCII85 = '!"#$%&\'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\\]^_`abcdefghijklmnopqrstu'
_Z85 = '0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ.-:+=^!/*?&<>()[]{}@%$#'
_BASE32 = '0123456789ABCDEFGHJKMNPQRSTVWXYZ'
_GMP = '0123456789ABCDEFGHIJKLMNOPQRSTUV'
_ZBASE32 = 'YBNDRFG8EJKMCPQXOT1UWISZA345H769'
_RFC4648 = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ234567'
_BASE36 = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'
_BINHEX4 = '!"#$%&\'()*+,-012345689@ABCDEFGHIJKLMNPQRSTUVXYZ[`abcdefhijklmpqr'
_UNIX_B64 = './0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
_6PACK = '0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ@_'

_ALPHABETS = {
    'HEX':_HEX,
    'BASE58':_BASE58,
    'RADIX64':_RADIX64,
    'ASCII85':_ASCII85,
    'Z85':_Z85,
    'BASE32':_BASE32,
    'GMP':_GMP,
    'ZBASE32':_ZBASE32,
    'RFC4648':_RFC4648,
    'BASE36':_BASE36,
    'BINHEX4':_BINHEX4,
    'UNIX_B64':_UNIX_B64,
    '6PACK':_6PACK
}

_ALPHA_AZ = '\033[31m{A-Z}\033[0m\033[7m'
_ALPHA_az = '\033[31m{a-z}\033[0m\033[7m'
_ALPHA_09 = '\033[31m{0-9}\033[0m'

def format_alphabet(a):
    f = [n for n,m in _ALPHABETS.items() if m==a]
    if f!=[]: return '\033[31m{}\033[0m'.format(''.join(f))

    a = a.replace('ABCDEFGHIJKLMNOPQRSTUVWXYZ',_ALPHA_AZ)
    a = a.replace('0123456789', _ALPHA_09)
    a = a.replace('abcdefghijklmnopqrstuvwxyz',_ALPHA_az)
    return a

# PRINTABLES DEFINES

_pre = '\033[100m \033[0m'
_pre_L = '\033[7m \033[0m'
_pre_succ = '\033[42m \033[0m'
_pre_fail = '\033[41m \033[0m'

def putline(str):
    _put(str, _pre)
def putLine(str):
    _put(str, _pre_L)
def _put(str, pre):
    print('{} {}'.format(pre, str))

def putErr(str):
    _put(str, _pre_fail)
def putSucc(str):
    _put(str, _pre_succ)

def putSep(sz = 80):
    assert type(sz) is IntType, "sz is not an integer: %r" % sz
    print('\033[90m{}\033[0m\n'.format('▄'*sz))

# ALGORYTHM

_CIPHERS = ['CHAR','E2']
_TRANSPORT = ['RAW','PaK']

def enc(kp, text):
    n, key = kp
    return [pow(ord(c),key,n) for c in text]

def enc2(kp, text, sz):
    text += ' '
    n, key = kp
    return [[int(i/sz**2%sz),int(i/sz%sz),i%sz] for i in [pow(i[0]*sz+i[1],key,n) for i in zip(text[0::2],text[1::2])]]
    # return [(int(i/sz**2%sz),int(i/sz%sz),i%sz) for i in [pow(i[0]*sz+i[1],key,n) for i in zip(text[0::2],text[1::2])]]

def dec(kp, enc):
    n, key = kp
    return [chr(pow(c, key, n)) for c in enc]

def dec2(kp, enc, sz):
    n, key = kp
    return [(i//sz,i%sz) for i in [pow(e[0]*sz**2+e[1]*sz+e[2],key,n) for e in enc]]
    # return [(i//sz**2,i//sz,i%sz) for i in [pow(e[0]*sz**2+e[1]*sz+e[2],key,n) for e in enc]]

def pak(s):
    return ''.join([chr(i) for l in s for i in l])

def unpak(s):
    t = [ord(j) for l in s for j in l]
    return [i for i in zip(t[0::3],t[1::3],t[2::3])]

def uncharize(s):
    return [ord(i) for i in s]

def charize(s):
    return ' '.join(["{}".format(i) for l in s for i in l])

def decode(s, dic):
    return ''.join([dic[i] for l in s for i in l])

# Small numbers

def Factorization(n):
    c = 2
    f = []
    while(c*c<=n):
        if n % c:
            c += 1
        else:
            n //= c
            f.append(c)
    if n > 1:
        f.append(n)
    return f

# Tests

import random

p = 96
q = 73

def gcd(a, b):
    return a if b==0 else gcd(b, a%b)

def isPrime(n):
    if(n<=1): return False
    if(n<=3): return True
    if(n%2==0 or n%3==0): return False
    c = 5
    while(c*c<=n):
        if(n%c==0 or n%(c+2)==0): return False
        c+= 6
    return True

def egcd(a, m):
    if a == 0:
        return (m, 0, 1)
    else:
        g, y, x = egcd(m % a, a)
        return (g, x - (m // a) * y, y)

def modinv(a, m):
    g, x, y = egcd(a, m)
    return x%m if g==1 else -1

def findModInverse(a, m):
    if gcd(a, m) != 1:
        return None
    u1, u2, u3 = 1, 0, a
    v1, v2, v3 = 0, 1, m

    while v3 != 0:
        q = u3 // v3
        v1, v2, v3, u1, u2, u3 = (u1 - q * v1), (u2 - q * v2), (u3 - q * v3), v1, v2, v3
    return u1 % m

def genkey(a, b):
    r = a
    r2 = b
    u = 1
    v = 0
    u2 = 0
    v2 = 1
    while(r2 != 0):
        q = r//r2
        rs = r
        us = u
        vs = v
        r = r2
        u = u2
        v = v2
        r2 = rs-q*r2
        u2 = us-q*u2
        v2 = vs-q*v2
    return r, u, v

def genPrime(keysize = 1024):
    while True:
        n = random.randrange(2**(keysize-1), 2**(keysize))
        if isPrime(n):
            return n

def kg(p, q):
    if(not (isPrime(p) and isPrime(q))): return -1
    if(p==q): return -2
    n = p*q
    phi = (p-1)*(q-1)
    m = max(p,q)
    m+=1
    if(m%2==0): m+=1
    e = random.randrange(m, phi, 2)
    g = gcd(e, phi)
    while g!= 1:
        e = random.randrange(m, phi, 2)
        g = gcd(e, phi)
    d = modinv(e, phi)
    pub = (n, e)
    prv = (n, d)
    return (pub, prv)

def myap3(s):
     return [i for i in range(3,s+1,2) if isPrime(i)]

def myap3m(s, e):
     return [i for i in range(s,e+1,2) if not i%3==0 and isPrime(i)]

def pickRandomsG(s, e, cnt):
    gen = myap3m(s,e)
    r = []
    for i in range(0,cnt):
        r += [random.choice(gen)]
    return r
#
# Gen primes :
# x = range(3,s+1,2)
# x =
# x.add(2)
# x.add(5)

# Tests
tg0 = gcd(3696, 2871)
tp0 = isPrime(11)
tp1 = isPrime(85674)
expct = 33
s = "PGCD : {}".format(tg0)
putSucc(s) if tg0==expct else putErr(s)
tp =[tp0, tp1]

expct = [True, False]

s = "Prime : {}".format(tp)
putSucc(s) if tp==expct else putErr(s)

p = 73
q = 97

expct = 56

#print(genkey(p,q))
m1 = modinv(23, 99)
m2 = findModInverse(23,99)

s = "Mod inverse : {} ({})".format(m1, 'modinv')
putSucc(s) if m1==expct else putErr(s)
s = "Mod inverse : {} ({})".format(m2, 'findModInverse')
putSucc(s) if m2==expct else putErr(s)

r = pickRandomsG(3367,7081,6)
s = "Random Primes : {} ".format(r)
putSucc(s) if False not in [isPrime(i) for i in r] else putErr(s)

kp = kg(11,23)
s = "Key pairs : {} ".format(kp)
putSucc(s) if (kp[0][0]==kp[1][0] and kp[0][1]!=kp[1][1]) else putErr(s)

r = myap3(42)
s = "Primes list : {} ".format(r)
putSucc(s) if False not in [isPrime(i) for i in r] else putErr(s)


# for i in range(0,10):
#     print(genPrime(32))

kp1, kp2 = kg(p, q)
s = "Key pairs : {} and {} ".format(kp1, kp2)
putSucc(s) if (kp1[0]==kp2[0] and kp1[1]!=kp2[1]) else putErr(s)

text = "Khoo !"
enc = enc(kp1, text)
putline("gen > {}".format(" ".join("{}".format(str(c)) for c in enc)))
dec = dec(kp2, enc)

s = "dec > {}".format(''.join(dec))
putSucc(s) if (''.join(dec)==text) else putErr(s)

# enc = enc2(kp1, text, 2**8)
# print(enc)
# print(dec2(kp2, enc, 2**8))

putSep()

dic = [i for i in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ .?€0123456789']
tab = dict((key, value) for (key, value) in [(dic[i],i) for i in range(0,len(dic))])

text = [2,4,17,8]
kp1 = (689,5)
sz = 26
enc = enc2(kp1, text, sz)
exp = [[0, 11, 10], [1, 0, 8]]
raw = pak(enc)
s = "gen > {}".format(" ".join("{:02x}".format(ord(c)) for c in raw))
putSucc(s) if enc==exp else putErr(s)
#print("gen >"," ".join("{:02x}".format(ord(c)) for c in raw),"({})".format("Valid"if(enc==exp) else "Invalid"))
# print(" ".join("{:02x}".format(ord(c)) for c in raw))
kp2 = (689,125)
unR = unpak(raw)
# print(unR)
dec = dec2(kp2, unR, sz)
exp = [(2, 4), (17, 8)]
# print("dec > {} ({})".format(dec,"Valid"if(dec==exp) else "Invalid"))
# print(decode(dec, dic))
msg = 'CERI'
dec = decode(dec, dic)
s = "dec > {}".format(dec)
putSucc(s) if dec==msg else putErr(s)

putSep()

_ALPHA = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz .?€0123456789♥<>'
dic = [i for i in _ALPHA]
tab = dict((key, value) for (key, value) in [(dic[i],i) for i in range(0,len(dic))])

msg = "Ohayo goshujinsama ♥"

text = [tab[i] for i in msg]
kp1 = (7081, 239)
sz = len(_ALPHA)
enc = enc2(kp1, text, sz)
raw = pak(enc)
putline("gen > {}".format(" ".join("{:02x}".format(ord(c)) for c in raw)))
kp2 = (7081, 4367)
unR = unpak(raw)
dec = dec2(kp2, unR, sz)
dec = decode(dec, dic)

s = "dec > {}".format(dec)
putSucc(s) if dec==msg else putErr(s)


putSep()

_ALPHA = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ .?€0123456789'
dic = [i for i in _ALPHA]
tab = dict((key, value) for (key, value) in [(dic[i],i) for i in range(0,len(dic))])

msg = 'ENVOYEZ 2500€.'
text = [tab[i] for i in msg]

#print(decode([text], dic))
#print(text)
kp1 = (7081, 3367)
kp2 = (7081, 3223)
sz = len(_ALPHA)
enc = enc2(kp1, text, sz)
# print("gen > {}".format(enc))
raw = pak(enc)
putline("gen > {}".format(" ".join("{:02x}".format(ord(c)) for c in raw)))
unR = unpak(raw)
# print(unR)
dec = dec2(kp2, unR, sz)
dec = decode(dec, dic)

s = "dec > {}".format(dec)
putSucc(s) if dec==msg else putErr(s)

putSep()

n = 13289

f1 = Factorization(n)
expct=[97, 137]
s = "fact > {}".format(f1)
putSucc(s) if f1==expct else putErr(s)

f2 = Factorization(253)
expct=[11, 23]
s = "fact > {}".format(f2)
putSucc(s) if f2==expct else putErr(s)

phi = reduce((lambda x,y: x*y), [i-1 for i in f1])
expct=13056
s = "φ > {}".format(phi)
putSucc(s) if phi==expct else putErr(s)

e = 12413

tg0 = gcd(phi, e)
s = "PGCD : {}".format(tg0)
putSucc(s) if tg0==1 else putErr(s)

print(egcd(e, expct))
print(modinv(12413, 13056))

def decrypt(p, q, e, n, ct):
    phi = (p-1) * (q-1)
    gcd, a, b = egcd(e, phi)
    d = a
    if d < 0:
        d += phi
    pt = pow(ct, d, n)
    return pt

m1 = [9197, 6284, 12836, 8709, 4584, 10239, 11553, 4584, 7008, 12523, 9862, 356, 5356, 1159, 10280, 12523, 7506, 6311]
p,q = 97, 137
n = 13289

print("dec > {}".format(' '.join(["{}".format(decrypt(p, q, e, n, m)) for m in m1 ])))

n = 755918011
e = 163119273
m2 = [671828605,407505023,288441355,679172842,180261802]

f2 = Factorization(n)
expct=[27449, 27539]
s = "fact > {}".format(f2)
putSucc(s) if f2==expct else putErr(s)

phi = reduce((lambda x,y: x*y), [i-1 for i in f2])
expct=755863024
s = "φ > {}".format(phi)
putSucc(s) if phi==expct else putErr(s)

tg0 = gcd(phi, e)
s = "PGCD : {}".format(tg0)
putSucc(s) if tg0==1 else putErr(s)

print("dec > {}".format(' '.join(["{}".format(decrypt(p, q, e, n, m)) for m in m2 ])))

putSep()


t = 12345
n,e = 45,123

#enc(kp1, text)

def sequify(l, sz):
    return [[int(i/sz%sz),i%sz] for i in l]

# m1 = [9197, 6284, 12836, 8709, 4584, 10239, 11553, 4584, 7008, 12523, 9862, 356, 5356, 1159, 10280, 12523, 7506, 6311]
# kp1 = [13289, 12413]


#_ALPHA = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
# _ALPHA = '                                 !"#$%&\'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\\]^_`abcdefghijklmnopqrstuvwxy'
#_ALPHA = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
# sz = len(_ALPHA)
# dic = [i for i in _ALPHA]
# tab = dict((key, value) for (key, value) in [(dic[i],i) for i in range(0,len(dic))])

# enc = sequify(m1, sz)
# print(enc)
# raw = pak(enc)
# print("gen >"," ".join("{:02x}".format(ord(c)) for c in raw))
# unR = unpak(raw)
# print(unR)
# dec = dec2(kp2, unR, sz)
# print("dec > ",dec)
# dec = decode(dec, dic)
# print("dec > {}".format(dec))
#dec = dec2(kp1, m1, 26)
#print(dec)
#
#

# Base58 key share
sz = len(_BASE58)
dic = [i for i in _BASE58]
tab = dict((key, value) for (key, value) in [(dic[i],i) for i in range(0,len(dic))])

#print(dec(kp1, m1))
#

# PRINTABLE SECTION

#_ALPHA = _BASE58

pkey = kp1[0]
print("")

putLine("Your public key is : \033[1m\033[7m {} \033[0m".format(pkey))

a = format_alphabet(_ALPHA)
putLine("Used alphabet is \033[7m{} \033[0m({})".format(a, len(_ALPHA)))

cip = 'E2'
putLine("Cipher used is \033[4m{}\033[24m".format(cip))

met = 'PaK'
putLine("Transport method used is \033[4m{}\033[24m".format(met))

import pyqrcode
pub = pyqrcode.create(pkey)
print(pub.terminal(quiet_zone=1))
