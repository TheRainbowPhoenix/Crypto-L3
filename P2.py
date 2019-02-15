# -*- coding: utf-8 -*-

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
    e = random.randrange(m+1, phi)
    g = gcd(e, phi)
    while g!= 1:
        e = random.randrange(m+1, phi)
        g = gcd(e, phi)
    d = modinv(e, phi)
    pub = (n, e)
    prv = (n, d)
    return (pub, prv)

def keygen(p, q):
    if(not (isPrime(p) and isPrime(q))): return -1
    if(p==q): return -2
    n=p*q
    phi = (p-1) * (q-1)
    e = random.randrange(1, phi)
    g = gcd(e, phi)
    while g!=1:
        e = random.randrange(1, phi)
        g = gcd(e, phi)
    d = 0
    # d = rev(e, phi)
    # e,n public
    # d,n private
    return (n,e,d)

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

# Tests
tg0 = gcd(3696, 2871)
tp0 = isPrime(11)
tp1 = isPrime(85674)
print("PGCD : {} (Expected 33)".format(tg0))
tp = "{}".format([tp0, tp1])
print("Prime : {}".format(tp))
print("PGCD : {} (Expected 33)".format(tg0))

p = 73
q = 97

print(genkey(p,q))
print(keygen(p,q))
print(modinv(23, 99))
print(findModInverse(23,99))

# for i in range(0,10):
#     print(genPrime(32))

kp1, kp2 = kg(p, q)
print(kp1, kp2)
text = "Khoo !"
enc = enc(kp1, text)
print(enc)
print(dec(kp2, enc))

# enc = enc2(kp1, text, 2**8)
# print(enc)
# print(dec2(kp2, enc, 2**8))

print("======================== ")

dic = [i for i in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ .?€0123456789']
tab = dict((key, value) for (key, value) in [(dic[i],i) for i in range(0,len(dic))])

text = [2,4,17,8]
kp1 = (689,5)
sz = 26
enc = enc2(kp1, text, sz)
exp = [[0, 11, 10], [1, 0, 8]]
raw = pak(enc)
print("gen >"," ".join("{:02x}".format(ord(c)) for c in raw),"({})".format("Valid"if(enc==exp) else "Invalid"))
print(" ".join("{:02x}".format(ord(c)) for c in raw))
kp2 = (689,125)
unR = unpak(raw)
print(unR)
dec = dec2(kp2, unR, sz)
exp = [(2, 4), (17, 8)]
print("dec > {} ({})".format(dec,"Valid"if(dec==exp) else "Invalid"))
print(decode(dec, dic))

print("======================== ")

_ALPHA = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz .?€0123456789♥<>'
dic = [i for i in _ALPHA]
tab = dict((key, value) for (key, value) in [(dic[i],i) for i in range(0,len(dic))])

msg = "Ohayo goshujinsama ♥"

text = [tab[i] for i in msg]
kp1 = (7081, 239)
sz = len(_ALPHA)
enc = enc2(kp1, text, sz)
raw = pak(enc)
print("gen >"," ".join("{:02x}".format(ord(c)) for c in raw))
kp2 = (7081, 4367)
unR = unpak(raw)
dec = dec2(kp2, unR, sz)
dec = decode(dec, dic)
print("dec > {} ({})".format(dec,"Valid"if(dec==msg) else "Invalid"))


print("======================== ")

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
#print("gen > {}".format(enc))
raw = pak(enc)
print("gen >"," ".join("{:02x}".format(ord(c)) for c in raw))
unR = unpak(raw)
#print(unR)
dec = dec2(kp2, unR, sz)
dec = decode(dec, dic)
print("dec > {} ({})".format(dec,"Valid"if(dec==msg) else "Invalid"))


t = 12345
n,e = 45,123

#enc(kp1, text)

# print(kg(11,23))

#m1 = [9197, 6284, 12836, 8709, 4584, 10239, 11553, 4584, 7008, 12523, 9862, 356, 5356, 1159, 10280, 12523, 7506, 6311]
#kp1 = [13289, 12413]

#print(dec(kp1, m1))
