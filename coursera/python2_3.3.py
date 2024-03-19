# -*- coding: utf-8 -*-
"""
Created on Mon Mar 11 14:51:34 2024

@author: chewei

python practice : 3.3
zip, map and lambda
"""

#%% Taiwan ID Checksum

def cksum_twid(idstr):
    """"compute checksum for Taiwan ID ;
        use tuple for data that will not be changed"""
    code1 = ord(idstr[0])       # convert first English character to two-digit number
    cmap = (10, 11, 12, 13, 14, 15,
            16, 17, 34, 18, 19, 20,
            21, 22, 35, 23, 24, 25,
            26, 27, 28, 29, 32, 30, 31, 33)     # turple
    num1 = cmap[code1-65]        # ASCII: A=65 -> 65-65=0 -> camp(0)=10 -> A=10
    newid = str(num1) + idstr[1:]
    weight = (1, 9, 8, 7, 6, 5, 4, 3, 2, 1, 1)
    
    checksum = 0
    for i in range(0, 11):
        checksum += weight[i] * int(newid[i])
    print(f"checksum = {checksum}")
    
id = 'A123456789'
cksum_twid(id)

#%% zip two varialbes

newid = '10123456789'
weight = (1, 9, 8, 7, 6, 5, 4, 3, 2, 1, 1)

"""將newid和weight兩兩配對"""
for apair in zip(newid, weight):
    print(apair)
    
#%% Taiwan ID Checksum + zip

def cksum_twid(idstr):
    """"compute checksum for Taiwan ID ;
        use tuple for data that will not be changed"""
    code1 = ord(idstr[0])       # convert first English character to two-digit number
    cmap = (10, 11, 12, 13, 14, 15,
            16, 17, 34, 18, 19, 20,
            21, 22, 35, 23, 24, 25,
            26, 27, 28, 29, 32, 30, 31, 33)     # turple
    num1 = cmap[code1-65]        # ASCII: A=65 -> 65-65=0 -> camp(0)=10 -> A=10
    newid = str(num1) + idstr[1:]
    weight = (1, 9, 8, 7, 6, 5, 4, 3, 2, 1, 1)
    
    checksum = 0
    for apair in zip(newid, weight):
        checksum += int(apair[0]) * apair[1]
    print(f"checksum = {checksum}")
    
id = 'A123456789'
cksum_twid(id)
    
#%% the lambda operator

def f1(x):
    return x**2
print(f1(8))

f2 = lambda x: x**2
print(f2(8))

#%% the map operator

list1 = [3, 5, 1.2, 4, 9]
out1 = map(f1, list1)
print(list(out1))

# using lambda
out2 = map(lambda x: x**2, list1)
print(list(out2))

#%% Taiwan ID Checksum + zip + lambda + map

def cksum_twid(idstr):
    """"compute checksum for Taiwan ID ;
        use tuple for data that will not be changed"""
    code1 = ord(idstr[0])       # convert first English character to two-digit number
    cmap = (10, 11, 12, 13, 14, 15,
            16, 17, 34, 18, 19, 20,
            21, 22, 35, 23, 24, 25,
            26, 27, 28, 29, 32, 30, 31, 33)     # turple
    num1 = cmap[code1-65]        # ASCII: A=65 -> 65-65=0 -> camp(0)=10 -> A=10
    newid = str(num1) + idstr[1:]
    weight = (1, 9, 8, 7, 6, 5, 4, 3, 2, 1, 1)
    
    out1 = map(lambda apair: int(apair[0]) * apair[1], 
               zip(newid, weight))
    '''' 先利用 zip 將 (newid, weight) 進行配對，'''
    
    checksum = sum(out1)
    print(f"checksum = {checksum}")
    
id = 'A123456789'
cksum_twid(id)