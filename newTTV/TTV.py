#!/home/mwh38/personality-machine/venv/bin/python3
from cgi import test
import sys
import pandas as pd
from collections import Counter
import random

files = sys.argv[1:]
df = pd.DataFrame({"path":files})
df["oldType"],df["file"]=zip(*[p.split("/")[-2:] for p in df["path"]])
df["prefix"],df["no"],_ = zip(*[f.split(".") for f in df["file"]])
nTest,nTrain,nVal = [sum(df["oldType"]==x) for x in ["test","train","val"]]
print(f"original: train {nTrain}  val {nVal}  test {nTest}")

prefixes = list(dict(Counter(df["prefix"])).items())
random.seed(15032002)
random.shuffle(prefixes)

trainPrefixes = []
trainCount = 0
valPrefixes = []
valCount = 0
i = 0

while trainCount < nTrain:
    trainPrefixes.append(prefixes[i][0])
    trainCount += prefixes[i][1]
    i += 1

while valCount < nVal:
    valPrefixes.append(prefixes[i][0])
    valCount += prefixes[i][1]
    i += 1

testPrefixes,counts = zip(*prefixes[i:])
testCount = sum(counts)

print(f"remapped: train {trainCount}  val {valCount}  test  {testCount}")
assert(trainCount+testCount+valCount==len(files))
assert(len(
    set(testPrefixes)
    .union(set(trainPrefixes))
    .union(set(valPrefixes))
    .symmetric_difference(set(df["prefix"]))
    ) == 0 )

def newType(prefix):
    if prefix in trainPrefixes:
        return "train"
    elif prefix in testPrefixes:
        return "test"
    else:
        assert(prefix in valPrefixes)
        return "val"

df["newType"] = df["prefix"].map(newType) 
'''pathPrefix = "/rds/user/mwh38/hpc-work/personality-machine/firstimpressions/"
df["path"] = [x[len(pathPrefix):] for x in df["path"]]'''

df.to_csv("TTVSplitRemapping.csv")