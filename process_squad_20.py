import io
import json
import numpy as np
import os
import spacy


nlp = spacy.load("en_core_web_lg")

pth = os.path.join(".","dev-v2.0.json")

f = open(pth, "r")

js = json.load(f)

cntxt = np.zeros((500,4), dtype='uint64')

ques = np.zeros((20, 4), dtype='uint64')

imp = np.zeros((1,), dtype='uint64')

ans = np.zeros((10, 4), dtype='uint64')


idx = 0

x = None
y = None


for jsd in js["data"][0]["paragraphs"]:

    para = nlp(jsd["context"])

    # for t in para:
    #     print(t.i, t.head.i, t.pos, t.norm)
   
    idx = 0
    for t in para:
        cntxt[idx] = [t.i, t.head.i, t.pos, t.norm]
        idx +=1

    for jsd2 in jsd["qas"]:
        qu = nlp(jsd2["question"])

        idx=0
        for t in qu:
            ques[idx] = [t.i, t.head.i, t.pos, t.norm]
            idx += 1

        if jsd2["is_impossible"] == True:
            imp[0] = 0

        else:
            imp[0] = 1


    if y is None:
        x = np.array([cntxt])
        y = np.array([[imp, ans]])
    else:
        x = np.concatenate((x, [ cntxt    ]))
        y = np.concatenate((y, [[imp, ans]]))

    # print("--------------------")
    # print(y)
    # print("--------------------")

