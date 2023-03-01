import numpy as np
import json

src1 = "transforms.json"
src2 = "extrinsic.npy"

f = open(src1)
data = json.load(f)

ext = np.load(src2, allow_pickle=True)

print(ext.shape)

for i, d in enumerate(data['frames']):
    idx = int(d['file_path'].split("/")[-1].split("_")[0].split("m")[-1])
    tM = np.array(d['transform_matrix'])
    tM[:3,:4] = ext[idx]
    data['frames'][i]['transform_matrix'] = tM.tolist()

jdata = json.dumps(data)
# Writing to sample.json
with open("transforms_merge.json", "w") as outfile:
    outfile.write(jdata)
