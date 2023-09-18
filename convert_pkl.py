import pickle
import json
import sys
import os

# folder = str(sys.argv[1])
#
# with open(os.path.join(folder, 'projections', 'local_prototypes', 'local_projection.pkl'),'rb') as f:
#     lp = pickle.load(f)
# json = json.dumps(lp)
# with open(os.path.join(folder, 'projections', 'local_prototypes', 'local_projection.json'), 'w') as f:
# 	f.write(json)

folder, file = os.path.split(str(sys.argv[1]))

with open(os.path.join(folder, file), 'rb') as f:
    lp = pickle.load(f)
json = json.dumps(lp)
with open(os.path.join(folder, file.split('.')[0] + '.json'), 'w') as f:
    f.write(json)
