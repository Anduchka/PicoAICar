import numpy as np
import json, math, sys

fn = sys.argv[1] if len(sys.argv) > 1 else "best_policy_gen000.npz"
z = np.load(fn)

w1 = z["w1"].astype(np.float32)
b1 = z["b1"].astype(np.float32)
w2 = z["w2"].astype(np.float32)
b2 = z["b2"].astype(np.float32)

with open("policy_export.py", "w") as f:
    f.write("# Auto-generated from %s\n" % fn)
    f.write("OBS_DIM = %d\n" % w1.shape[1])
    f.write("HID = %d\n" % w1.shape[0])
    f.write("W1 = %s\n" % w1.tolist())
    f.write("B1 = %s\n" % b1.tolist())
    f.write("W2 = %s\n" % w2.tolist())
    f.write("B2 = %s\n" % b2.tolist())
    
print("Wrote policy_export.py")