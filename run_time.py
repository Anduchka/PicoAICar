from math import tanh
from policy_export import W1, B1, W2, B2, HID, OBS_DIM

def mlp_act(obs):
    h = [0.0]*HID
    
    for i in range(HID):
        s = B1[i]
        row = W1[i]
        for j in range(OBS_DIM):
            s += row[j] * obs[j]
        h[i] = tanh(s)
    
    out = [0.0, 0.0]
    
    for k in range(2):
        s = B2[k]
        row = W2[k]
        for i in range(HID):
            s += row[i] * h[i]
        out[k] = tanh(s)
    
    return out[0], out[1]