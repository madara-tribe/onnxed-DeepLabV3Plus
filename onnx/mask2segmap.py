import numpy as np

colorB = [0, 0, 255, 255, 69]
colorG = [0, 0, 255, 0, 47]
colorR = [0, 255, 0, 0, 142]
CLS=5
CLASS_COLOR = list()
for i in range(0, CLS):
    CLASS_COLOR.append([colorR[i], colorG[i], colorB[i]])
COLORS = np.array(CLASS_COLOR, dtype="float32")

def give_color_to_seg_img(seg, n_classes):
    if len(seg.shape)==3:
        seg = seg[:,:,0]
    seg_img = np.zeros( (seg.shape[0],seg.shape[1],3) ).astype('float')
    #colors = sns.color_palette("hls", n_classes) #DB
    colors = COLORS #DB
    for c in range(n_classes):
        segc = (seg == c)
        seg_img[:,:,0] += (segc*( colors[c][0]/255.0 ))
        seg_img[:,:,1] += (segc*( colors[c][1]/255.0 ))
        seg_img[:,:,2] += (segc*( colors[c][2]/255.0 ))

    return seg_img

