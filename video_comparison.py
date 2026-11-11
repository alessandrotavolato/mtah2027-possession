"""Make a local-only three-panel diagnostic after video-gap prediction.
Video frames contain licensed content; do not publish this diagnostic.
"""
import argparse,json
from pathlib import Path
import cv2
import numpy as np
from video_gap_experiment import load_ball,load,select_gaps,estimate,project,PITCH,FPS

p=argparse.ArgumentParser();p.add_argument('--video',required=True);p.add_argument('--report',required=True);p.add_argument('--out',required=True);a=p.parse_args()
r=json.loads(Path(a.report).read_text()); ball=load_ball(); H,_,_=load();cap=cv2.VideoCapture(a.video)
for sec in (3,1,6):
 d=r['durations'].get(str(sec),{});threshold=d.get('agreement_threshold_m');
 if threshold is None or not d.get('held_accepted'): continue
 for start in select_gaps(ball,sec*FPS)[1]:
  x=estimate(cap,H,ball,start,sec*FPS)
  if x is not None and x['agreement_m']<=threshold: break
 else: continue
 break
else: raise SystemExit('No accepted held-out examples')
i=x['width']//2;frame=x['start']+i
cap.set(cv2.CAP_PROP_POS_FRAMES,frame);ok,image=cap.read()
if not ok:raise SystemExit('Could not read frame')
coords=[ball[frame],x['interp'][i],x['recovered'][i]]
pixels=[project(H[frame],xy) for xy in coords]
center=pixels[0];h,w=image.shape[:2];radius=90
cx,cy=np.rint(center).astype(int);cx=np.clip(cx,radius,w-radius);cy=np.clip(cy,radius,h-radius)
x0,y0=cx-radius,cy-radius
base=image[y0:y0+2*radius,x0:x0+2*radius]
colors=[(255,255,255),(0,220,255),(255,150,0)]
titles=['Observed tracking','Linear interpolation','Video estimate']
panels=[]
for title,xy,pt,col in zip(titles,coords,pixels,colors):
 panel=cv2.resize(base,(360,360),interpolation=cv2.INTER_LINEAR)
 q=np.rint((pt-[x0,y0])*2).astype(int)
 if 0<=q[0]<360 and 0<=q[1]<360:cv2.circle(panel,tuple(q),12,col,3)
 else:cv2.putText(panel,'outside crop',(8,40),cv2.FONT_HERSHEY_SIMPLEX,.65,col,2)
 err=float(np.linalg.norm((xy-ball[frame])*PITCH))
 canvas=np.zeros((435,360,3),dtype=np.uint8);canvas[40:400]=panel
 cv2.putText(canvas,title,(10,26),cv2.FONT_HERSHEY_SIMPLEX,.65,(255,255,255),2)
 cv2.putText(canvas,'Error vs observed: %.2f m'%err,(10,425),cv2.FONT_HERSHEY_SIMPLEX,.55,(255,255,255),1)
 panels.append(canvas)
cv2.imwrite(a.out,np.hstack(panels));cap.release()
print('Example is first accepted 3-second holdout when available; all panels share one frame and crop. Original tracking is a noisy reference, not ground truth. frame',frame)
