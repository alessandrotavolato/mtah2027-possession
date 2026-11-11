"""Export video observations only when a validated recovery gate permits it.

An all-fail report creates an empty frame,x_m,y_m,confidence CSV: downstream
fusion must use no video-derived positions. A pass requires an implemented,
independently validated real-gap export; this script refuses to invent one.
"""
import argparse,csv,json
from pathlib import Path
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--report',required=True,help='video recovery report JSON')
p.add_argument('--out',required=True,help='output CSV path')
a=p.parse_args(); r=json.loads(Path(a.report).read_text())
if any(d.get('gate') == 'PASS' for d in r['durations'].values()):
 raise SystemExit('A duration passed; do not export without independent validation and real-gap tracking implementation.')
with open(a.out,'w',newline='') as f:
 csv.writer(f).writerow(['frame','x_m','y_m','confidence'])
print('All durations failed; no video-derived observations exported:',a.out)
