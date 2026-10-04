from pathlib import Path
from PIL import Image
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Arc, Rectangle

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / 'figures'
FIG.mkdir(exist_ok=True)

mpl.rcParams.update({'font.family': 'DejaVu Sans', 'pdf.fonttype': 42, 'ps.fonttype': 42})

# IEEE graphical abstract: exact 660 x 295 px.
fig = plt.figure(figsize=(6.6, 2.95), dpi=100, facecolor='white')
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 660); ax.set_ylim(0, 295); ax.axis('off')

panels = [
    (12, 20, 188, 258, '#F2F5F7'),
    (220, 20, 208, 258, '#EEF5FB'),
    (448, 20, 200, 258, '#EEF8F1'),
]
for x, y, w, h, fc in panels:
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                                boxstyle='round,pad=2,rounding_size=7',
                                linewidth=1.1, edgecolor='#555555', facecolor=fc))

# --- Panel 1: deployment shift ---
ax.text(106, 260, 'Sparse IMU', ha='center', va='center', fontsize=12.5, weight='bold')
ax.text(106, 243, 'gait sensing', ha='center', va='center', fontsize=12.5, weight='bold')
# limb + two adjacent sensors
ax.plot([62, 88, 105], [214, 165, 111], lw=4, color='#444444', solid_capstyle='round')
ax.scatter([62, 88, 105], [214, 165, 111], s=20, color='#444444', zorder=3)
for x, y in [(75, 190), (97, 136)]:
    ax.add_patch(Rectangle((x-13, y-7), 26, 14, facecolor='white', edgecolor='#333333', lw=0.8))
# local frame + rotation
cx, cy = 148, 166
ax.arrow(cx, cy, 25, 0, head_width=4, head_length=5, length_includes_head=True, color='#333333', lw=1.1)
ax.arrow(cx, cy, 0, 25, head_width=4, head_length=5, length_includes_head=True, color='#333333', lw=1.1)
ax.add_patch(Arc((cx, cy), 50, 50, theta1=15, theta2=110, lw=1.6, color='#555555'))
ax.text(106, 82, 'Sensor-frame rotation', ha='center', fontsize=10.8, weight='bold')
ax.text(106, 64, 'changes axis channels', ha='center', fontsize=10.2)
ax.text(106, 43, '0-15 deg stress test', ha='center', fontsize=10.2)

# --- Panel 2: virtual benchmark ---
ax.text(324, 260, 'Virtual benchmark', ha='center', va='center', fontsize=12.5, weight='bold')
ax.text(324, 243, '10-deg rotation', ha='center', va='center', fontsize=11.5, weight='bold')
# horizontal result bands
rows = [
    ('Unaugmented', '8.08-8.24 deg RMSE', '#F4E6E6'),
    ('10-deg augmentation', '3.66-4.27 deg RMSE', '#E4F0FA'),
    ('Frame registration', '3.19-3.92 deg RMSE', '#E7F4EA'),
]
ys = [207, 153, 99]
for (lab, val, fc), y in zip(rows, ys):
    ax.add_patch(FancyBboxPatch((238, y-19), 172, 40, boxstyle='round,pad=1.5,rounding_size=4',
                                linewidth=0.8, edgecolor='#666666', facecolor=fc))
    ax.text(324, y+7, lab, ha='center', fontsize=10.5, weight='bold')
    ax.text(324, y-8, val, ha='center', fontsize=10.3)
ax.text(324, 58, 'Augmentation passes', ha='center', fontsize=9.8, weight='bold')
ax.text(324, 43, 'the 10-deg gate', ha='center', fontsize=9.8, weight='bold')
ax.text(324, 28, 'No cap passes 12.5/15 deg', ha='center', fontsize=9.2)

# --- Panel 3: physical-IMU external test ---
ax.text(548, 260, 'Physical-IMU test', ha='center', va='center', fontsize=12.5, weight='bold')
ax.text(548, 243, '10-deg rotation', ha='center', va='center', fontsize=11.5, weight='bold')
checks = [
    ('Frame rotation', 'degrades 4/4 endpoints'),
    ('Augmentation', 'improves 3/4 endpoints'),
    ('Registration', 'improves 4/4 endpoints'),
]
ys = [201, 148, 95]
for (lab, val), y in zip(checks, ys):
    ax.text(548, y+7, lab, ha='center', fontsize=11.0, weight='bold')
    ax.text(548, y-10, val, ha='center', fontsize=10.5)
ax.text(548, 56, 'Physical-IMU signals;', ha='center', fontsize=9.2)
ax.text(548, 42, 'frame rotations imposed', ha='center', fontsize=9.0)
ax.text(548, 28, 'mathematically', ha='center', fontsize=9.0)

# arrows between panels
for x1, x2 in [(202, 218), (430, 446)]:
    ax.add_patch(FancyArrowPatch((x1, 148), (x2, 148), arrowstyle='-|>',
                                 mutation_scale=10, lw=1.2, color='#444444'))

png = FIG / 'graphical_abstract.png'
pdf = FIG / 'graphical_abstract.pdf'
fig.savefig(png, dpi=100, facecolor='white')
fig.savefig(pdf, facecolor='white')
plt.close(fig)

# Palette PNG keeps the exact pixel dimensions and is comfortably under IEEE's recommended 45 KB.
im = Image.open(png).convert('RGB').quantize(colors=64, method=Image.Quantize.MEDIANCUT)
im.save(png, optimize=True, dpi=(300, 300))
im.save(FIG / 'gagraphic.png', optimize=True, dpi=(300, 300))

print(f'Wrote {png} and {pdf}')
print(f'PNG dimensions: {im.size[0]} x {im.size[1]}')
print(f'PNG bytes: {png.stat().st_size}')
