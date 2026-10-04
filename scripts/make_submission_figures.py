from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle

ROOT = Path(__file__).resolve().parents[1]
SD = ROOT / 'source_data'
FIG = ROOT / 'figures'
FIG.mkdir(exist_ok=True)

mpl.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size': 9.5,
    'axes.titlesize': 10.5,
    'axes.labelsize': 9.5,
    'legend.fontsize': 8.5,
    'xtick.labelsize': 8.5,
    'ytick.labelsize': 8.5,
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
    'axes.spines.top': False,
    'axes.spines.right': False,
})

COLORS = {
    'Unaugmented': '#4d4d4d',
    '10-deg augmentation': '#0072B2',
    'Template registration': '#D55E00',
    'Gait-functional frame': '#009E73',
}
METHOD_MARKERS = {
    'Unaugmented': 'o',
    '10-deg augmentation': 's',
    'Template registration': 'D',
    'Gait-functional frame': '^',
}
ENDPOINT_COLORS = {
    'ankle_dorsiflex_deg': '#0072B2',
    'ankle_plantarflex_deg': '#56B4E9',
    'knee_stance_flex_deg': '#D55E00',
    'knee_swing_flex_deg': '#CC79A7',
}
ENDPOINT_MARKERS = {
    'ankle_dorsiflex_deg': 'o',
    'ankle_plantarflex_deg': 's',
    'knee_stance_flex_deg': 'D',
    'knee_swing_flex_deg': '^',
}
ENDPOINT_LINESTYLES = {
    'ankle_dorsiflex_deg': '-',
    'ankle_plantarflex_deg': '--',
    'knee_stance_flex_deg': '-.',
    'knee_swing_flex_deg': ':',
}
ENDPOINT_LABEL = {
    'ankle_dorsiflex_deg': 'Ankle\ndorsiflexion',
    'ankle_plantarflex_deg': 'Ankle\nplantarflexion',
    'knee_stance_flex_deg': 'Knee stance\nflexion',
    'knee_swing_flex_deg': 'Knee swing\nflexion',
}
ENDPOINT_ORDER = list(ENDPOINT_LABEL)


def save(fig, stem):
    fig.savefig(FIG / f'{stem}.pdf', bbox_inches='tight', pad_inches=0.02)
    fig.savefig(FIG / f'{stem}.png', dpi=350, bbox_inches='tight', pad_inches=0.02)
    plt.close(fig)


def rounded_box(ax, xy, wh, title, lines, fc, ec='#666666'):
    x,y=xy; w,h=wh
    box=FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.012,rounding_size=0.015',
                       linewidth=1.1,edgecolor=ec,facecolor=fc)
    ax.add_patch(box)
    ax.text(x+0.025*w,y+h-0.13*h,title,weight='bold',fontsize=11,va='center')
    ax.text(x+0.025*w,y+h-0.27*h,'\n'.join(lines),fontsize=8.7,va='top',linespacing=1.35)


def figure1():
    fig, ax = plt.subplots(figsize=(10.8, 3.55))
    ax.set_xlim(0,1); ax.set_ylim(0,1); ax.axis('off')
    rounded_box(ax,(0.015,0.10),(0.29,0.80),'Virtual stress test',[
        '138 adults; 747 gait cycles',
        'Thigh-shank and shank-foot pairs',
        '7 training caps x 7 exact test angles',
        '50 rotation-axis realizations',
        '5 participant-disjoint outer folds'
    ],'#eef5fb')
    rounded_box(ax,(0.355,0.10),(0.29,0.80),'Robustness decision',[
        'Weight-normalized augmentation',
        'Ridge penalty reselected by inner CV',
        'Two label-free frame normalizers',
        'Gate: upper 95% CI <= 4.5 deg',
        'for every endpoint at every test angle',
        'through the claimed field bound'
    ],'#f7f3e8')
    rounded_box(ax,(0.695,0.10),(0.29,0.80),'Independent physical-IMU stress test',[
        'Camargo open biomechanics dataset',
        '19 adults; 248 straight-walking windows',
        'Measured thigh, shank and foot IMUs',
        'Measured knee/ankle kinematics + GRF',
        'No external hyperparameter retuning',
        'Virtual-selected penalties transferred'
    ],'#eef8f1')
    for a,b in [(0.305,0.355),(0.645,0.695)]:
        ax.add_patch(FancyArrowPatch((a,0.50),(b,0.50),arrowstyle='-|>',mutation_scale=16,
                                     linewidth=1.3,color='#555555'))
    ax.text(0.5,0.985,'Orientation-robust sparse inertial gait estimation',ha='center',va='top',
            fontsize=12.5,weight='bold')
    save(fig,'fig1_study_design')


def figure2():
    w=pd.read_csv(SD/'virtual_worst_endpoint_matrix.csv')
    dep=pd.read_csv(SD/'virtual_deployment_rule.csv')
    caps=sorted(w.train_augmentation_max_deg.unique())
    tests=sorted(w.test_rotation_deg.unique())
    mat=w.pivot(index='train_augmentation_max_deg',columns='test_rotation_deg',values='worst_ci_high_deg').loc[caps,tests].to_numpy()
    fig,axs=plt.subplots(1,2,figsize=(10.8,4.15),gridspec_kw={'width_ratios':[1.55,0.9]})
    ax=axs[0]
    x_edges=np.arange(len(tests)+1)-0.5
    y_edges=np.arange(len(caps)+1)-0.5
    im=ax.pcolormesh(x_edges,y_edges,mat,cmap='RdYlBu_r',vmin=1.5,vmax=8.5,
                     shading='flat',edgecolors='none',rasterized=False)
    ax.set_xlim(-0.5,len(tests)-0.5); ax.set_ylim(-0.5,len(caps)-0.5)
    for i,c in enumerate(caps):
        for j,t in enumerate(tests):
            v=mat[i,j]
            ax.text(j,i,f'{v:.2f}',ha='center',va='center',fontsize=7.5,
                    color='white' if v>6.3 else 'black')
    selected={float(r.field_q95_rotation_deg):float(r.selected_train_augmentation_max_deg)
              for _,r in dep[dep.decision=='DEPLOY'].iterrows()}
    for t,c in selected.items():
        i=caps.index(c);j=tests.index(t)
        ax.add_patch(Rectangle((j-.48,i-.48),.96,.96,fill=False,linewidth=2.1,edgecolor='black'))
    ax.set_xticks(range(len(tests)),[f'{x:g}' for x in tests])
    ax.set_yticks(range(len(caps)),[f'{x:g}' for x in caps])
    ax.set_xlabel('Exact test rotation per sensor (deg)')
    ax.set_ylabel('Training augmentation cap (deg)')
    ax.set_title('A. Worst endpoint: upper 95% CI of RMSE')
    ax.contour(np.arange(len(tests)),np.arange(len(caps)),mat,levels=[4.5],colors='black',linewidths=1.2,linestyles='--')

    ax=axs[1]
    xx=dep.field_q95_rotation_deg.to_numpy(float)
    ok=dep.decision.eq('DEPLOY').to_numpy()
    yy=dep.selected_train_augmentation_max_deg.to_numpy(float)
    ax.plot(xx[ok],yy[ok],'-o',color='#0072B2',lw=2,ms=6,label='Selected cap')
    ax.scatter(xx[~ok],np.full((~ok).sum(),14.3),marker='x',s=65,color='#D55E00',lw=2,label='No tested cap passed')
    for x,y in zip(xx[ok],yy[ok]):
        ax.annotate(f'{y:g} deg',(x,y),xytext=(0,8),textcoords='offset points',ha='center',fontsize=8)
    ax.set_xlim(-.6,15.6);ax.set_ylim(-.6,15.8)
    ax.set_xticks(tests);ax.set_yticks(tests)
    ax.set_xlabel('Field rotation bound (deg)')
    ax.set_ylabel('Selected training cap (deg)')
    ax.set_title('B. Fixed operating rule')
    ax.grid(axis='both',alpha=.2)
    ax.legend(loc='upper left',frameon=False)
    fig.suptitle('Full virtual rotation matrix defines an operating envelope',fontsize=12,weight='bold',y=1.01)
    fig.tight_layout()
    save(fig,'fig2_virtual_rotation_matrix')


def virtual_method_frame(angle):
    m=pd.read_csv(SD/'virtual_rotation_matrix_summary.csv')
    b=pd.read_csv(SD/'virtual_orientation_baselines.csv')
    rows=[]
    for label,cap in [('Unaugmented',0.0),('10-deg augmentation',10.0)]:
        q=m[(m.train_augmentation_max_deg==cap)&(m.test_rotation_deg==angle)].copy()
        q['method']=label
        q=q.rename(columns={'ci_low':'lo','ci_high':'hi'})
        rows.append(q[['method','target','mean_participant_rmse_deg','lo','hi']])
    for method,label in [('template_registration','Template registration'),('gait_functional_calibration','Gait-functional frame')]:
        q=b[(b.method==method)&(b.test_rotation_deg==angle)].copy();q['method']=label
        q=q.rename(columns={'ci95_low_deg':'lo','ci95_high_deg':'hi'})
        rows.append(q[['method','target','mean_participant_rmse_deg','lo','hi']])
    return pd.concat(rows,ignore_index=True)


def figure3():
    fig,axs=plt.subplots(1,2,figsize=(10.8,4.0),sharey=True)
    methods=['Unaugmented','10-deg augmentation','Template registration','Gait-functional frame']
    offsets=np.array([-.27,-.09,.09,.27])
    for ax,angle in zip(axs,[0.0,10.0]):
        d=virtual_method_frame(angle)
        for k,method in enumerate(methods):
            q=d[d.method==method].set_index('target').loc[ENDPOINT_ORDER]
            y=q.mean_participant_rmse_deg.to_numpy();lo=q.lo.to_numpy();hi=q.hi.to_numpy()
            x=np.arange(4)+offsets[k]
            ax.errorbar(x,y,yerr=np.vstack([y-lo,hi-y]),fmt=METHOD_MARKERS[method],capsize=2.3,lw=1.2,ms=5.2,
                        label=method,color=COLORS[method],markeredgecolor='black',markeredgewidth=0.35)
        ax.axhline(4.5,color='#555555',ls='--',lw=1)
        ax.set_xticks(range(4),[ENDPOINT_LABEL[e] for e in ENDPOINT_ORDER])
        ax.set_title(f"{'A' if angle==0 else 'B'}. Test rotation = {angle:g} deg")
        ax.grid(axis='y',alpha=.2)
        ax.set_ylim(0,9.0)
    axs[0].set_ylabel('Mean participant RMSE (deg)')
    axs[1].legend(frameon=False,loc='upper right')
    fig.suptitle('Augmentation and explicit frame normalization occupy different accuracy-robustness points',fontsize=11.8,weight='bold',y=1.02)
    fig.tight_layout()
    save(fig,'fig3_virtual_orientation_methods')


def external_method_frame(angle=10.0):
    m=pd.read_csv(SD/'external_rotation_matrix_summary.csv')
    b=pd.read_csv(SD/'external_orientation_baselines.csv')
    rows=[]
    for label,cap in [('Unaugmented',0.0),('10-deg augmentation',10.0)]:
        q=m[(m.train_augmentation_max_deg==cap)&(m.test_rotation_deg==angle)].copy();q['method']=label
        q=q.rename(columns={'ci95_low_deg':'lo','ci95_high_deg':'hi'})
        rows.append(q[['method','target','mean_participant_rmse_deg','lo','hi']])
    for method,label in [('template_registration','Template registration'),('gait_functional_calibration','Gait-functional frame')]:
        q=b[(b.method==method)&(b.test_rotation_deg==angle)].copy();q['method']=label
        q=q.rename(columns={'ci95_low_deg':'lo','ci95_high_deg':'hi'})
        rows.append(q[['method','target','mean_participant_rmse_deg','lo','hi']])
    return pd.concat(rows,ignore_index=True)


def figure4():
    fig,axs=plt.subplots(1,2,figsize=(10.8,4.0),gridspec_kw={'width_ratios':[1,1.12]})
    d=pd.read_csv(SD/'external_unaugmented_degradation.csv')
    ax=axs[0]
    for endpoint in ENDPOINT_ORDER:
        q=d[d.target==endpoint].sort_values('test_rotation_deg')
        x=q.test_rotation_deg.to_numpy();y=q.mean_paired_degradation_deg.to_numpy();lo=q.ci95_low_deg.to_numpy();hi=q.ci95_high_deg.to_numpy()
        ax.errorbar(x,y,yerr=np.vstack([y-lo,hi-y]),marker=ENDPOINT_MARKERS[endpoint],
                    linestyle=ENDPOINT_LINESTYLES[endpoint],ms=4.5,lw=1.45,capsize=2,
                    label=ENDPOINT_LABEL[endpoint].replace('\n',' '),color=ENDPOINT_COLORS[endpoint],
                    markeredgecolor='black',markeredgewidth=0.3)
    ax.axhline(0,color='#555555',lw=.9)
    ax.set_xlabel('Exact test rotation per sensor (deg)')
    ax.set_ylabel('Paired RMSE increase vs 0 deg (deg)')
    ax.set_title('A. Physical-IMU frame-rotation sensitivity')
    ax.grid(axis='y',alpha=.2);ax.legend(frameon=False,fontsize=7.8)

    ax=axs[1];dd=external_method_frame(10.0);methods=['Unaugmented','10-deg augmentation','Template registration','Gait-functional frame'];offs=np.array([-.27,-.09,.09,.27])
    for k,method in enumerate(methods):
        q=dd[dd.method==method].set_index('target').loc[ENDPOINT_ORDER]
        y=q.mean_participant_rmse_deg.to_numpy();lo=q.lo.to_numpy();hi=q.hi.to_numpy();x=np.arange(4)+offs[k]
        ax.errorbar(x,y,yerr=np.vstack([y-lo,hi-y]),fmt=METHOD_MARKERS[method],capsize=2.3,lw=1.2,ms=5.2,
                    label=method,color=COLORS[method],markeredgecolor='black',markeredgewidth=0.35)
    ax.set_xticks(range(4),[ENDPOINT_LABEL[e] for e in ENDPOINT_ORDER])
    ax.set_ylabel('Mean participant RMSE (deg)')
    ax.set_title('B. External physical-IMU test at 10 deg')
    ax.grid(axis='y',alpha=.2);ax.set_ylim(2.2,8.0)
    ax.legend(frameon=False,fontsize=7.8,loc='upper right')
    fig.suptitle('Physical-IMU frame-rotation sensitivity and mitigation',fontsize=11.8,weight='bold',y=1.02)
    fig.tight_layout()
    save(fig,'fig4_external_replication')


if __name__ == '__main__':
    figure1(); figure2(); figure3(); figure4()
    print('Wrote figures 1-4 to', FIG)
