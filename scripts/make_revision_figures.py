from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
SD = ROOT / 'source_data'
FIG = ROOT / 'figures'
FIG.mkdir(exist_ok=True)

mpl.rcParams.update({
    'font.family': 'DejaVu Sans', 'font.size': 9.5, 'axes.titlesize': 10.5,
    'axes.labelsize': 9.5, 'legend.fontsize': 8.2, 'xtick.labelsize': 8.5,
    'ytick.labelsize': 8.5, 'pdf.fonttype': 42, 'ps.fonttype': 42,
    'axes.spines.top': False, 'axes.spines.right': False,
})

COLORS = {'Unaugmented':'#4d4d4d','10-deg augmentation':'#0072B2','Template registration':'#D55E00','Gait-functional frame':'#009E73'}
METHOD_MARKERS={'Unaugmented':'o','10-deg augmentation':'s','Template registration':'D','Gait-functional frame':'^'}
ENDPOINT_COLORS={'ankle_dorsiflex_deg':'#0072B2','ankle_plantarflex_deg':'#56B4E9','knee_stance_flex_deg':'#D55E00','knee_swing_flex_deg':'#CC79A7'}
ENDPOINT_MARKERS={'ankle_dorsiflex_deg':'o','ankle_plantarflex_deg':'s','knee_stance_flex_deg':'D','knee_swing_flex_deg':'^'}
ENDPOINT_LINESTYLES={'ankle_dorsiflex_deg':'-','ankle_plantarflex_deg':'--','knee_stance_flex_deg':'-.','knee_swing_flex_deg':':'}
ENDPOINT_LABEL={'ankle_dorsiflex_deg':'Ankle\ndorsiflexion','ankle_plantarflex_deg':'Ankle\nplantarflexion','knee_stance_flex_deg':'Knee stance\nflexion','knee_swing_flex_deg':'Knee swing\nflexion'}
ENDPOINT_ORDER=list(ENDPOINT_LABEL)


def save(fig, stem):
    fig.savefig(FIG/f'{stem}.pdf', bbox_inches='tight', pad_inches=0.03)
    fig.savefig(FIG/f'{stem}.png', dpi=350, bbox_inches='tight', pad_inches=0.03)
    plt.close(fig)


def virtual_method_frame(angle):
    m=pd.read_csv(SD/'virtual_rotation_matrix_summary.csv')
    b=pd.read_csv(SD/'virtual_orientation_baselines.csv')
    rows=[]
    for label,cap in [('Unaugmented',0.0),('10-deg augmentation',10.0)]:
        q=m[(m.train_augmentation_max_deg==cap)&(m.test_rotation_deg==angle)].copy(); q['method']=label
        q=q.rename(columns={'ci_low':'lo','ci_high':'hi'})
        rows.append(q[['method','target','mean_participant_rmse_deg','lo','hi']])
    for method,label in [('template_registration','Template registration'),('gait_functional_calibration','Gait-functional frame')]:
        q=b[(b.method==method)&(b.test_rotation_deg==angle)].copy(); q['method']=label
        q=q.rename(columns={'ci95_low_deg':'lo','ci95_high_deg':'hi'})
        rows.append(q[['method','target','mean_participant_rmse_deg','lo','hi']])
    return pd.concat(rows,ignore_index=True)


def fig2_virtual_methods():
    methods=['Unaugmented','10-deg augmentation','Template registration','Gait-functional frame']
    offs=np.array([-.27,-.09,.09,.27])
    fig,axs=plt.subplots(1,2,figsize=(10.9,4.35))
    for ax,angle,panel in zip(axs,[0.0,10.0],['A','B']):
        dd=virtual_method_frame(angle)
        for k,method in enumerate(methods):
            q=dd[dd.method==method].set_index('target').loc[ENDPOINT_ORDER]
            y=q.mean_participant_rmse_deg.to_numpy(); lo=q.lo.to_numpy(); hi=q.hi.to_numpy(); x=np.arange(4)+offs[k]
            ax.errorbar(x,y,yerr=np.vstack([y-lo,hi-y]),fmt=METHOD_MARKERS[method],capsize=2.3,lw=1.2,ms=5.2,
                        label=method,color=COLORS[method],markeredgecolor='black',markeredgewidth=0.35)
        ax.axhline(4.5,color='#555555',ls='--',lw=1)
        ax.set_xticks(range(4),[ENDPOINT_LABEL[e] for e in ENDPOINT_ORDER])
        ax.set_title(f'{panel}. Absolute RMSE at {angle:g}-deg test rotation')
        ax.grid(axis='y',alpha=.2); ax.set_ylim(0,9.0)
    axs[0].set_ylabel('Mean participant RMSE (deg)')
    axs[1].legend(frameon=False,loc='upper center',bbox_to_anchor=(0.5,-0.23),ncol=2,columnspacing=1.0,handlelength=1.3)
    fig.suptitle('Virtual accuracy and invariance under sensor-frame rotation',fontsize=11.8,weight='bold',y=0.98)
    fig.subplots_adjust(left=0.075,right=0.985,bottom=0.25,top=0.82,wspace=0.22)
    save(fig,'fig3_virtual_orientation_methods')


def external_method_frame(angle=10.0):
    m=pd.read_csv(SD/'external_rotation_matrix_summary.csv')
    b=pd.read_csv(SD/'external_orientation_baselines.csv')
    rows=[]
    for label,cap in [('Unaugmented',0.0),('10-deg augmentation',10.0)]:
        q=m[(m.train_augmentation_max_deg==cap)&(m.test_rotation_deg==angle)].copy(); q['method']=label
        q=q.rename(columns={'ci95_low_deg':'lo','ci95_high_deg':'hi'})
        rows.append(q[['method','target','mean_participant_rmse_deg','lo','hi']])
    for method,label in [('template_registration','Template registration'),('gait_functional_calibration','Gait-functional frame')]:
        q=b[(b.method==method)&(b.test_rotation_deg==angle)].copy(); q['method']=label
        q=q.rename(columns={'ci95_low_deg':'lo','ci95_high_deg':'hi'})
        rows.append(q[['method','target','mean_participant_rmse_deg','lo','hi']])
    return pd.concat(rows,ignore_index=True)


def fig3_external():
    fig,axs=plt.subplots(1,2,figsize=(10.9,4.45),gridspec_kw={'width_ratios':[1,1.12]})
    d=pd.read_csv(SD/'external_unaugmented_degradation.csv')
    ax=axs[0]
    for endpoint in ENDPOINT_ORDER:
        q=d[d.target==endpoint].sort_values('test_rotation_deg')
        x=q.test_rotation_deg.to_numpy(); y=q.mean_paired_degradation_deg.to_numpy(); lo=q.ci95_low_deg.to_numpy(); hi=q.ci95_high_deg.to_numpy()
        ax.errorbar(x,y,yerr=np.vstack([y-lo,hi-y]),marker=ENDPOINT_MARKERS[endpoint],linestyle=ENDPOINT_LINESTYLES[endpoint],
                    ms=4.5,lw=1.45,capsize=2,label=ENDPOINT_LABEL[endpoint].replace('\n',' '),color=ENDPOINT_COLORS[endpoint],
                    markeredgecolor='black',markeredgewidth=0.3)
    ax.axhline(0,color='#555555',lw=.9)
    ax.set_xlabel('Exact test rotation per sensor (deg)')
    ax.set_ylabel('Paired RMSE increase from 0 deg (deg)')
    ax.set_title('A. Within-participant increase from baseline')
    ax.grid(axis='y',alpha=.2)
    ax.legend(frameon=False,fontsize=7.5,loc='upper center',bbox_to_anchor=(0.5,-0.23),ncol=2,columnspacing=.8,handlelength=1.5)

    ax=axs[1]; dd=external_method_frame(10.0); methods=['Unaugmented','10-deg augmentation','Template registration','Gait-functional frame']; offs=np.array([-.27,-.09,.09,.27])
    for k,method in enumerate(methods):
        q=dd[dd.method==method].set_index('target').loc[ENDPOINT_ORDER]
        y=q.mean_participant_rmse_deg.to_numpy(); lo=q.lo.to_numpy(); hi=q.hi.to_numpy(); x=np.arange(4)+offs[k]
        ax.errorbar(x,y,yerr=np.vstack([y-lo,hi-y]),fmt=METHOD_MARKERS[method],capsize=2.3,lw=1.2,ms=5.2,
                    label=method,color=COLORS[method],markeredgecolor='black',markeredgewidth=0.35)
    ax.set_xticks(range(4),[ENDPOINT_LABEL[e] for e in ENDPOINT_ORDER])
    ax.set_ylabel('Absolute mean participant RMSE (deg)')
    ax.set_title('B. Absolute RMSE at 10-deg test rotation')
    ax.grid(axis='y',alpha=.2); ax.set_ylim(2.2,8.1)
    ax.legend(frameon=False,fontsize=7.5,loc='upper center',bbox_to_anchor=(0.5,-0.23),ncol=2,columnspacing=.8,handlelength=1.3)
    fig.suptitle('Physical-IMU sensor-frame rotation sensitivity and mitigation',fontsize=11.8,weight='bold',y=0.98)
    fig.subplots_adjust(left=.075,right=.985,bottom=.26,top=.82,wspace=.25)
    save(fig,'fig4_external_replication')

if __name__=='__main__':
    fig2_virtual_methods(); fig3_external(); print('revised figures written')
