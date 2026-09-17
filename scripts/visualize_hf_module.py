"""Render code-matched 3-D high-pass MRI examples and a detailed LHFC diagram.

Dependencies: numpy, nibabel, matplotlib. No trained checkpoint is required for
the fixed filter. Learned pyramid tensors are intentionally drawn as schematics.
The filter matches LaplacianEdge3d: a 3x3x3 uniform kernel, zero padding, I-blur(I).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import nibabel as nib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Polygon, FancyBboxPatch, Circle

plt.rcParams.update({'font.family':'sans-serif', 'font.sans-serif':['DejaVu Sans','Arial'], 'font.size':10,
                     'mathtext.fontset':'stix', 'pdf.fonttype':42,
                     'ps.fonttype':42, 'svg.fonttype':'none', 'savefig.dpi':300})
INK='#243442'; ORANGE='#BD7139'; BLUE='#507F9E'; TEAL='#458579'


def box_blur3d(x):
    """Same mathematical operator and zero padding as the model, in float32."""
    x=np.asarray(x,dtype=np.float32)
    padded=np.pad(x,1,mode='constant',constant_values=0)
    out=np.zeros_like(x)
    for i in range(3):
        for j in range(3):
            for k in range(3):
                out += padded[i:i+x.shape[0],j:j+x.shape[1],k:k+x.shape[2]] * np.float32(1/27)
    return out


def tile(ax,arr,cmap='gray',vmin=0,vmax=1,resolution=128,extent=None):
    """Editable vector cells for PDFs/SVGs; image data is saved separately intact."""
    rows=np.linspace(0,arr.shape[0]-1,min(resolution,arr.shape[0])).round().astype(int)
    cols=np.linspace(0,arr.shape[1]-1,min(resolution,arr.shape[1])).round().astype(int)
    shown=arr[np.ix_(rows,cols)]
    if extent is None:
        extent=(0,arr.shape[1],0,arr.shape[0])
    x0,x1,y0,y1=extent
    # Flip the samples to reproduce imshow(origin='upper') without inverting a
    # shared diagram axis. Face-colored overlapping strokes prevent white
    # hairline seams in PDF viewers without embedding a raster image.
    artist=ax.pcolormesh(np.linspace(x0,x1,shown.shape[1]+1),
                        np.linspace(y0,y1,shown.shape[0]+1),shown[::-1],
                        cmap=cmap,vmin=vmin,vmax=vmax,shading='flat',
                        edgecolors='face',linewidth=.5,antialiased=False,rasterized=False)
    return artist


def export(fig,out,name):
    fig.canvas.draw()
    # Preserve physical panel geometry without adding dependencies on the server.
    rects=[(a.get_position().bounds*np.array([*fig.get_size_inches(),*fig.get_size_inches()])*72).tolist()
           for a in fig.axes]
    (out/f'{name}.panel_rectangles.json').write_text(json.dumps({'units':'pt','rectangles_xywh':rects},indent=2))
    try:
        from audit_panel_alignment import require_matplotlib_panel_alignment
    except ImportError:
        pass
    else:
        require_matplotlib_panel_alignment(fig,json_out=out/f'{name}.alignment.json')
    fig.savefig(out/f'{name}.pdf',facecolor='white')
    fig.savefig(out/f'{name}.svg',facecolor='white')
    fig.savefig(out/f'{name}.png',facecolor='white',dpi=300)
    plt.close(fig)


def comparison(out,img,low,high,mask,roi,args,limit):
    fig,axes=plt.subplots(2,4,figsize=(12,6.8))
    fig.subplots_adjust(left=.035,right=.975,top=.84,bottom=.13,wspace=.12,hspace=.20)
    arrays=(img,low,high,np.abs(high))
    titles=('Input MRI', 'Low-pass image', 'Signed high-frequency residual', 'Residual magnitude')
    equations=(r'$I$',r'$I_{\ell}=K*I$',r'$I_h=I-I_{\ell}$',r'$|I_h|$ (display only)')
    r0,r1,c0,c1=roi
    for j,(arr,title,eq) in enumerate(zip(arrays,titles,equations)):
        lo,hi=(-limit,limit) if j==2 else (0,limit) if j==3 else (0,1)
        for row in (0,1):
            ax=axes[row,j]
            patch=arr if row==0 else arr[r0:r1,c0:c1]
            tile(ax,patch,vmin=lo,vmax=hi,resolution=args.vector_resolution)
            ax.set_aspect('equal'); ax.set_axis_off()
            if row==0:
                # Rectangle coordinates correspond to the flipped image display.
                ax.add_patch(Rectangle((c0,arr.shape[0]-r1),c1-c0,r1-r0,
                                       fill=False,edgecolor='#EEA64B',lw=1.0))
                ax.set_title(title+'\n'+eq,fontsize=10,pad=8)
            elif mask is not None:
                mm=mask[r0:r1,c0:c1]
                if mm.any() and not mm.all():
                    ax.contour(np.arange(mm.shape[1])+.5,np.arange(mm.shape[0])+.5,
                               mm[::-1],levels=[.5],colors=['#E69F00'],linewidths=.65)
    axes[1,0].text(0,-.10,'Same ROI in all four columns',transform=axes[1,0].transAxes,fontsize=9)
    fig.suptitle('High-frequency decomposition of a real 3D MRI volume',fontsize=15,y=.97)
    fig.text(.5,.91,f'{args.modality}  |  volume axis {args.axis}, slice {args.selected_slice}  |  3 x 3 x 3 uniform filter; zero padding',ha='center',fontsize=10)
    fig.text(.5,.06,f'Input and low-pass: [0, 1]. Signed residual: [{-limit:.3g}, {limit:.3g}]. Magnitude: [0, {limit:.3g}].',ha='center',fontsize=9)
    fig.text(.5,.025,'Orange contour: provided GT label; ROI and display windows are identical between full image and zoom. No prediction is shown.',ha='center',fontsize=8)
    export(fig,out,'hf_before_after')


def detailed_module(out,img,low,high,args,limit):
    fig=plt.figure(figsize=(12,8.8))
    ax=fig.add_axes([.02,.035,.96,.92]); ax.set_xlim(0,120); ax.set_ylim(0,92); ax.axis('off')
    def text(x,y,s,size=10,color=INK,weight='normal',ha='center'):
        ax.text(x,y,s,ha=ha,va='center',fontsize=size,color=color,fontweight=weight)
    def arrow(points,color=INK,ls='-'):
        for p,q in zip(points[:-2],points[1:-1]):
            ax.plot([p[0],q[0]],[p[1],q[1]],color=color,lw=1.15,ls=ls)
        ax.annotate('',xy=points[-1],xytext=points[-2],arrowprops={'arrowstyle':'->','lw':1.15,'color':color,'linestyle':ls})
    def box(x,y,w,h,label,fc='#FAEADD',ec=ORANGE,size=9):
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.25,rounding_size=.65',facecolor=fc,edgecolor=ec,lw=.9))
        text(x+w/2,y+h/2,label,size)
    def feature(x,y,w=3.4,h=6,color='#E5B788'):
        ax.add_patch(Rectangle((x,y),w,h,fc=color,ec='#6F8491',lw=.7))
        ax.add_patch(Polygon([(x,y+h),(x+1.4,y+h+1.5),(x+w+1.4,y+h+1.5),(x+w,y+h)],fc='#F7F7F5',ec='#6F8491',lw=.7))
        ax.add_patch(Polygon([(x+w,y),(x+w+1.4,y+1.5),(x+w+1.4,y+h+1.5),(x+w,y+h)],fc='#B88860' if color=='#E5B788' else '#759EB5',ec='#6F8491',lw=.7))
    for y,h in [(61,29),(30,28),(1,26)]:
        ax.add_patch(FancyBboxPatch((1,y),118,h,boxstyle='round,pad=0.0,rounding_size=1.5',fc='#FCFCFC',ec='#B4BEC5',lw=.9,linestyle=(0,(4,3))))
    text(4,87,'(a) Fixed 3D frequency decomposition (applied per modality)',11,weight='bold',ha='left')
    for cx,arr,title in [(12,img,r'Input $I$'),(52,low,r'Low-frequency $I_{\ell}$'),(100,high,r'High-frequency $I_h$')]:
        tile(ax,arr,vmin=-limit if cx==100 else 0,vmax=limit if cx==100 else 1,
             resolution=args.vector_resolution,extent=(cx-7,cx+7,66,81))
        text(cx,63.5,title,10)
    arrow([(19,73.5),(25,73.5)])
    box(25,70,14,7,'3D box filter\n3 × 3 × 3',size=9)
    arrow([(39,73.5),(45,73.5)])
    arrow([(59,73.5),(74.5,73.5)])
    ax.add_patch(Circle((77,73.5),2.1,fc='white',ec=INK,lw=1))
    text(77,73.5,'+',18)
    text(72,76,'−',11)
    arrow([(21,73.5),(21,83),(77,83),(77,75.7)])
    text(79,79,'+',11)
    arrow([(79.2,73.5),(93,73.5)],ORANGE)
    text(82,65,r'$I_h = I-K*I$',12)
    text(4,55,'(b) Learned multiscale feature pyramid',11,weight='bold',ha='left')
    text(6,43,r'$I_h$',14)
    arrow([(9,43),(12,43)],ORANGE)
    box(12,38,14,10,'Conv 3³\nGN · ReLU',size=9)
    arrow([(26.3,43),(30,43)],ORANGE)
    for j,(x,ch,scale) in enumerate([(30,24,1),(55,24,2),(80,48,4),(105,96,8)]):
        feature(x,40,3.8,6)
        text(x+2,36.5,rf'$e_{j}$',13)
        dims=rf'{ch}\times D\times H\times W' if scale==1 else rf'{ch}\times\frac{{D}}{{{scale}}}\times\frac{{H}}{{{scale}}}\times\frac{{W}}{{{scale}}}'
        text(x+2,32.8,'$'+dims+'$',9)
        if j<3:
            arrow([(x+5.4,43),(x+8.3,43)],ORANGE)
            box(x+8.5,38,12.8,10,'AvgPool / 2\nConv 3³\nGN · ReLU',size=8)
            arrow([(x+21.5,43),(x+24.8,43)],ORANGE)
    text(4,24,'(c) Scale-matched concatenation in each decoder block',11,weight='bold',ha='left')
    feature(9,13,3,4,'#B6D3E5'); text(6,15,r'$S_s$',12)
    feature(9,5,3,4,'#B6D3E5'); text(6,7,r'$U_s$',12)
    arrow([(13.5,15),(32,15),(32,12),(51,12)])
    arrow([(13.5,7),(18,7)])
    box(18,4.5,15,5,'Upsample × 2',fc='#E4EFF5',ec=BLUE,size=8)
    arrow([(33.5,7),(42,7),(42,10),(51,10)])
    feature(43,16,2.3,3.5); text(39,18,r'$e_s$',12)
    arrow([(46.7,18),(54,18),(54,14.5)],ORANGE)
    ax.add_patch(Circle((54,11.3),2.8,fc='white',ec=INK,lw=1)); text(54,11.3,'C',12)
    arrow([(57,11.3),(63,11.3)])
    box(63,7.3,19,8,'Residual block',fc='#E5F1EE',ec=TEAL)
    arrow([(82.5,11.3),(90,11.3)])
    feature(91,8,4,6,'#B6D3E5'); text(103,11.3,r'$D_s$',14)
    text(73,3.5,r'$D_s=\mathrm{ResBlock}([S_s,\mathrm{Up}(U_s),e_s])$',11)
    fig.text(.5,.982,'LHFC: high-frequency feature extraction and multiscale concatenation',ha='center',fontsize=14,fontweight='bold')
    fig.text(.5,.959,f'Illustrative modality: {args.modality} | base channels = 24',ha='center',fontsize=8)
    fig.text(.5,.01,'Real MRI and filter outputs in (a); learned tensors in (b,c) are structural illustrations. C: channel concatenation; GN: GroupNorm.',ha='center',fontsize=8)
    export(fig,out,'lhfc_module_detail')


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--image',type=Path,required=True,help='One real 3D modality NIfTI (.nii or .nii.gz)')
    ap.add_argument('--mask',type=Path,help='Aligned optional GT NIfTI; used only for slice/ROI/contour')
    ap.add_argument('--mask-label',type=int,help='Specific GT label (e.g. BraTS ET=4); otherwise all nonzero labels')
    ap.add_argument('--modality',default='T1ce')
    ap.add_argument('--axis',type=int,choices=[0,1,2],default=2,help='Native NIfTI voxel axis; no reorientation')
    ap.add_argument('--slice',type=int,default=None,help='Zero-based slice after optional crop; default maximum GT area or middle')
    ap.add_argument('--crop',choices=['none','brats-legacy'],default='none',help='brats-legacy matches data/dataset.py crop [40:210,40:210,20:120] before min-max')
    ap.add_argument('--normalization',choices=['minmax','none'],default='minmax')
    ap.add_argument('--hf-limit',type=float,help='Symmetric display limit; default 99th percentile |Ih| in displayed nonzero input')
    ap.add_argument('--roi-size',type=int,default=48)
    ap.add_argument('--roi-center',nargs=2,type=int,metavar=('ROW','COL'),help='Center in rotated displayed full slice; otherwise GT centroid or image center')
    ap.add_argument('--vector-resolution',type=int,default=128,help='Maximum cells per image axis for editable preview; underlying numerical slice remains intact')
    ap.add_argument('--verify-torch',action='store_true',help='Compare NumPy implementation with the actual LaplacianEdge3d class on CPU')
    ap.add_argument('--out',type=Path,default=Path('outputs/hf_module'))
    args=ap.parse_args()
    if args.vector_resolution<16 or args.roi_size<4:
        ap.error('vector-resolution >= 16 and roi-size >= 4 required')
    nii=nib.load(str(args.image),mmap=False); raw=nii.get_fdata(dtype=np.float32)
    if raw.ndim!=3 or not np.isfinite(raw).all():
        raise ValueError('Image must be finite and 3D')
    mask=None
    if args.mask:
        mn=nib.load(str(args.mask),mmap=False)
        if mn.shape!=nii.shape or not np.allclose(mn.affine,nii.affine,atol=1e-4):
            raise ValueError('GT must have the same shape and affine as the MRI; resample it explicitly first')
        labels=mn.get_fdata(dtype=np.float32)
        mask=labels==args.mask_label if args.mask_label is not None else labels!=0
        if not mask.any():
            raise ValueError('Selected GT label is absent')
    if args.crop=='brats-legacy':
        if any(n<end for n,end in zip(raw.shape,(210,210,120))):
            raise ValueError('Volume too small for brats-legacy crop')
        raw=raw[40:210,40:210,20:120]
        mask=mask[40:210,40:210,20:120] if mask is not None else None
    vmin,vmax=float(raw.min()),float(raw.max())
    if vmax<=vmin:
        raise ValueError('Constant MRI volume')
    image=(raw-vmin)/(vmax-vmin) if args.normalization=='minmax' else raw.copy()
    if args.normalization=='none' and (image.min()<0 or image.max()>1):
        raise ValueError('--normalization none expects preprocessed [0,1] input for shared display windows')
    low=box_blur3d(image); high=image-low
    verification=None
    if args.verify_torch:
        sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
        import torch
        from models.resunet_edge import LaplacianEdge3d
        with torch.no_grad():
            actual=LaplacianEdge3d()(torch.from_numpy(image[None,None])).numpy()[0,0]
        verification=float(np.max(np.abs(actual-high)))
        if not np.allclose(actual,high,rtol=1e-5,atol=2e-6):
            raise AssertionError(f'Model comparison failed: {verification}')
    index=args.slice
    selection='user-specified'
    if index is None:
        if mask is not None and mask.any():
            index=int(np.argmax(mask.sum(axis=tuple(i for i in range(3) if i!=args.axis))))
            selection='maximum selected GT cross-sectional area (illustration only)'
        else:
            index=image.shape[args.axis]//2; selection='middle slice'
    if not 0<=index<image.shape[args.axis]:
        raise ValueError('Slice outside volume')
    args.selected_slice=index
    cut=lambda x: np.rot90(np.take(x,index,axis=args.axis))
    im,lo,hi=cut(image),cut(low),cut(high)
    mm=cut(mask) if mask is not None else None
    if args.roi_center is not None:
        center=args.roi_center
    elif mm is not None and mm.any():
        center=np.argwhere(mm).mean(axis=0)
    else:
        center=np.array(im.shape)/2
    nr,nc=im.shape; rh=min(args.roi_size,nr); rw=min(args.roi_size,nc)
    r0=int(np.clip(round(center[0]-rh/2),0,nr-rh)); c0=int(np.clip(round(center[1]-rw/2),0,nc-rw))
    roi=(r0,r0+rh,c0,c0+rw)
    vals=np.abs(hi[im!=0])
    limit=args.hf_limit if args.hf_limit is not None else float(np.percentile(vals if vals.size else np.abs(hi),99))
    if not np.isfinite(limit) or limit<=0:
        raise ValueError('Residual window must be positive')
    out=args.out; out.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(out/'source_slice_values.npz',input=im,lowpass=lo,highpass=hi,
                        magnitude=np.abs(hi),gt=mm if mm is not None else np.zeros(im.shape,dtype=bool))
    meta={'image':str(args.image.resolve()),'mask':str(args.mask.resolve()) if args.mask else None,
          'modality':args.modality,'shape':list(image.shape),'axis':args.axis,'slice':index,
          'slice_selection':selection,'crop':args.crop,'normalization':args.normalization,
          'raw_min':vmin,'raw_max':vmax,'kernel':'ones(3,3,3)/27','padding':'zero, width 1',
          'operation':'3D convolution then subtraction, before 2D slice and ROI',
          'rotation':'np.rot90 once for display; no anatomical reorientation',
          'roi_rc':roi,'gt_label':args.mask_label,'input_lowpass_window':[0,1],
          'signed_hf_window':[-limit,limit],'magnitude_window':[0,limit],
          'hf_window_policy':'user-specified' if args.hf_limit else 'p99 absolute residual in nonzero displayed input',
          'vector_resolution':args.vector_resolution,'model_check_max_abs_error':verification,
          'numerical_reconstruction_max_abs_error':float(np.max(np.abs(image-low-high))),
          'learned_feature_maps':'not generated; boxes only; trained weights required',
          'interpretation':'Illustration of the fixed operator; not evidence of improved lesion detection'}
    (out/'visualization_metadata.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
    for name,arr in [('input',im),('lowpass',lo),('signed_highpass',hi),('highpass_magnitude',np.abs(hi))]:
        v0,v1=(-limit,limit) if name=='signed_highpass' else (0,limit) if name=='highpass_magnitude' else (0,1)
        plt.imsave(out/f'{name}.png',arr,cmap='gray',vmin=v0,vmax=v1)
    comparison(out,im,lo,hi,mm,roi,args,limit)
    detailed_module(out,im,lo,hi,args,limit)
    print(f'Saved to {out.resolve()} | slice={index} | signed HF limit={limit:.5g}')


if __name__=='__main__':
    main()
