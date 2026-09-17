"""Export real, editable 3-D auxiliary-boundary-supervision label examples.

Matches BCEDiceWithBoundaryLoss._extract_boundary_gt on binary targets.
No checkpoint or synthetic prediction is used. Dependencies: numpy, nibabel,
matplotlib, scipy. Run from the repository root; see docs/ABS_VISUALIZATION.md.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import nibabel as nib
from scipy import ndimage
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','DejaVu Sans'],
                     'font.size':10,'mathtext.fontset':'dejavusans','pdf.fonttype':42,
                     'svg.fonttype':'none','savefig.dpi':300})
GREEN='#32685E'; INK='#263B43'

def erode3d(target):
    """All 27 neighbours must be foreground; outside-volume values are zero.

    For binary inputs this equals avg_pool3d(k=3,s=1,p=1)>0.999, including
    its default count_include_pad=True. Apply before slicing, per region.
    """
    target=np.asarray(target,dtype=bool)
    padded=np.pad(target,1,constant_values=False)
    eroded=np.ones_like(target)
    for i in range(3):
        for j in range(3):
            for k in range(3):
                eroded &= padded[i:i+target.shape[0],j:j+target.shape[1],k:k+target.shape[2]]
    return eroded

def binary_target(labels,dataset,region,et_label):
    if dataset=='ucsf':
        if region!='lesion':
            raise ValueError('UCSF binary training uses --region lesion (all nonzero labels).')
        return labels>0
    if region=='WT': return np.isin(labels,[1,2,et_label])
    if region=='TC': return np.isin(labels,[1,et_label])
    if region=='ET': return labels==et_label
    raise ValueError('BraTS requires --region WT, TC or ET; these are overlapping binary channels.')

def vector_tile(ax,array,binary=False,resolution=96):
    """Editable cells, no raster image embedded in the PDF/SVG."""
    rows=np.linspace(0,array.shape[0]-1,min(resolution,array.shape[0])).round().astype(int)
    cols=np.linspace(0,array.shape[1]-1,min(resolution,array.shape[1])).round().astype(int)
    shown=array[np.ix_(rows,cols)]
    from matplotlib.colors import ListedColormap
    cmap=ListedColormap(['#152226','#BDA4E8']) if binary else 'gray'
    ax.pcolormesh(np.arange(shown.shape[1]+1),np.arange(shown.shape[0]+1),shown[::-1],
                  cmap=cmap,vmin=0,vmax=1,edgecolors='face',linewidth=.35,
                  antialiased=False,rasterized=False)
    ax.set_aspect('equal'); ax.axis('off')

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    source=ap.add_mutually_exclusive_group(required=True)
    source.add_argument('--image',type=Path,help='Native NIfTI illustration; requires --mask.')
    source.add_argument('--prepared',type=Path,help='UCSF cached training NPZ: use its exact image/mask arrays.')
    ap.add_argument('--mask',type=Path)
    ap.add_argument('--prepared-image-channel',type=int,default=0,choices=[0,1,2,3],help='UCSF: 0=T1post, 1=T1pre, 2=FLAIR, 3=subtraction.')
    ap.add_argument('--dataset',choices=['ucsf','brats'],required=True)
    ap.add_argument('--region',choices=['lesion','WT','TC','ET'])
    ap.add_argument('--et-label',type=int,default=4,help='BraTS2020 enhancing-tumour label; use 3 only for remapped data.')
    ap.add_argument('--axis',type=int,choices=[0,1,2],help='Default: 2 for native XYZ, 0 for cached DHW.')
    ap.add_argument('--slice',type=int,dest='slice_index',help='Zero-based slice index in the chosen input grid.')
    ap.add_argument('--margin',type=int,default=12)
    ap.add_argument('--vector-resolution',type=int,default=96)
    ap.add_argument('--verify-torch',action='store_true',help='Also compare with the actual loss implementation (requires torch).')
    ap.add_argument('--out',type=Path,default=Path('outputs/abs_detail'))
    args=ap.parse_args()
    args.region=args.region or ('lesion' if args.dataset=='ucsf' else 'ET')
    args.axis=args.axis if args.axis is not None else (0 if args.prepared else 2)
    if args.margin<0 or args.vector_resolution<8: ap.error('margin must be nonnegative and vector-resolution >= 8')
    if args.prepared:
        if args.dataset!='ucsf' or args.mask:ap.error('--prepared is for UCSF cached training tensors; do not also pass --mask.')
        with np.load(args.prepared,allow_pickle=False) as cached:
            inputs=np.asarray(cached['image'],dtype=np.float32)
            gt=np.asarray(cached['mask'])
            spacing=[float(x) for x in cached['spacing_dhw']]
        if inputs.ndim!=4 or inputs.shape[0]!=4 or gt.shape!=(1,*inputs.shape[1:]):
            raise ValueError('Expected UCSF cache image (4,D,H,W), mask (1,D,H,W).')
        image=inputs[args.prepared_image_channel];labels=gt[0]
        data_grid='prepared_training_tensor_DHW'
    else:
        if not args.mask:ap.error('--image requires --mask.')
        image_nii=nib.load(str(args.image)); mask_nii=nib.load(str(args.mask))
        image=image_nii.get_fdata(dtype=np.float32); labels=np.asarray(mask_nii.dataobj)
        if not np.allclose(image_nii.affine,mask_nii.affine,atol=1e-4): raise ValueError('Image and mask affines differ; do not silently overlay them.')
        spacing=[float(x) for x in image_nii.header.get_zooms()[:3]]
        data_grid='native_nifti_XYZ_illustration'
    if image.ndim!=3 or labels.shape!=image.shape: raise ValueError('Image and mask must be matching 3-D volumes.')
    if not np.isfinite(image).all(): raise ValueError('Image contains non-finite values.')
    if not np.equal(labels,np.round(labels)).all(): raise ValueError('Mask must contain discrete integer labels.')
    if args.dataset=='ucsf' and not set(np.unique(labels)).issubset({0,1}):
        raise ValueError('UCSF training requires the binary original _seg mask, not _BraTS-seg labels.')
    if args.prepared and (float(image.min())<0 or float(image.max())>1):
        raise ValueError('Expected normalized UCSF cached training image in [0,1].')
    if args.dataset=='brats' and not set(np.unique(labels)).issubset({0,1,2,args.et_label}):
        raise ValueError('Unexpected BraTS labels. Check --et-label and use the original segmentation, not WT/TC/ET logits.')
    target=binary_target(labels,args.dataset,args.region,args.et_label)
    if not target.any(): raise ValueError('Selected region has no foreground.')
    eroded=erode3d(target); boundary=target & ~eroded
    # Independent reference check, including foreground at image borders.
    reference=ndimage.binary_erosion(target,structure=np.ones((3,3,3),bool),border_value=0)
    if not np.array_equal(eroded,reference): raise AssertionError('3-D erosion reference mismatch')
    verification={'scipy_binary_erosion_exact_match':True,'torch_loss_exact_match':None}
    if args.verify_torch:
        import sys
        sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
        import torch
        from losses.enhanced import BCEDiceWithBoundaryLoss
        tensor=torch.from_numpy(target.astype(np.float32))[None,None]
        expected=BCEDiceWithBoundaryLoss(boundary_weight=.1)._extract_boundary_gt(tensor).numpy()[0,0]
        if not np.array_equal(expected,boundary): raise AssertionError('Actual torch loss mismatch')
        verification['torch_loss_exact_match']=True
    areas=target.sum(axis=tuple(i for i in range(3) if i!=args.axis))
    index=int(np.argmax(areas)) if args.slice_index is None else args.slice_index
    if not 0<=index<image.shape[args.axis]: raise ValueError('Slice index out of range.')
    take=lambda a: np.rot90(np.take(a,index,axis=args.axis))
    y=take(target); e=take(eroded); b=take(boundary); im=take(image)
    if not y.any(): raise ValueError('Selected slice has no target; choose a different slice.')
    components,n=ndimage.label(y,structure=np.ones((3,3),bool))
    sizes=np.bincount(components.ravel()); sizes[0]=0
    selected=components==int(np.argmax(sizes))
    rr,cc=np.where(selected)
    r0=max(0,int(rr.min())-args.margin); r1=min(y.shape[0],int(rr.max())+args.margin+1)
    c0=max(0,int(cc.min())-args.margin); c1=min(y.shape[1],int(cc.max())+args.margin+1)
    # Display normalization only; there is no model inference/preprocessing.
    lo,hi=float(image.min()),float(image.max())
    im=np.clip(im,0,1) if args.prepared else np.clip((im-lo)/max(hi-lo,1e-12),0,1)
    roi=(r0,r1,c0,c1); crop=lambda a:a[r0:r1,c0:c1]
    e2=ndimage.binary_erosion(y,structure=np.ones((3,3),bool),border_value=0)
    b2=y & ~e2
    args.out.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(args.out/'abs_arrays.npz',image_slice=im,target_slice=y,eroded_slice=e,
        boundary_slice=b,boundary_2d_slice=b2,roi=np.array(roi),image_roi=crop(im),
        target_roi=crop(y),eroded_roi=crop(e),boundary_roi=crop(b),boundary_2d_roi=crop(b2))
    arrays=[crop(im),crop(y),crop(e),crop(b)]
    names=['MRI ROI','GT region','3D-eroded GT','3D inner boundary']
    equations=[r'$I$',r'$Y$',r'$\mathrm{E}_{\mathrm{3D}}(Y)$',r'$Y_b=Y-\mathrm{E}_{\mathrm{3D}}(Y)$']
    fig,axes=plt.subplots(1,4,figsize=(10.8,3.8))
    fig.subplots_adjust(left=.035,right=.965,top=.76,bottom=.18,wspace=.16)
    for i,(ax,arr,name,eq) in enumerate(zip(axes,arrays,names,equations)):
        vector_tile(ax,arr,i>0,args.vector_resolution)
        ax.set_title(name,color=INK,pad=10)
        ax.text(.5,-.09,eq,transform=ax.transAxes,ha='center',va='top',fontsize=12)
    fig.suptitle(f'ABS target construction | {args.dataset.upper()} | {args.region} | slice {index}',fontsize=13,color=INK)
    fig.text(.5,.035,'Boundary labels are extracted from the 3D volume before slicing. All displayed maps are ground-truth-derived.',ha='center',fontsize=9,color=INK)
    fig.canvas.draw()
    panel_rects=[list(ax.get_position().bounds*np.array([*fig.get_size_inches(),*fig.get_size_inches()])*72) for ax in axes]
    (args.out/'example_panel_geometry.json').write_text(json.dumps({'units':'pt','rectangles_xywh':panel_rects},indent=2))
    try:
        from audit_panel_alignment import require_matplotlib_panel_alignment
    except ImportError:
        pass
    else:
        require_matplotlib_panel_alignment(fig,json_out=args.out/'example_panel_alignment.json')
    fig.savefig(args.out/'ABS_real_examples.pdf',facecolor='white')
    fig.savefig(args.out/'ABS_real_examples.svg',facecolor='white')
    fig.savefig(args.out/'ABS_real_examples.png',facecolor='white',dpi=300)
    plt.close(fig)
    for arr,name in zip(arrays,['mri_roi','gt_roi','eroded_roi','boundary_roi']):
        plt.imsave(args.out/f'{name}.png',arr,cmap='gray',vmin=0,vmax=1)
    fig,axes=plt.subplots(1,2,figsize=(6,3.6))
    fig.subplots_adjust(left=.06,right=.94,top=.78,bottom=.18,wspace=.20)
    for ax,arr,title in zip(axes,[crop(b),crop(b2)],['Implemented 3D boundary','2D slice boundary (diagnostic)']):
        vector_tile(ax,arr,True,args.vector_resolution); ax.set_title(title,fontsize=10)
    fig.suptitle('A 2D contour is not the implemented supervision target',fontsize=12)
    fig.savefig(args.out/'3d_vs_2d_diagnostic.png',dpi=300,facecolor='white');plt.close(fig)
    foreground=int(target.sum()); boundary_count=int(boundary.sum())
    meta={'dataset':args.dataset,'region':args.region,'image':str(args.image) if args.image else None,
        'mask':str(args.mask) if args.mask else None,'prepared_cache':str(args.prepared) if args.prepared else None,
        'data_grid':data_grid,'boundary_operator_matches_training':True,
        'training_preprocessed_arrays_used':bool(args.prepared),
        'native_grid_note':'Native NIfTI examples demonstrate the operator; they are not the resized training targets.' if not args.prepared else None,
        'shape':list(image.shape),'spacing_native_axes_mm':spacing,
        'native_axis':args.axis if not args.prepared else None,'array_axis':args.axis,
        'slice_index_zero_based':index,'display_rotation':'np.rot90(k=1)',
        'roi_rotated_rows_cols':list(roi),'roi_selection':'largest 8-connected target component on selected slice, with margin',
        'slice_selection':'max_target_area' if args.slice_index is None else 'explicit',
        'operator':'Y_b = Y - (avg_pool3d(Y,kernel_size=3,stride=1,padding=1,count_include_pad=True)>0.999)',
        'operator_units':'voxels; not a constant physical-width surface on anisotropic data',
        'volume_foreground_voxels':foreground,'volume_boundary_voxels':boundary_count,
        'volume_boundary_fraction':boundary_count/foreground,
        'slice_boundary_3d_vs_2d_differing_voxels':int((b!=b2).sum()),
        'image_normalization_display_only':{'method':'identity (cached training normalization)' if args.prepared else 'whole-volume min-max','min':lo,'max':hi},
        'prediction_maps_present':False,'checkpoint_used':None,
        'vector_display_resolution_limit':args.vector_resolution,'numerical_arrays_downsampled':False,
        'boundary_weight_training':.1,'verification':verification}
    (args.out/'metadata.json').write_text(json.dumps(meta,indent=2,ensure_ascii=False),encoding='utf-8')
    print(json.dumps({'output':str(args.out),'slice':index,'roi':roi,'boundary_fraction':boundary_count/foreground,
                      'verification':verification},ensure_ascii=False))

if __name__=='__main__': main()
