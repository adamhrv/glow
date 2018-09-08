import numpy as np


from PIL import Image
from os.path import join
import cv2 as cv
from pathlib import Path
from glob import glob
import imutils

def main(args):

    #from model import encode, manipulate_range, mix_range
    #from align_face import align

    files_a = glob(join(args['a'],"*.jpg"))
    files_a += glob(join(args['a'],'*.JPG'))
    files_b = glob(join(args['b'],"*.jpg"))
    
    for file_a in files_a:
      for file_b in files_b:
        fname = '{}_x_{}'.format(Path(file_a).stem, Path(file_b).stem)
        ims = get_mixs(file_a, file_b, args['points'], args['expand'])
        # resize
        ims = [imutils.resize(x, width=args['width']) for x in ims]
        ims = [cv.cvtColor(x, cv.COLOR_BGR2RGB) for x in ims]
        # get middle
        im_middle = ims[len(ims)//2]
        fp_out = join(args['output'], '{}.jpg'.format(fname))
        cv.imwrite(fp_out, im_middle)
        # make grid
        im_grid = montage(ims, ncols=3)
        fp_out = join(args['output'],'{}_grid.jpg'.format(fname))
        cv.imwrite(fp_out, im_grid)
        print('[+] ', fp_out)



def montage(frames,ncols=4,width=None):
  """Convert list of frames into a grid montage
  param: frames: list of frames as Numpy.ndarray
  param: ncols: number of columns
  param: width: resize images to this width before adding to grid
  returns: Numpy.ndarray grid of all images
  """
  
  rows = []
  for i,im in enumerate(frames):
    if width is not None:
      im = imutils.resize(im,width=width)
    h,w = im.shape[:2]
    if i % ncols == 0:
      if i > 0:
        rows.append(ims)
      ims = []
    ims.append(im)
  if len(ims) > 0:
    for j in range(ncols-len(ims)):
      ims.append(np.zeros_like(im))
    rows.append(ims)
  row_ims = []
  for row in rows:
    row_im = np.hstack(np.array(row))
    row_ims.append(row_im)
  contact_sheet = np.vstack(np.array(row_ims))
  return contact_sheet


# Reshape multiple images to a grid
def ims2grid(im,h,w,mirror=False):
    im = np.reshape(im, [h, w, 256, 256, 3])
    im = np.transpose(im, [0,2,1,3,4])
    if mirror:
        im = im[:,:,::-1,:,:] ## reflect width wise
    im = np.reshape(im, [h*256, w*256, 3])
    return im

def resize(arr, res, ratio=1.):
    shape = (int(res*ratio),res)
    return np.array(Image.fromarray(arr).resize(shape, resample=Image.ANTIALIAS))

def get_mixs(name1, name2, points, expand):
    im1 = align(name1, expand=expand)
    im2 = align(name2, expand=expand)
    z1 = encode(im1)
    z2 = encode(im2)
    ims, _ = mix_range(z1, z2, points)
    return ims

if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('-a',required=True,help='Image A')
    ap.add_argument('-b',required=True,help='Image B')
    ap.add_argument('--width',default=600,help='Output resolution')
    ap.add_argument('--grid',action='store_true',help='Save grid')
    ap.add_argument('-p','--points',default=9, type=int, help='Number of interpolations')
    ap.add_argument('-o','--output',required=True,help='Output directory')
    ap.add_argument('-e','--expand', default=0, type=int, help='Expand pixels')
    
    args = vars(ap.parse_args())
    from imageio import  mimwrite, get_writer
    from model import encode, manipulate_range, mix_range
    from align_face import align
    main(args)