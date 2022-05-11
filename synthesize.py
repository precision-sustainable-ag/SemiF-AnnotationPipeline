import random
import uuid
from pathlib import Path

import cv2
import numpy as np
from omegaconf import DictConfig
from tqdm import trange

from semif_utils.mongo_utils import Connect, from_db, to_db
from synth_utils_v2 import (POTMAPS, SynthDataContainer, bbox_iou,
                            center_on_background, get_img_bbox, img2RGBA,
                            transform_position)


class SynthPipeline:

    def __init__(self, datacontainer, cfg: DictConfig) -> None:
        self.synth_config = cfg.synth
        self.count = cfg.synth.count
        self.back_config = cfg.synth.backgrounds
        self.pot_config = cfg.synth.pots

        self.backgrounds = datacontainer.backgrounds
        self.used_backs = []

        self.pots = datacontainer.pots
        self.used_pots = []

        self.cutouts = datacontainer.cutouts
        self.used_cutouts = []

        self.replace_backs = self.replace_backgrounds()

#---------------------- Get images -------------------------------

    def replace_backgrounds(self):
        """Checks if count is larger than total number of backgrounds. Returns True for replacements. 
        """
        return True if self.count > len(self.backgrounds) else False

    def get_backs(self, sortby=None):
        if self.replace_backs:
            self.back = random.choice(self.backgrounds)
        else:
            self.backgrounds = [x for x in self.backgrounds if x not in self.used_backs]
            self.back = random.choice(self.backgrounds)
        self.used_backs.append(self.back)
        
        return self.back

    def get_pots(self, num_pots, sortby=None):
        self.pots = [x for x in self.pots if x not in self.used_pots]
        self.pots = random.sample(self.pots, num_pots)
        self.used_pots.append(self.pots)
        return self.pots

    def get_cutouts(self, num_cutouts, sortby=None):
        self.cutouts = [x for x in self.cutouts if x not in self.used_cutouts]
        self.cutouts = random.sample(self.cutouts, num_cutouts)
        self.used_cutouts.append(self.cutouts)
        return self.cutouts

#------------------- Overlap checks --------------------------------

    def check_overlap(self,x, y,potshape, pot_positions):  # x = w ; h = y
        """Check overlap from list of previous bbox coordinates"""
        if not None in pot_positions:
            new_bbox = get_img_bbox(x, y, potshape)
            for y_new, x_new, oldpotshape in pot_positions:
                old_bbox = get_img_bbox(x_new, y_new, oldpotshape)
                
                iou = bbox_iou(old_bbox, new_bbox)    
                while iou > 0.05:
                    x, y = x + random.randint(-2500, 2500), y + random.randint(-2000, 2000)
                    new_bbox = get_img_bbox(x, y, potshape)
                    iou = bbox_iou(old_bbox, new_bbox)
                    print(iou)
            
            x, y = new_bbox[0], new_bbox[1]
        return x, y


    def check_negative_positions(self, h0, w0, pot):
        """ Crops pot image if position coordinates are negative. 
            Crop amount is absolute value of negative position value. """
        if w0 < 0:
            pot = pot[:, abs(w0):]
            w0 = 0

        if h0 < 0:
            pot = pot[abs(h0):, :]
            h0 = 0
        return h0, w0, pot


    def check_positive_position(self, h0, w0, potshape, backshape, pot):
        """ Crops pot image if position coordinates extend beyond background frame in positive direction.
        """
        pot_h, pot_w, _ = potshape
        back_h, back_w, _ = backshape

        if w0 + pot_w > back_w:
            back_w = back_w - w0
            pot = pot[:, :back_w]

        if h0 + pot_h > back_h:
            back_h = back_h - h0
            pot = pot[:back_h,:]
        return pot

#-------------------------- Overlay and blend --------------------------------------

    def blend(self, h0, w0, fore, back, mask=None):
        # image info
        fore_h, fore_w, _ = fore.shape    
        h1 = h0 + fore_h
        w1 = w0 + fore_w
        # masks
        fore_mask = fore[..., 3:] / 255
        alpha_l = 1.0 - fore_mask
        # blend
        back[h0:h1,w0:w1] = alpha_l * back[h0:h1,w0:w1] + fore_mask* fore 
        if mask is None:
            return back, h0, w0
        else:
            mask[h0:h1,w0:w1] = (255) * fore_mask + mask[h0:h1,w0:w1] * alpha_l
            return back, mask, h0, w0
    
    def overlay(self, x0, y0, fore, back, mask=None):
        # check positions
        y0, x0, fore  = self.check_negative_positions(y0, x0, fore)
        fore = self.check_positive_position(y0, x0, fore.shape, back.shape, fore)
        return self.blend(y0, x0, fore, back, mask=mask)
    
    #---------------------- Save to directory and DB --------------------------------

    def save_synth(res, mask, backpath, potpath, cutpath):
        savedir = "../data/quick_synth_test/synth_results/"
        fstem = uuid.uuid4().hex
        fname = fstem + ".png"
        fmaskname = fstem + "_mask.png"
        savepath = Path(savedir,fname)
        savemask = Path(savedir,fmaskname)
        res = cv2.cvtColor(res, cv2.COLOR_RGBA2BGRA)
        mask = cv2.cvtColor(mask, cv2.COLOR_RGBA2BGRA)
        cv2.imwrite(str(savepath), res)
        cv2.imwrite(str(savemask), mask)

#-------------------------- Pipeline -------------------------------------

    def pipeline(self):
        # Config pot placement info
        potmaps = random.choice(POTMAPS)
        spread_factor = random.randint(0, 100)
        
        # Get synth data samples
        self.back = self.get_backs()
        back = cv2.cvtColor(cv2.resize(self.back.array, (9600,6450)), cv2.COLOR_BGR2RGBA)
        self.pots = self.get_pots(len(potmaps))
        self.cutouts = self.get_cutouts(len(potmaps))
        cutout_zero_mask = np.zeros_like(back)

        pot_positions = []
        for potidx in range(len(potmaps)):

            # Pot info
            pot_position = potmaps[potidx]
            pot = random.choice(self.pots).array
            potshape = pot.shape
            
            # Unique cutout
            cutout = img2RGBA(random.choice(self.cutouts).array)
            cutoutshape = cutout.shape

            # Check coordinates for pot
            x, y = transform_position(pot_position,potshape, spread_factor)
            x, y = self.check_overlap(x, y,potshape, pot_positions)
            
            #Overlay pot on background
            potted, poty, potx = self.overlay( y, x, pot, back)
            pot_positions.append([poty, potx, potshape])
            
            # Get cutout position from pot position
            cutx, cuty = center_on_background(poty, potx, potshape, cutoutshape)
            # Overlay cutout on pot
            res, mask, y0, x0  = self.overlay(cutx, cuty, cutout, potted, mask=cutout_zero_mask)

            return res, mask
            

def main(cfg: DictConfig) -> None:
    # Create synth data container
    syn = SynthDataContainer(datadir=cfg.synth.datadir, db_db="test")
    # Call pipeline
    gen = SynthPipeline(syn, cfg)
    # Iterate over image
    for cnt in trange(cfg.synth.count):
        gen.pipeline()
    print("----- END -----")

