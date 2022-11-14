"""
Author: Ivo Gollini Navarrete
Date: 21/august/2022
Institution: MBZUAI
"""

# IMPORTS
import os
import numpy as np
# import pandas as pd
import SimpleITK as sitk
import pydicom as dicom
import nibabel as nib

import torch
import torchio.transforms as tt
from torchvision.ops import masks_to_boxes
from lungmask import mask

# from tqdm import tqdm

class Preprocess:
    """
    Prepare data for experiments.
    """
    def __init__(self, datapath, dataoutput, dataset):
        self.datapath = datapath
        self.dataoutput = dataoutput
        self.dataset = dataset

        if torch.cuda.is_available():
            self.device = torch.device("cuda:0")
            print('GPU Available')
        else: 
            self.device = torch.device("cpu")
            print("GPU Not Available ")

    def msd(self):
        print('Preprocessing MSD dataset...')

        print(self.datapath)
        ct_list = sorted(os.listdir(os.path.join(self.datapath, 'imagesTr')))
        seg_list = sorted(os.listdir(os.path.join(self.datapath, 'labelsTr')))

        for pat, label in zip(ct_list, seg_list):
            patient = pat.split('.')[0]
            print(patient)

            ct = nib.load(os.path.join(self.datapath, 'imagesTr', pat))
            ct = ct.get_fdata()
            ct = np.transpose(ct, (2,1,0))
            ct = np.flip(ct, axis=1).copy()

            seg = nib.load(os.path.join(self.datapath, 'labelsTr', label))
            seg = seg.get_fdata()
            seg = np.transpose(seg, (2,1,0))
            seg = np.flip(seg, axis=1).copy()

            preprocessed_ct = self.normalize(ct[None, :, :, :])
            print('CT Normalized')
            self.save_img(patient, preprocessed_ct)
            
            preprocessed_seg = self.normalize(seg[None, :, :, :], is_label=True)
            print('Seg Normalized')
            self.save_img(patient, preprocessed_seg, is_Label=True)

            self.extract_lungs(patient, ct, seg[None, :, :, :])

        self.tumor_bbx(self.dataoutput+'/full_ct')
        self.tumor_bbx(self.dataoutput+'/lungs_roi')
        return

    def radiomics(self):

        print('Preprocessing Radiomics dataset...')

        for root, directories, files in os.walk(self.datapath, topdown=True):
            for patient in sorted(directories):
                print(patient)
                study = os.listdir(os.path.join(root, patient))[0]
                elements = sorted(os.listdir(os.path.join(root, patient, study)))
                print(elements)
                for element in elements:
                    if not "Segmentation" in element:
                        dcms_path = os.path.join(root, patient, study, element)
                        if len(os.listdir(dcms_path)) == 1: continue
                        else:
                            patient_ct = self.read_ct(dcms_path)
                            patient_ct = sitk.GetArrayFromImage(patient_ct)
                            preprocessed_ct = self.normalize(patient_ct[None, :, :, :])
                            self.save_img(patient, preprocessed_ct)
                    
                    else:
                        seg_path = os.path.join(root, patient, study, element, '1-1.dcm')
                        seg_data = dicom.read_file(seg_path)
                        seg_array = seg_data.pixel_array
                        seg_num = len(seg_data.SegmentSequence)
                        print("Seg shape: {}, with {} segmentations".format(seg_array.shape, seg_num))

                        seg_idx = 0
                        for i in range(seg_num):
                            label_idx = seg_data.SegmentSequence[i].SegmentLabel
                            if label_idx == "Neoplasm, Primary":
                                print("Primary Neoplasm in slice {}".format(i))
                                seg_idx = i
                                break

                        dim0 = int(seg_array.shape[0]/seg_num)
                        seg_tensor = torch.reshape(torch.from_numpy(seg_array), (seg_num, dim0, 512, 512))
                        patient_seg = seg_tensor[seg_idx]
                        patient_seg = patient_seg.expand(1, patient_seg.shape[0], patient_seg.shape[1], patient_seg.shape[2]).numpy()

                        preprocessed_seg = self.normalize(patient_seg, is_label=True)
                        self.save_img(patient, preprocessed_seg, is_Label=True)
                
                self.extract_lungs(patient, patient_ct, patient_seg)
            break

        self.tumor_bbx(self.dataoutput+'/full_ct')
        self.tumor_bbx(self.dataoutput+'/lungs_roi')

    def radiogenomics(self):

        print('Preprocessing Radiogenomics dataset...')

        patients_list =  sorted(os.listdir(self.datapath))
        for patient in patients_list:
            print(patient)

            patient_path = os.path.join(self.datapath, patient, 'CT')
            patient_ct = self.read_ct(patient_path)
            patient_ct = sitk.GetArrayFromImage(patient_ct)
            preprocessed_ct = self.normalize(patient_ct[None, :, :, :])
            self.save_img(patient, preprocessed_ct)

            if os.path.exists(os.path.join(self.datapath, patient, 'seg')):
                patient_seg = self.read_ct(os.path.join(self.datapath, patient, 'seg'))
                patient_seg = sitk.GetArrayFromImage(patient_seg)
                
                preprocessed_seg = self.normalize(patient_seg, is_label=True)
                self.save_img(patient, preprocessed_seg, is_Label=True)

                self.extract_lungs(patient, patient_ct, patient_seg)
                self.extract_tumor(patient, patient_ct, torch.tensor(patient_seg[0]))
            
            else:
                self.extract_lungs(patient, patient_ct)
                continue
        self.tumor_bbx(self.dataoutput+'/full_ct')
        self.tumor_bbx(self.dataoutput+'/lungs_roi')

    def extract_tumor(self, patient, ct, seg=None):
        tcr = self.roi_coord(seg, roi='tumor') # tcr = tumor_roi_coord
        tumor_ct = ct.copy()
        tumor_ct = self.crop_coord(tcr, tumor_ct)
        tumor_ct = self.normalize(tumor_ct, out_shape=(64, 64, 64))
        self.save_img(patient, tumor_ct, mode='tumor_roi')

    def extract_lungs(self, patient, ct, seg=None):

        model = mask.get_model('unet','R231').to(self.device)
        extracted = mask.apply(ct, model)
        extracted = torch.tensor(extracted)

        lungs_mask = extracted.clone()
        lcr = self.roi_coord(lungs_mask, roi='lungs') # lcr = lung_roi_coord
        lungs_ct = ct.copy()
        lungs_ct = self.crop_coord(lcr, lungs_ct)
        lungs_ct = self.normalize(lungs_ct)
        self.save_img(patient, lungs_ct, mode='lungs_roi')
        
        if seg is not None:            
            lungs_seg = seg.copy()
            lungs_seg = self.crop_coord(lcr, lungs_seg, is_label=True)
            lungs_seg = self.normalize(lungs_seg, is_label=True)
            self.save_img(patient, lungs_seg, mode='lungs_roi', is_Label=True)
    
    def crop_coord(self, coord, image, is_label=False):
        if is_label:
            image = image[:, coord[4]:coord[5], coord[2]:coord[3], coord[0]:coord[1]]
        else:
            image = image[coord[4]:coord[5], coord[2]:coord[3], coord[0]:coord[1]]
            image = image[None,:,:,:]

        return image

    def roi_coord(self, mask, roi='lungs'):
        frame_list = []
        x_min, y_min, x_max, y_max = np.inf, np.inf, -np.inf, -np.inf

        if roi == 'lungs':
            mask[mask == 2] = 1

        for i in range(len(mask)):
            if mask[i].max() > 0:
                frame_list.append(i)

                ct_slice = mask[i]
                ct_slice = ct_slice[None, :, :]

                bbx = masks_to_boxes(ct_slice)
                bbx = bbx[0].detach().tolist()

                if bbx[0] < x_min: x_min = int(bbx[0])
                if bbx[1] < y_min: y_min = int(bbx[1])
                if bbx[2] > x_max: x_max = int(bbx[2])
                if bbx[3] > y_max: y_max = int(bbx[3])

        z_min = frame_list[0]
        z_max = frame_list[-1]
        return [x_min, x_max, y_min, y_max, z_min, z_max]

    def read_ct(self, path):
            reader = sitk.ImageSeriesReader()
            dcm_names = reader.GetGDCMSeriesFileNames(path)
            reader.SetFileNames(dcm_names)
            image = reader.Execute()
            return image
    
    def save_img(self, patient, image, mode='full_ct', roi=None, is_Label=False):
        if not os.path.exists(os.path.join(self.dataoutput, mode, patient)):
            os.makedirs(os.path.join(self.dataoutput, mode, patient))
        if roi is None:
            if is_Label:
                torch.save(image, os.path.join(self.dataoutput, mode, patient, patient + '_seg.pt'))
            else:
                torch.save(image, os.path.join(self.dataoutput, mode, patient, patient + '_ct.pt'))

        else:
            if is_Label:
                torch.save(image, os.path.join(self.dataoutput, mode, patient, patient + '_' + roi + '_seg.pt'))
            else:
                torch.save(image, os.path.join(self.dataoutput, mode, patient, patient + '_' + roi + '_ct.pt'))

    def normalize(self, image, space=(1,1,1.5), out_shape=(256, 256, 256), is_label=False):
        if is_label:
            # image = tt.Resample(space, image_interpolation='nearest')(image)
            image = tt.Resize(out_shape, image_interpolation='nearest')(image)
            image = tt.RescaleIntensity(out_min_max=(0,1))(image)
        else:
            image = tt.Resample(space, image_interpolation='bspline')(image)
            image = tt.Resize(out_shape, image_interpolation='bspline')(image)
            image = tt.Clamp(out_min= -200, out_max=250)(image)
            image = tt.RescaleIntensity(out_min_max=(0,1))(image)
        return image

    def tumor_bbx(self, outpath):
        print('Generating tumor bbx ', self.dataset)
        print(outpath)
    
        patients_list = sorted(os.listdir(outpath))
        for pat in patients_list:
            print(pat)
            element_list = sorted(os.listdir(os.path.join(outpath, pat)))
            for element in element_list:
                if "_seg" not in element: continue
                patient_seg = torch.load(os.path.join(outpath, pat, element))
                tumor_cord = self.roi_coord(torch.tensor(patient_seg[0])) # x_min, x_max, y_min, y_max, z_min, z_max

                mask = np.zeros(patient_seg.shape[1:])
                mask[tumor_cord[4]:tumor_cord[5]+1, tumor_cord[2]:tumor_cord[3]+1, tumor_cord[0]:tumor_cord[1]+1] = 1
                torch.save(mask[None,:,:,:], os.path.join(outpath, pat, pat + '_bbx.pt'))