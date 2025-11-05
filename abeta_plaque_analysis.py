# %% <initialize imageJ gateway>
import imagej
ij = imagej.init("/Applications/Fiji.app")
ij.getVersion() #prints the local Fiji version

# %% <import modules>
import skimage as sk
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scyjava import jimport
import os
import scipy
import cv2
import pandas as pd
import math
from scipy import ndimage as ndi

#%% <add imageplus>
IJ=jimport("ij.IJ")
ImagePlus = jimport("ij.ImagePlus")

#%% <functions>
def open_image(img):
    jimage=ij.io().open(img) #as imageJ Dataset
    imp = ij.convert().convert(jimage, ImagePlus)
    return jimage, imp
def z_stack_projection(img):
    ZProjector = jimport("ij.plugin.ZProjector") #make sure its a ImagePlus Composite image
    projection_type= "max"
    z_proj = ZProjector.run (img, projection_type) #run z projection on max intensity
    return z_proj
def split_channels (img):
    split=img.splitChannels(True)
    chan_1=ij.py.from_java(split[0])
    chan_2=ij.py.from_java(split[1])
    return chan_1, chan_2

def filtering(img, top_thresh, low_thresh):
    denoise=sk.restoration.denoise_wavelet(img)
    blurred = sk.filters.gaussian(denoise, sigma=2.0)
    t_thresh=sk.filters.threshold_otsu(blurred)
    if t_thresh/np.median(blurred)<=1.5:
        thresh= t_thresh + np.percentile(blurred, top_thresh)
    elif t_thresh/np.median(blurred)>=3:
        thresh= t_thresh + np.percentile(blurred, 20)
    else:
        thresh= t_thresh + np.percentile(blurred,low_thresh)
    fos_cells=np.where(blurred >= thresh, 1, 0)
    filtered=ndi.median_filter(fos_cells, size=5)
    eroded=ndi.binary_erosion(filtered)
    dilated= ndi.binary_dilation(eroded, iterations=1)
    eroded=ndi.binary_erosion(dilated, iterations=2)
    filt=ndi.median_filter(eroded, size=5)
    return filt

def particle_analyzer (img):
    contours= sk.measure.find_contours(img, .8)
    label_image= sk.measure.label(img)
    regions = sk.measure.regionprops(label_image)
    props = sk.measure.regionprops_table(label_image, properties=('area', 'area_bbox', 'area_convex','area_filled','axis_major_length','axis_minor_length', 'eccentricity', 'equivalent_diameter_area', 'extent','perimeter'))
    props_table=pd.DataFrame(props)
    props_table['area_um']= props_table['area_filled']/(1.7067**2)*1.1377 #by scaling factor (image is scaled by 0.5)
    filtered_df=props_table[props_table['area_um']>= 250] #250 um , each pixel is 264.38 microns
    filtered_df=filtered_df[filtered_df['eccentricity']<= 0.9]
    filtered_df=filtered_df[(filtered_df['axis_minor_length']/filtered_df['axis_major_length']) >= axis_ratio]
    return filtered_df

def show_labels(img, img_original):
    label_image= sk.measure.label(img)
    image_label_overlay = sk.color.label2rgb(label_image, image=img, bg_label=0)
    f, (ax1, ax2)=plt.subplots(1,2)
    #fig, ax2 = plt.subplots(figsize=(10, 6))
    ax1.imshow(img_original, cmap="gray")
    ax2.imshow(image_label_overlay)

    for region in sk.measure.regionprops(label_image):
        # take regions with large enough areas
        if region.area >= (250*2.19*1.1377):
            if region.eccentricity <=(0.9):
            # draw rectangle around segmented coins
                minr, minc, maxr, maxc = region.bbox
                rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr,
                                          fill=False, edgecolor='red', linewidth=2)
                ax2.add_patch(rect)


    ax1.set_axis_off()
    ax2.set_axis_off()
    plt.tight_layout()
    plt.show()

def save_labels(img, img_original, save_path, name):
    label_image= sk.measure.label(img)
    image_label_overlay = sk.color.label2rgb(label_image, image=img, bg_label=0)
    f, (ax1, ax2)=plt.subplots(1,2)
    ax1.imshow(img_original, cmap="gray_r")
    ax2.imshow(image_label_overlay, cmap="gray_r")

    for region in sk.measure.regionprops(label_image):
        # take regions with large enough areas
        if region.area >= (250*2.19*1.1377):
            if region.eccentricity <=(0.9):
                if region.axis_minor_length / region.axis_major_length >= axis_ratio:
            # draw rectangle around segmented coins
                    minr, minc, maxr, maxc = region.bbox
                    rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr,
                                          fill=False, edgecolor='red', linewidth=1)
                    ax2.add_patch(rect)

    ax1.set_axis_off()
    ax2.set_axis_off()
    plt.tight_layout()
    plt.savefig(save_path+"/"+str(name)+".png")

def show_original_filt (original, filt):
    fig, (ax1, ax2) = plt.subplots(1,2)
    ax1.imshow(original, cmap="gray_r")
    ax2.imshow(filt, cmap="gray_r")
    ax1.set_axis_off()
    ax2.set_axis_off()
    plt.tight_layout()

#%% <define path and create list with file names>

dir_path="PathGoesHere"
all_files=[]
#type(all_files)

for filename in os.listdir(dir_path):
    if filename.endswith(".lsm"):
        # ONLY for ming's images
        #if "HPC" in filename:
        all_files.append(filename)
print(all_files)
len(all_files)


#%% <create empty dataframe>
df= pd.DataFrame(columns= ['area', 'area_bbox', 'area_convex','area_filled','axis_major_length','axis_minor_length', 'eccentricity', 'equivalent_diameter_area', 'extent','perimeter'])

#%% <analyze and create table>
top_thresh=98
low_thresh=95
axis_ratio= 0.5

#%%
not_read=[]
save_path="PathGoesHere"

#%%
for filename in all_files:
    try:
        name=filename.strip(".lsm")
        image_path=dir_path + "/" + filename
        jimage, imp= open_image(image_path)
        z_proj= z_stack_projection (imp)
        red_chan, green_chan= split_channels(z_proj)
        img_filtered=filtering(red_chan, 98, 95)
        particle_analysis_table = particle_analyzer(img_filtered)
        particle_analysis_table["image_id"]= str(filename)
        #save_labels(img_filtered, red_chan, save_path, name)
        try:
            imp.close()
        except:
            print("not closed")
        df=df.append(particle_analysis_table)
    except:
        print(filename + " not readable")
        not_read.append(filename)

df.to_csv("PathGoesHere/abeta_analysis.csv")

# %%
print(len(not_read))
