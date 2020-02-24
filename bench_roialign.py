import argparse
import numpy as np
import pandas as pd
import time
import torch

from detectron2.layers.roi_align import ROIAlign

ITERATIONS = 100

def measure_roialign_perf(input_shape, roi_shape, output_size, spatial_scale,
                          sampling_ratio=0, aligned=True):
    """
    Args:
        input: NCHW images
        rois: Bx5 boxes. First column is the index into N. The other 4 columns
            are xyxy.
        output_size (tuple): h, w
        spatial_scale (float): scale the input boxes by this number
        sampling_ratio (int): number of inputs samples to take for each output
            sample. 0 to take samples densely.
        aligned (bool): if False, use the legacy implementation in Detectron.
            If True, align the results more perfectly.
    """
    assert roi_shape[1] == 5, "ERROR: ROI shape expected to be of form (m,5)"
    
    # Preparing Inputs
    n = input_shape[0]
    b = roi_shape[0]
    inputbatch = torch.randn(input_shape, dtype=torch.float, requires_grad=True)
    # creating ROI tensor - shape (b,5)
    # RoI tensor [:, 1:] contains coordiantes of bounding boxes - xyxy.
    # (100,1200) range chosen based on COCO max image size.
    bboxes = torch.FloatTensor(roi_shape[0], 4).uniform_(100,1200)
    # First column of RoI tensor maps bounding box to image in batch.
    # Based on my observations, the boxes are ordered by image index in batch,
    # ie all boxes corresponding to first image first, then for the second
    # image, third image and so on.
    boxToNMapping = torch.tensor(
        np.expand_dims(np.array([i * n // b for i in range(b)]), axis=1),
        dtype=torch.float)
    roi = torch.cat((boxToNMapping, bboxes), dim=1)
    roi.requires_grad=True
    #print(inputbatch.shape, roi.shape)

    # Defining Op
    roi_align = ROIAlign(output_size, spatial_scale, sampling_ratio, aligned)
    
    roi_align.cuda()    
    inputbatch = inputbatch.cuda()
    roi = roi.cuda()

    # Forward Pass
    # warmup - 2 iters
    roi_align.forward(inputbatch, roi)
    roi_align.forward(inputbatch, roi)
    
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(ITERATIONS):
        #output = roi_align.forward(inputbatch.cuda(), roi.cuda()) 
        output = roi_align.forward(inputbatch, roi) 
    torch.cuda.synchronize()
    end = time.time()       
    fwd_time = (end - start) * 1000 / ITERATIONS

    # Backward Pass
    # required hack to call backward()
    output_sum = output.sum()
    # warmup
    output_sum.backward(retain_graph=True)
    output_sum.backward(retain_graph=True)

    torch.cuda.synchronize()
    bwd_start = time.time()
    for _ in range(ITERATIONS):
        output_sum.backward(retain_graph=True)
    torch.cuda.synchronize()
    bwd_end = time.time()       
    bwd_time = (bwd_end - bwd_start) * 1000 / ITERATIONS
    
    return fwd_time, bwd_time


def run_roialign(input_df):
    result_df = input_df.copy()

    for index, row in input_df.iterrows():
        input_shape = (row['n'], row['c'], row['h'], row['w'])
        roi_shape = (row['roi_dimA'], row['roi_dimB'])
        output_size = (row['output_h'], row['output_w'])
        spatial_scale = row['scale']
        sampling_ratio = row['sampling_ratio']
        aligned = True if row['aligned'] == "True" else False 
        fwd_time, bwd_time = measure_roialign_perf(input_shape, roi_shape, output_size, spatial_scale, sampling_ratio, aligned)
        result_df.loc[index, 'fwd_time(ms)'] = fwd_time
        result_df.loc[index, 'bwd_time(ms)'] = bwd_time

    return result_df


def main(filename):
    input_df = pd.read_csv(filename, header=0)
    result_df = run_roialign(input_df)
    print(result_df)
    processfilename = filename.split('/')
    outputfile = ('/').join(processfilename[:-1]) + "maskrcnn-roialign-result.csv"
    result_df.to_csv(outputfile, index=False) 


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="filepath to csv containing roi input sizes")
    parser.add_argument("--iterations", type=int, help="filepath to hip_api_trace.txt")

    args = parser.parse_args()
    
    if args.iterations:
        ITERATIONS = args.iterations

    main(args.input)
