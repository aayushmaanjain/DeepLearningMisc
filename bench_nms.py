import argparse
import numpy as np
import random
import pandas as pd
import time
import torch

from detectron2.layers import batched_nms

ITERATIONS = 100

def measure_nms_perf(boxes_shape, scores_shape, levels_shape, threshold):
    """
    Args:
    """
    assert len(boxes_shape) == 2
    assert len(scores_shape) == 1
    assert len(levels_shape) == 1
    assert boxes_shape[0] == scores_shape[0]
    assert boxes_shape[0] == levels_shape[0]

    # Preparing Inputs
    # (0,1100) range chosen based on boxes observed in runs of detectron.
    boxes = torch.FloatTensor(boxes_shape[0], boxes_shape[1]).uniform_(0,1100)

    # creating a random distribution between [-0.8, 0.8)
    scores_per_img = 1.6 * torch.rand(scores_shape, dtype=torch.float) - 0.8

    if levels_shape[0] > 8000:
        # max lvl value = 4. First 2000 entries: 0, next 2000 entries: 1, ... : 3, remaining entries: 4
        lvl = torch.tensor(np.array([i/2000 for i in range(levels_shape[0])]), dtype=torch.long)
    else:
        # overdoing simple things
        lower_bound = levels_shape[0] // 5
        upper_bound = levels_shape[0] // 4
        np_lvl = []
        count  = 0
        for lvl in range(4):
            tmp_shape = (random.randint(lower_bound, upper_bound))
            tmp = np.full(tmp_shape, lvl, dtype=int)
            #tmp = np.array([lvl for i in range(random.randint(lower_bound, upper_bound))])
            np_lvl.append(tmp)
            count += len(tmp)

        #np_lvl.append(np.array([4 for _ in range(levels_shape - count)]))
        np_lvl.append(np.full(levels_shape[0] - count, 4, dtype=int))
        lvl = torch.tensor(np.concatenate(np_lvl), dtype=torch.long)
    
    assert lvl.shape == levels_shape, "ensure lvl shape is correct"    

    boxes = boxes.cuda()    
    scores_per_img = scores_per_img.cuda()
    lvl = lvl.cuda()

    # Forward Pass
    # warmup - 2 iters
    batched_nms(boxes, scores_per_img, lvl, threshold)
    batched_nms(boxes, scores_per_img, lvl, threshold)
    
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(ITERATIONS):
        batched_nms(boxes, scores_per_img, lvl, threshold)
    torch.cuda.synchronize()
    end = time.time()       
    fwd_time = (end - start) * 1000 / ITERATIONS

    return fwd_time


def run_batched_nms(input_df):
    result_df = input_df.copy()

    for index, row in input_df.iterrows():
        boxes_shape = (int(row['boxes_dimA']), int(row['boxes_dimB']))
        scores_shape = (int(row['scores_dimA']),)
        levels_shape = (int(row['lvl_dimA']),)
        threshold = row['thresh']
        fwd_time = measure_nms_perf(boxes_shape, scores_shape, levels_shape, threshold)
        result_df.loc[index, 'fwd_time(ms)'] = fwd_time

    return result_df


def main(filename):
    input_df = pd.read_csv(filename, header=0)
    result_df = run_batched_nms(input_df)
    print(result_df)
    processfilename = filename.split('/')
    outputfile = ('/').join(processfilename[:-1]) + "maskrcnn-nms-results.csv"
    result_df.to_csv(outputfile, index=False) 


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="filepath to csv containing roi input sizes")
    parser.add_argument("--iterations", type=int, help="filepath to hip_api_trace.txt")

    args = parser.parse_args()
    
    if args.iterations:
        ITERATIONS = args.iterations

    main(args.input)
