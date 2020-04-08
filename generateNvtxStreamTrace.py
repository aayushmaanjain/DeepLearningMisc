#!/usr/bin/env python
# generateNvtxStreamTrace.py 
#
# Created by Aayushmaan Jain
#
# This script takes nvprof output(s) generated with NVTX markers in csv format
# as input and generates a chrome-tracing compatible json for visualizing ops/
# kernels.
#
# This works for single as well as multi-gpu nvprof profiling outputs. In case
# of multi-gpu profiling, pass space separated list of profiling output files
# as arguments, preceded by the '-f' flag.  

import argparse
from collections import namedtuple
import json
import numpy as np
import pandas as pd
import re
import torch

Marker = namedtuple('Marker', 'index name depth')

def parseNvtxFile(filename):
    initdf = pd.read_csv(filename, skiprows=6,
                         names=['start','duration','gridX','gridY','gridZ','blockX','blockY','blockZ',
                                'registersPerThread','staticSMem','dynamicSMem','size','throughput',
                                'srcMemType','dstMemType','device','context','stream','name','corrid'])
    # staticSMem - KB, dynamicSMem - KB, size - MB, throughput - GB/s

    initdf.dropna(subset=['name'], inplace=True)
    initdf.drop(['gridX','gridY','gridZ','blockX','blockY','blockZ','srcMemType','dstMemType','device'], axis=1, inplace=True)

    # demangling the name
    initdf['name'] = initdf['name'].apply(torch._C._demangle)

    # type cast
    initdf[["start", "duration", "stream", "corrid"]] = initdf[["start", "duration", "stream", "corrid"]].apply(pd.to_numeric, errors='ignore')

    startprof = initdf.index[initdf['name'].str.contains("\[Marker\] __start_profile")].tolist()
    assert len(startprof) == 1
    stopprof = initdf.index[initdf['name'].str.contains("\[Marker\] __stop_profile")].tolist()
    assert len(stopprof) == 1
    initdf = initdf.loc[startprof[0]:stopprof[0],:]

    print("initdf generated from {}: {}".format(filename, initdf.shape))
    return initdf


def mapOpsToCorrids(markers):
    stack = []
    opsToCorrid = {}  # marker operation index -> cuda launch kernel correlation id
    nopCorrids = []

    for index, row in markers.iloc[1:-1].iterrows():
        name = row['name']
        if "pin_memory" in name:
            continue

        if "[Range start]" in name:
            pat = re.compile(r'\[Range start\] (?P<name>[a-zA-Z0-9_:]*), (seq = \d+)?(, )?(?P<size>sizes = \[[\[\],\d ]*\])? \(Domain: \<unnamed\>\)')
            details = pat.match(name)
            if not details:
                print(" *** Error handling regex name matching:{}".format(name))
                continue
            mname = details.group('name')
            if details.group('size'):
                mname = mname + ", " + details.group('size')
            marker = Marker(index, mname, len(stack))
            #print("pushing into stack: {}, {}".format(index, row['name']))
            stack.append(marker)
            opsToCorrid[marker] = []
            
        elif "[Range end]" in name:
            marker = stack.pop()
            top = markers.loc[marker.index, 'name']
            match = top.replace("start","end")
            tmpst = []
            while(len(stack) and (match != name)):
                tmpst.append(marker)
                marker = stack.pop()
                match = markers.loc[marker.index, 'name'].replace("start", "end")
            
            #if len(tmpst):
            #    print(" *** does not match; this shouldn't happen ideally: {}".format(index))
            
            while(len(tmpst)):
                m = tmpst.pop()
                stack.append(m)

        elif (name == "cudaLaunchKernel") or ("cudaMemcpy" in name):
            for marker in stack:
                opsToCorrid[marker].append(row['corrid'])
            if len(stack) == 0:
                #print(" *** Kernel with corrid: {} doesn't lie between any markers".format(row['corrid']))
                nopCorrids.append(int(row['corrid']))
                
        # hack
        elif "nccl" in name:
            marker = Marker(index, "nccl", 0)
            opsToCorrid[marker] = [row['corrid']]
     
        else:
            print("ERROR: unhandled case while generating opsToCorrid.")

    # Deleting markers that don't contain any kernel
    delkeys = []
    for i, corrids in opsToCorrid.items():
        if len(corrids) == 0:
            delkeys.append(i)

    for key in delkeys:
        opsToCorrid.pop(key, None)
    
    return opsToCorrid, nopCorrids


def mapCorridsToKernels(df, opToCorrids):
    # Get a list of all corrids
    allCorrids = []
    for _, corrids in opToCorrids.items():
        allCorrids.extend(corrids)
    allCorrids = set(allCorrids)

    # Map corrid to Kernel
    corridToKernel = {}  # cuda launch kernel correlation id -> kernel with corresponding correlation id
    for corrid in allCorrids:
        rowIndex = df.index[df['corrid'] == int(corrid)].tolist()
        assert len(rowIndex) == 1, "multiple kernels with same corrid {}: {}".format(corrid, rowIndex)
        corridToKernel[corrid] = df.loc[rowIndex[0]]
    
    return corridToKernel


def mapOpsToKernels(initdf):

    markers = initdf[(initdf['name'].str.contains("\[Range start\]")) | (initdf['name'].str.contains("\[Range end\]")) \
                     | (initdf['name'].str.contains("Marker")) \
                     | (initdf['name'] == "cudaLaunchKernel") | (initdf['name'].str.contains("cudaMemcpy"))
                     | (initdf['name'].str.contains("nccl"))]

    opToCorrids, nopCorrids = mapOpsToCorrids(markers)
    
    # Remove all markers to get a df containing all kernels 
    df = initdf.dropna(subset=['registersPerThread','staticSMem','dynamicSMem','size','throughput'], how='all')

    # Kernels that don't lie between markers and consequently won't be added to the chrometrace
    # TODO improvement: add these kernels to chrometrace.
    print('-' * 30)
    print("Kernels not belonging to any op:", len(nopCorrids))
    print("NOTE: These kernels won't appear in the output chrometrace.")  
    nopKernels = df[df['corrid'].isin(nopCorrids)]
    print(nopKernels['name'].value_counts())
    print('-' * 30)

    # Getting kernel details from op corrids 
    corridToKernel = mapCorridsToKernels(df, opToCorrids)

    opToKernels = {}  # pytorch op marker -> index of corresponding kernel call

    for opIndex, corrids in opToCorrids.items():
        opToKernels[opIndex] = []
        for corrid in corrids:
            opToKernels[opIndex].append(corridToKernel[corrid])

    return opToKernels


def getNvtxTraceFromDF(initdf, pid):
    opToKernels = mapOpsToKernels(initdf) 

    events = []
    for marker, kernels in opToKernels.items():
        if marker.depth > 0:
            continue
        streamToKernel = {}
        for kernel in kernels:
            stream = int(kernel['stream'])
            if stream not in streamToKernel.keys():
                streamToKernel[stream] = []
            streamToKernel[stream].append(kernel)
            
        #opStartTime = min(kernel['start'] for kernel in kernels)
        #opEndTime =  max((kernel['start'] + kernel['duration']) for kernel in kernels)
        
        for stream, kernels in streamToKernel.items():
            streamStartTime = min(kernel['start'] for kernel in kernels)
            streamEndTime = max((kernel['start'] + kernel['duration']) for kernel in kernels)
            cat = marker.name.split(',')[0]
            events.append({"name": marker.name,
                           "cat": cat,
                           "ph": "X", 
                           "ts": streamStartTime, 
                           "dur": round(streamEndTime - streamStartTime, 3), 
                           "tid": stream, 
                           "pid": pid, 
                           "args": {"numKernels": len(kernels)}})
                
            for kernel in kernels:
                events.append({"name": kernel['name'],
                               "cat": cat,
                               "ph": "X", 
                               "ts": kernel['start'], 
                               "dur": round(kernel['duration'], 3), 
                               "tid": stream, 
                               "pid": pid, 
                               "args": {}})    

    print("#Events captured for pid {}: {}".format(pid, len(events)))
    return events


def main(args):
    print(args.nvtx_files)
    allEvents = []
    for i, filename in enumerate(args.nvtx_files):
        initdf = parseNvtxFile(filename) 
        events = getNvtxTraceFromDF(initdf, i)
        allEvents.extend(events)
    
    with open(args.output, 'w') as ofile:
        #json.dump(allEvents, ofile)
        ofile.write(
                '[' +
                ',\n'.join(json.dumps(event) for event in allEvents) +
                ']\n')
 

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--nvtx-files', nargs='+', required=True, help="filepath(s) to NVTX csv file(s)")
    parser.add_argument('-o', '--output', type=str, default="chrome_trace.json", help="output json filename")

    args = parser.parse_args()
    
    main(args)
