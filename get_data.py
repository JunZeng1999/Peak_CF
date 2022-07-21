import os
import json
import pymzml
import numpy as np
from tqdm import tqdm
from bintrees import FastAVLTree
import matplotlib.pyplot as plt


def construct_EIC(eic_dict):
    """
    Construct an EIC object from dict
    :param eic: a dict with 'description' (not necessary),
                            'code' (basically the name of file, not necessary),
                            'label' (annotated class),
                            'number of peaks' (quantity of peaks within ROI),
                            'begins' (a list of scan numbers),
                            'ends' (a list of scan numbers),
                            'intersections' (a list of scan numbers),
                            'scan' (first and last scan of EIC),
                            'rt',
                            'intensity',
                            'mz'
    """
    return EIC(eic_dict['scan'], eic_dict['rt'], eic_dict['intensity'], eic_dict['mz'], np.mean(eic_dict['mz']))


class EIC:
    def __init__(self, scan, rt, i, mz, mzmean):
        self.scan = scan
        self.rt = rt
        self.i = i
        self.mz = mz
        self.mzmean = mzmean

    def __repr__(self):
        return 'mz = {:.4f}, rt = {:.2f} - {:.2f}'.format(self.mzmean, self.rt[0], self.rt[1])

    def save_annotated(self, path, code=None, label=0, number_of_peaks=0, peaks_labels=None, borders=None,
                             description=None):
        roi = dict()
        roi['code'] = code
        roi['label'] = label
        roi['number of peaks'] = number_of_peaks
        roi["peaks' labels"] = [] if peaks_labels is None else peaks_labels
        roi['borders'] = [] if borders is None else borders
        roi['description'] = description
        roi['rt'] = self.rt
        roi['scan'] = self.scan
        roi['intensity'] = list(map(float, self.i))
        roi['mz'] = list(map(float, self.mz))

        with open(path, 'w') as jsonfile:
            json.dump(roi, jsonfile)


class ProcessEIC(EIC):
    def __init__(self, scan, rt, i, mz, mzmean):
        super().__init__(scan, rt, i, mz, mzmean)
        self.points = 1


def get_EICs(path, delta_mz=0.005, required_points=15, dropped_points=3, progress_callback=None):
    '''
    :param path: path to mzml file
    :param delta_mz:
    :param required_points:
    :param dropped_points: can be zero points
    :param pbar: an pyQt5 progress bar to visualize
    :return: EICs - a list of EIC objects found in current file
    '''
    # read all scans in mzML file
    run = pymzml.run.Reader(path)
    scans = []
    for scan in run:
        if scan.ms_level == 1:
            scans.append(scan)

    EICs = []  # completed EICs
    process_EICs = FastAVLTree()  # processed EICs

    # initialize a processed data
    number = 1  # number of processed scan
    init_scan = scans[0]
    start_time = init_scan.scan_time[0]

    min_mz = max(init_scan.mz)
    max_mz = min(init_scan.mz)
    for mz, i in zip(init_scan.mz, init_scan.i):
        if i != 0:
            process_EICs[mz] = ProcessEIC([1, 1],
                                          [start_time, start_time],
                                          [i],
                                          [mz],
                                          mz)
            min_mz = min(min_mz, mz)
            max_mz = max(max_mz, mz)

    for scan in tqdm(scans):
        if number == 1:  # already processed scan
            number += 1
            continue
        # expand EIC
        for n, mz in enumerate(scan.mz):
            if scan.i[n] != 0:
                ceiling_mz, ceiling_item = None, None
                floor_mz, floor_item = None, None
                if mz < max_mz:
                    _, ceiling_item = process_EICs.ceiling_item(mz)
                    ceiling_mz = ceiling_item.mzmean
                if mz > min_mz:
                    _, floor_item = process_EICs.floor_item(mz)
                    floor_mz = floor_item.mzmean
                # choose closest
                if ceiling_mz is None and floor_mz is None:
                    time = scan.scan_time[0]
                    process_EICs[mz] = ProcessEIC([number, number],
                                                  [time, time],
                                                  [scan.i[n]],
                                                  [mz],
                                                  mz)
                    continue
                elif ceiling_mz is None:
                    closest_mz, closest_item = floor_mz, floor_item
                elif floor_mz is None:
                    closest_mz, closest_item = ceiling_mz, ceiling_item
                else:
                    if ceiling_mz - mz > mz - floor_mz:
                        closest_mz, closest_item = floor_mz, floor_item
                    else:
                        closest_mz, closest_item = ceiling_mz, ceiling_item

                if abs(closest_item.mzmean - mz) < delta_mz:
                    eic = closest_item
                    if eic.scan[1] == number:
                        # EICs is already extended (two peaks in one mz window)
                        eic.mzmean = (eic.mzmean * eic.points + mz) / (eic.points + 1)
                        eic.points += 1
                        eic.mz[-1] = (eic.i[-1] * eic.mz[-1] + scan.i[n] * mz) / (eic.i[-1] + scan.i[n])
                        eic.i[-1] = (eic.i[-1] + scan.i[n])
                    else:
                        eic.mzmean = (eic.mzmean * eic.points + mz) / (eic.points + 1)
                        eic.points += 1
                        eic.mz.append(mz)
                        eic.i.append(scan.i[n])
                        eic.scan[1] = number  # show that we extended the eic
                        eic.rt[1] = scan.scan_time[0]
                else:
                    time = scan.scan_time[0]
                    process_EICs[mz] = ProcessEIC([number, number],
                                                  [time, time],
                                                  [scan.i[n]],
                                                  [mz],
                                                  mz)
        # check and cleanup
        to_delete = []
        for mz, eic in process_EICs.items():
            if eic.scan[1] < number <= eic.scan[1] + dropped_points:
                # insert 'zero' in the end
                eic.mz.append(eic.mzmean)
                eic.i.append(0)
            elif eic.scan[1] != number:
                to_delete.append(mz)
                if eic.points >= required_points:
                    EICs.append(EIC(
                        eic.scan,
                        eic.rt,
                        eic.i,
                        eic.mz,
                        eic.mzmean
                    ))
        process_EICs.remove_items(to_delete)
        try:
            min_mz, _ = process_EICs.min_item()
            max_mz, _ = process_EICs.max_item()
        except ValueError:
            min_mz = float('inf')
            max_mz = 0
        number += 1
        if progress_callback is not None and not number % 10:
            progress_callback.emit(int(number * 100 / len(scans)))
    # add final eics
    for mz, eic in process_EICs.items():
        if eic.points >= required_points:
            for n in range(dropped_points - (number - 1 - eic.scan[1])):
                # insert 'zero' in the end
                eic.mz.append(eic.mzmean)
                eic.i.append(0)
            EICs.append(EIC(
                        eic.scan,
                        eic.rt,
                        eic.i,
                        eic.mz,
                        eic.mzmean
                        ))
    # expand constructed eic
    for eic in EICs:
        for n in range(dropped_points):
            # insert in the begin
            eic.i.insert(0, 0)
            eic.mz.insert(0, eic.mzmean)
        eic.scan = (eic.scan[0] - dropped_points, eic.scan[1] + dropped_points)
        assert eic.scan[1] - eic.scan[0] == len(eic.i) - 1
    return EICs