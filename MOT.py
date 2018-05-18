import os
import sys
import sqlite3
import numpy as np
import xml.etree.ElementTree as et
from collections import defaultdict
from enum import Enum, IntEnum


class Extensions(Enum):
    POLYTRACK = (1, '.sqlite')
    XML_PETS = (2, '.xml')

    def __str__(self):
        return self.value[1]


class Polytrack(IntEnum):
    OBJECT_ID = 0
    FRAME_NUMBER = 1
    X_TOP_LEFT = 2
    Y_TOP_LEFT = 3
    X_BOTTOM_RIGHT = 4
    Y_BOTTOM_RIGHT = 5


class SQLHandler:
    def __init__(self):
        pass

    def create_connection(self, file):
        connection = sqlite3.connect(file)
        return connection

    def select_tracking_info(self, connection):
        query = 'SELECT * FROM bounding_boxes'
        min_query = 'SELECT MIN(frame_number) FROM bounding_boxes'
        max_query = 'SELECT MAX(frame_number) FROM bounding_boxes'

        cursor = connection.cursor()
        cursor.execute(query)
        bboxes = cursor.fetchall()
        cursor.execute(min_query)
        min_frame = cursor.fetchall()
        cursor.execute(max_query)
        max_frame = cursor.fetchall()

        return bboxes, min_frame, max_frame


class InputData:
    def __init__(self, file):
        self.file = file
        self.sql_handler = SQLHandler()
        self.min_frame = 0
        self.max_frame = 0
        self.tracks = defaultdict(list)

    def convert_annotations(self):
        pass


class PolytrackData(InputData):
    def __init__(self, file):
        super().__init__(self)
        self.file = file

    def convert_annotations(self):
        connection = self.sql_handler.create_connection(self.file)
        with connection:
            bboxes, min_frame, max_frame = self.sql_handler.select_tracking_info(connection)
            self.min_frame = min_frame[0][0]
            self.max_frame = max_frame[0][0]
            for i in range(len(bboxes)):
                rectangle = Rectangle(bboxes[i][Polytrack.X_TOP_LEFT], bboxes[i][Polytrack.Y_TOP_LEFT],
                                      bboxes[i][Polytrack.X_BOTTOM_RIGHT], bboxes[i][Polytrack.Y_BOTTOM_RIGHT])
                custom_bbox = CustomBBox(bboxes[i][Polytrack.OBJECT_ID], rectangle)
                self.tracks[bboxes[i][Polytrack.FRAME_NUMBER]].append(custom_bbox)

    def __getitem__(self, index):
        return self.tracks[index]


class XMLPetsData(InputData):
    def __init__(self, file):
        super().__init__(self)
        self.file = file

    def convert_annotations(self):
        tree = et.parse(self.file)
        root = tree.getroot()
        for frame in root:
            frame_number = int(frame.attrib)


class Rectangle:
    def __init__(self, x_tl, y_tl, x_br, y_br):
        self.x_tl = x_tl
        self.y_tl = y_tl
        self.x_br = x_br
        self.y_br = y_br

    def width(self):
        return self.x_br - self.x_tl

    def height(self):
        return self.y_br - self.y_tl

    def area(self):
        return self.width() * self.height()

    def intersection_area(self, rect):
        dx = min(self.x_br, rect.x_br) - max(self.x_tl, rect.x_tl)
        dy = min(self.y_br, rect.y_br) - max(self.y_tl, rect.y_tl)
        if dx >= 0 and dy >= 0:
            return dx * dy
        else:
            return 0.0


class CustomBBox:
    def __init__(self, index, rectangle):
        self.index = index
        self.rectangle = rectangle


class MOTMetrics:
    def __init__(self, annotations, hypotheses):
        self.annotations = annotations
        self.hypotheses = hypotheses

    def intersection_over_union(self, cbbox_annot, cbbox_hypo, time):
        '''
        Computes the IoU between 2 CustomBBoxes at a given time t
        '''
        annotations = self.annotations[time]
        hypotheses = self.hypotheses[time]

        new_annot = self.find_correspondances(cbbox_annot, annotations)
        new_hypo = self.find_correspondances(cbbox_hypo, hypotheses)

        intersection_area = new_annot.rectangle.intersection_area(new_hypo.rectangle)
        union_area = new_annot.rectangle.area() + new_hypo.rectangle.area() - intersection_area
        if intersection_area > 0.0:
            return intersection_area / union_area
        return 0.0

    def object_exists(self, match, time, gt):
        '''
        Check if CustomBBox exists at a given time t
        '''
        tracks = []
        if gt:
            tracks = self.annotations[time]
        else:
            tracks = self.hypotheses[time]

        for track in tracks:
            if match.index == track.index:
                return True
        return False

    def find_no_match(self, matches, time, gt):
        '''
        Finds CustomBBoxes that have no matches yet
        '''
        tracks = []
        objects = []
        if gt:
            tracks = self.annotations[time]
            objects = matches.keys()
        else:
            tracks = self.hypotheses[time]
            objects = matches.values()

        objects_index = [o.index for o in objects]
        not_matched = []
        for track in tracks:
            if self.object_exists(track, time, gt) and track.index not in objects_index:
                not_matched.append(track)
        return not_matched

    def greedy_search(self, scores, threshold):
        '''
        Finds strongest match between GT and hypotheses, based on overlap ratio
        '''
        association = np.zeros(shape=(scores.shape[0], scores.shape[1]), dtype=np.int8)

        max_value = np.amax(scores)
        max_linear_position = np.argmax(scores)

        while True:
            max_position = np.unravel_index(max_linear_position, scores.shape)
            scores[max_position] = 0.0

            if max_value > threshold:
                scores[max_position[0], :] = 0.0
                scores[:, max_position[1]] = 0.0
                association[max_position] = 1

            max_value = np.amax(scores)
            max_linear_position = np.argmax(scores)

            if max_value == 0.0:
                break
        return association

    def find_correspondances(self, old_match, tracks):
        '''
        Finds the corresponding CustomBBox a time t based on the one given at time t-1
        '''
        for track in tracks:
            if track.index == old_match.index:
                return track
        return None

    def compute_metrics(self, first_instant, last_instant, overlap_ratio):
        '''
        Reference:
        Keni, Bernardin, and Stiefelhagen Rainer. "Evaluating multiple object tracking performance: the CLEAR MOT metrics." EURASIP Journal on Image and Video Processing 2008 (2008)

        Computes MOTP and MOTA as defined in the article above
        '''

        # counters for different metrics to compute MOTP and MOTA
        correct_tracks = 0
        false_positives = 0
        misses = 0
        mismatches = 0
        gt = 0
        distance = 0.0

        # dict(k,v) where k = GT_CustomBBox and v = Hypotheses_CustomBBox
        matches = {}

        for t in range(first_instant, last_instant + 1):
            previous_matches = matches.copy()

            not_valid_matches = []
            for match in matches:
                # check if current match still exists
                if self.object_exists(match, t, True) and self.object_exists(matches[match], t, False):
                    dist = self.intersection_over_union(match, matches[match], t)
                    if dist > overlap_ratio:
                        distance += dist
                    else:
                        not_valid_matches.append(match)
                else:
                    not_valid_matches.append(match)

            # lost matches since last instant
            for match in not_valid_matches:
                del matches[match]

            # get gt and hypotheses with no matches
            gt_not_matched = self.find_no_match(matches, t, True)
            hypo_not_matched = self.find_no_match(matches, t, False)

            nb_gt = len(matches) + len(gt_not_matched)
            nb_hypo = len(matches) + len(hypo_not_matched)

            # compute scores between unmatched gt and hypotheses (m x n matrix, where m = nb of not_matched gt and n = nb of not_matched_hypo)
            scores = np.zeros(shape=(len(gt_not_matched), len(hypo_not_matched)))
            i = 0
            if len(hypo_not_matched) > 0:
                for gtnm in gt_not_matched:
                    j = 0
                    for hyponm in hypo_not_matched:
                        dist = self.intersection_over_union(gtnm, hyponm, t)
                        scores[i, j] = dist
                        j += 1
                    i += 1

            # add new matches to the current ones
            if np.sum(scores) > 0.0:
                associations = self.greedy_search(np.copy(scores), overlap_ratio)
                for i in range(associations.shape[0]):
                    for j in range(associations.shape[1]):
                        if associations[i, j] == 1:
                            matches[gt_not_matched[i]] = hypo_not_matched[j]
                            distance += scores[i, j]

            # counters update
            correct_tracks += len(matches)
            false_positives += nb_hypo - len(matches)
            misses += nb_gt - len(matches)
            gt += nb_gt

            # check mismatches between this instant (t) and the last (t-1)
            bad_matches = []
            for match in matches:
                if match in previous_matches:
                    if matches[match] != previous_matches[match]:
                        bad_matches.append(match)
                elif matches[match] in previous_matches.values():
                    bad_matches.append(matches[match])
            for prev_match in previous_matches:
                if prev_match not in matches:
                    if previous_matches[prev_match] not in matches.values():
                        bad_matches.append(previous_matches[prev_match])

            mismatches += len(set(bad_matches))

        # metrics calculation
        motp = None
        mota = None

        if correct_tracks > 0:
            motp = distance / correct_tracks

        if gt > 0:
            mota = 1.0 - (float(misses + false_positives + mismatches) / gt)

        return motp, mota


def main():
    if len(sys.argv) != 4:
        print('Not enough arguments (filename_gt, filename_hypotheses, overlap_ratio)')
        return

    file_annotations = sys.argv[1]
    file_hypotheses = sys.argv[2]
    ratio = float(sys.argv[3])

    annotations_name, annotations_extension = os.path.splitext(file_annotations)
    hypotheses_name, hypotheses_extension = os.path.splitext(file_hypotheses)

    if annotations_extension == str(Extensions.POLYTRACK) and hypotheses_extension == str(Extensions.POLYTRACK):
        data_annotations = PolytrackData(file_annotations)
        data_hypotheses = PolytrackData(file_hypotheses)
    else:
        print('File format not valid (Polytrack (.sqlite) or XML PETS)')
        return

    data_annotations.convert_annotations()
    data_hypotheses.convert_annotations()

    mot_metrics = MOTMetrics(data_annotations, data_hypotheses)
    motp, mota = mot_metrics.compute_metrics(data_annotations.min_frame, data_annotations.max_frame, ratio)

    print('MOTP = %.4f' % motp)
    print('MOTA = %.4f' % mota)


if __name__ == "__main__":
    main()
