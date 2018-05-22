import os
import sys
import sqlite3
import math
import numpy as np
import xml.etree.ElementTree as ET
from collections import defaultdict
from enum import Enum, IntEnum
from munkres import Munkres


class Extensions(Enum):
    """
    enum used to compare extensions of input files
    """
    POLYTRACK = (1, '.sqlite')
    XML_PETS = (2, '.xml')

    def __str__(self):
        return self.value[1]


class Polytrack(IntEnum):
    """
    enum used to index extracted data from Polytrack files
    """
    OBJECT_ID = 0
    FRAME_NUMBER = 1
    X_TOP_LEFT = 2
    Y_TOP_LEFT = 3
    X_BOTTOM_RIGHT = 4
    Y_BOTTOM_RIGHT = 5


class PetsXML(Enum):
    """
    enum used to index extracted data from PETS XML files
    """
    START_FRAME = (0, 'start_frame')
    END_FRAME = (1, 'end_frame')
    OBJECT_ID = (2, 'obj_id')
    FRAME_NO = (3, 'frame_no')
    X = (4, 'x')
    Y = (5, 'y')
    WIDTH = (6, 'width')
    HEIGHT = (7, 'height')

    def __str__(self):
        return self.value[1]


class SQLHandler:
    """
    handles connection and SQL operations to get Polytrack data
    """
    def __init__(self):
        pass

    @staticmethod
    def create_connection(file):
        """
        connects to a database (file)
        :param file: name of the file to connect
        :return connection: sqlite3 object
        """
        connection = sqlite3.connect(file)
        return connection

    @staticmethod
    def select_tracking_info(connection):
        """
        retrieves pertinent information from database file
        :param connection: sqlite3 object connected to file
        :return bboxes: all the bounding boxes from file
        :return min_frame: minimum instant in file
        :return max_frame: maximum instant in file
        """
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
    """
    parent class to more specific data files
    """
    def __init__(self, file):
        """
        constructor
        :param file: name of the file
        """
        self.file = file
        self.min_frame = 0
        self.max_frame = 0
        self.tracks = defaultdict(list)

    def convert_annotations(self):
        pass


class PolytrackData(InputData):
    """
    represents data coming from Polytrack files
    """
    def __init__(self, file):
        """
        constructor
        :param file: name of the file
        """
        super().__init__(self)
        self.file = file

    def convert_annotations(self):
        """
        converts the annotations from Polytrack file into a format to compute CLEAR MOT metrics
        :return: saves data in dict(k, v) where k = instant (frame) and v = list(CustomBBox)
        """
        connection = SQLHandler.create_connection(self.file)
        with connection:
            bboxes, min_frame, max_frame = SQLHandler.select_tracking_info(connection)
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
    """
    represents data coming from the PETS XML files
    """
    def __init__(self, file):
        """
        constructor
        :param file: name of the file
        """
        super().__init__(self)
        self.file = file

    def convert_annotations(self):
        """
        converts annotations from PETS XML file into a format to compute CLEAR MOT metrics
        :return: saves data in dict(k, v) where k = instant (frame) and v = list(CustomBBox)
        """
        tree = ET.parse(self.file)
        root = tree.getroot()
        min_frame = root.attrib[str(PetsXML.START_FRAME)]
        max_frame = root.attrib[str(PetsXML.END_FRAME)]

        # weird thing where end_frame in the XML was saved as ''NUMBER'' (string within a string)
        # these little manipulations make the conversion to integer possible
        char_list = list(max_frame)
        number = ''
        for c in char_list:
            if c.isdecimal():
                number += c
        self.min_frame = int(min_frame)
        self.max_frame = int(number)
        for trajectory in root:
            obj_id = trajectory.attrib[str(PetsXML.OBJECT_ID)]
            for frame in trajectory:
                frame_number = frame.attrib[str(PetsXML.FRAME_NO)]
                x_tl = frame.attrib[str(PetsXML.X)]
                y_tl = frame.attrib[str(PetsXML.Y)]
                x_br = frame.attrib[str(PetsXML.WIDTH)] + x_tl
                y_br = frame.attrib[str(PetsXML.HEIGHT)] + y_tl
                rectangle = Rectangle(int(x_tl), int(y_tl), int(x_br), int(y_br))
                custom_bbox = CustomBBox(obj_id, rectangle)
                self.tracks[int(frame_number)].append(custom_bbox)

    def __getitem__(self, index):
        return self.tracks[index]


class Point:
    """
    class to represent a point
    """
    def __init__(self, x, y):
        self.x = x
        self.y = y


class Rectangle:
    """
    class to represent a rectangle
    """
    def __init__(self, x_tl, y_tl, x_br, y_br):
        """
        constructor
        :param x_tl: x coordinate, top left
        :param y_tl: y coordinate, top left
        :param x_br: x coordinate, bottom right
        :param y_br: y coordinate, bottom right
        """
        self.x_tl = x_tl
        self.y_tl = y_tl
        self.x_br = x_br
        self.y_br = y_br
        self.center_point = Point(round(self.width() / 2), round(self.height() / 2))

    def width(self):
        """
        computes the width of the rectangle
        :return: width
        """
        return self.x_br - self.x_tl

    def height(self):
        """
        computes the height of the rectangle
        :return: height
        """
        return self.y_br - self.y_tl

    def area(self):
        """
        computes the area of a rectangle
        :return: area
        """
        return self.width() * self.height()

    def intersection_area(self, rect):
        """
        computes the are of the intersection between 2 rectangles
        :param rect: other rectangle to do the computation
        :return: area
        """
        dx = min(self.x_br, rect.x_br) - max(self.x_tl, rect.x_tl)
        dy = min(self.y_br, rect.y_br) - max(self.y_tl, rect.y_tl)
        if dx >= 0 and dy >= 0:
            return dx * dy
        else:
            return 0.0

    def centroids_distance(self, rect):
        """

        :param rect:
        :return:
        """
        dx = abs(self.center_point.x - rect.center_point.x)
        dy = abs(self.center_point.y - rect.center_point.y)
        return math.sqrt(math.pow(dx, 2) + math.pow(dy, 2))


class CustomBBox:
    """
    class to represent a rectable with its id
    """
    def __init__(self, index, rectangle):
        """
        constructor
        :param index: object id
        :param rectangle: bounding box
        """
        self.index = index
        self.rectangle = rectangle


class MOTMetrics:
    """
    class to compute MOTP and MOTA
    """
    def __init__(self, annotations, hypotheses):
        """
        constructor
        :param annotations: dict(k,v) of ground truth where k = instant t and v = list(CustomBBox)
        :param hypotheses: dict(k,v) of tracking hypotheses where k = instant t and v = list(CustomBBox)
        """
        self.annotations = annotations
        self.hypotheses = hypotheses

    def distance_between_objects(self, cbbox_annot, cbbox_hypo, time, IoU):
        """
        computes IoU (intersection over union) between 2 CustomBBox at a given time t
        :param cbbox_annot: CustomBBox from the ground_truth
        :param cbbox_hypo: CustomBBox from hypotheses
        :param time: instant t
        :return : IoU
        """
        annotations = self.annotations[time]
        hypotheses = self.hypotheses[time]

        new_annot = MOTMetrics.find_correspondences(cbbox_annot, annotations)
        new_hypo = MOTMetrics.find_correspondences(cbbox_hypo, hypotheses)

        if IoU:
            intersection_area = new_annot.rectangle.intersection_area(new_hypo.rectangle)
            union_area = new_annot.rectangle.area() + new_hypo.rectangle.area() - intersection_area
            if intersection_area > 0.0:
                return intersection_area / union_area
            return 0.0
        else:
            return new_annot.rectangle.centroids_distance(new_hypo.rectangle)


    def object_exists(self, match, time, gt):
        """
        checks the existence of a CustomBBox at a given time t
        :param match: CustomBBox to check
        :param time: instant t
        :param gt: determine if the match is ground_truth or hypotheses
        :return boolean: True of False depending if the match exists
        """
        if gt:
            tracks = self.annotations[time]
        else:
            tracks = self.hypotheses[time]

        for track in tracks:
            if match.index == track.index:
                return True
        return False

    def find_no_match(self, matches, time, gt):
        """
        finds CustomBBoxes that have no matches from annotations or hypotheses
        :param matches: list(CustomBBox) of current matches
        :param time: instant t
        :param gt: determine if the matches are with ground_truth or hypotheses
        :return not_matches: list(CustomBBox) that are not matched
        """
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

    @staticmethod
    def find_correspondences(old_match, tracks):
        """
        finds corresponding CustomBBox between time t and time t-1
        :param old_match: CustomBBox
        :param tracks: list(CustomBBox)
        :return track: CustomBBox or None if no match was found
        """
        for track in tracks:
            if track.index == old_match.index:
                return track
        return None

    @staticmethod
    def make_matrix_square(matrix, IoU, threshold):
        """
        makes a square matrix from a rectangular one, if necessary
        :param matrix: a numpy matrix
        :return new_mat: a numpy square matrix
        """
        nb_rows = matrix.shape[0]
        nb_cols = matrix.shape[1]
        new_mat = matrix.copy()

        max_dimension = max(nb_rows, nb_cols)
        while nb_cols < max_dimension:
            if IoU:
                col = np.zeros(shape=(max_dimension, 1))
            else:
                col = np.full(shape=(max_dimension, 1), fill_value=threshold * 2)
            new_mat = np.hstack((new_mat, col))
            nb_cols += 1
        while nb_rows < max_dimension:
            if IoU:
                row = np.zeros(shape=(1, max_dimension))
            else:
                row = np.full(shape=(1, max_dimension), fill_value=threshold * 2)
            new_mat = np.vstack((new_mat, row))
            nb_rows += 1

        return new_mat

    def compute_metrics(self, first_instant, last_instant, threshold, IoU):
        """
        Reference:
        Keni, Bernardin, and Stiefelhagen Rainer. "Evaluating multiple object tracking performance:
        the CLEAR MOT metrics." EURASIP Journal on Image and Video Processing 2008 (2008)
        computes MOTP and MOTA as defined in the article above
        :param first_instant: time t to start evaluating
        :param last_instant: time t to stop evaluating
        :param overlap_ratio: distance element as an overlap ratio between 2 boxes
        :return motp, mota: multiple object tracking precision/accuracy
        """
        # counters for different metrics to compute MOTP and MOTA
        correct_tracks = 0
        false_positives = 0
        misses = 0
        mismatches = 0
        gt = 0
        distance = 0.0

        # dict(k,v) where k = GT_CustomBBox and v = Hypotheses_CustomBBox
        matches = {}

        # for hungarian algorithm
        munk = Munkres()

        for t in range(first_instant, last_instant + 1):
            previous_matches = matches.copy()

            not_valid_matches = []
            for match in matches:
                # check if current match still exists
                if self.object_exists(match, t, True) and self.object_exists(matches[match], t, False):
                    dist = self.distance_between_objects(match, matches[match], t, IoU)
                    # we work with bounding boxes
                    if IoU:
                        if dist >= threshold:
                            distance += dist
                        else:
                            not_valid_matches.append(match)
                    # we work with distance between centroids
                    else:
                        if dist < threshold:
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

            # compute scores between unmatched gt and hypotheses (m x n matrix,
            # where m = nb of not_matched gt and n = nb of not_matched_hypo)
            scores = np.zeros(shape=(len(gt_not_matched), len(hypo_not_matched)))
            i = 0
            if len(hypo_not_matched) > 0:
                for gtnm in gt_not_matched:
                    j = 0
                    for hyponm in hypo_not_matched:
                        dist = self.distance_between_objects(gtnm, hyponm, t, IoU)
                        scores[i, j] = dist
                        j += 1
                    i += 1

            # make sure to have square matrix for hungarian algo
            costs = MOTMetrics.make_matrix_square(scores, IoU, threshold)
            # hungarian algo minimizes assignments => take the complement of each value (1 - val)
            if IoU:
                costs = np.ones(costs.shape) - costs
            # add new matches to the current ones
            if costs is not None and costs.size > 0:
                associations = munk.compute(costs.copy())
                for r, c in associations:
                    if IoU:
                        if 1.0 - costs[r][c] >= threshold:
                            # ignore the assignments caused by the padding
                            if r < len(gt_not_matched) and c < len(hypo_not_matched):
                                matches[gt_not_matched[r]] = hypo_not_matched[c]
                                distance += scores[r][c]
                    else:
                        if costs[r][c] < threshold:
                            # ignore the assignments caused by the paddig
                            if r < len(gt_not_matched) and c < len(hypo_not_matched):
                                matches[gt_not_matched[r]] = hypo_not_matched[c]
                                distance += scores[r][c]

            # counters update
            correct_tracks += len(matches)
            false_positives += nb_hypo - len(matches)
            misses += nb_gt - len(matches)
            gt += nb_gt

            # check mismatches between this instant (t) and the last (t-1)
            bad_matches = []
            for match in matches:
                if match in previous_matches:
                    # for the same gt, 2 different hypotheses
                    if matches[match] != previous_matches[match]:
                        bad_matches.append(match)
                # for the same hypotheses, 2 different gt
                elif matches[match] in previous_matches.values():
                    bad_matches.append(matches[match])
            for prev_match in previous_matches:
                if prev_match not in matches:
                    # hypothesis still present, but not gt
                    if previous_matches[prev_match] in matches.values():
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
    if len(sys.argv) != 5:
        print('Not enough arguments (filename_gt, filename_hypotheses, method [bboverlap or centroid]), overlap_ratio')
        return

    file_annotations = sys.argv[1]
    file_hypotheses = sys.argv[2]
    method = None
    if sys.argv[3] == 'bboverlap':
        method = True
    elif sys.argv[3] == 'centroid':
        method = False

    if method is None:
        print('Third argument must either be bboverlap or centroid')
        return

    ratio = float(sys.argv[4])

    annotations_name, annotations_extension = os.path.splitext(file_annotations)
    hypotheses_name, hypotheses_extension = os.path.splitext(file_hypotheses)

    if annotations_extension == str(Extensions.POLYTRACK) and hypotheses_extension == str(Extensions.POLYTRACK):
        data_annotations = PolytrackData(file_annotations)
        data_hypotheses = PolytrackData(file_hypotheses)
    elif annotations_extension == str(Extensions.XML_PETS) and hypotheses_extension == str(Extensions.XML_PETS):
        data_annotations = XMLPetsData(file_annotations)
        data_hypotheses = XMLPetsData(file_hypotheses)
    else:
        print('File format not valid (Polytrack (.sqlite) or XML PETS (.xml))')
        return

    data_annotations.convert_annotations()
    data_hypotheses.convert_annotations()

    mot_metrics = MOTMetrics(data_annotations, data_hypotheses)
    motp, mota = mot_metrics.compute_metrics(data_annotations.min_frame, data_annotations.max_frame, ratio, method)

    print('MOTP = %.4f' % motp)
    print('MOTA = %.4f' % mota)


if __name__ == "__main__":
    main()
