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
                rectangle = Rectangle(int(float(x_tl)), int(float(y_tl)), int(float(x_br)), int(float(y_br)))
                custom_bbox = CustomBBox(obj_id, rectangle)
                self.tracks[int(frame_number)].append(custom_bbox)

    def __getitem__(self, index):
        return self.tracks[index]


class Point:
    """
    class to represent a point
    """
    def __init__(self, x, y):
        """
        constructor
        :param x: x coordinate
        :param y: y coordinate
        """
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
        self.center_point = Point(round(x_tl + (self.width() / 2)), round(y_tl + (self.height() / 2)))

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
        computes the area of the intersection between 2 rectangles
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
        computes the distance between the centers of 2 rectangles
        :param rect: other rectangle to do the computation
        :return: distance
        """
        dx = abs(self.center_point.x - rect.center_point.x)
        dy = abs(self.center_point.y - rect.center_point.y)
        return math.sqrt(math.pow(dx, 2) + math.pow(dy, 2))


class CustomBBox:
    """
    class to represent a rectangle with its id
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

    def distance_between_objects(self, cbbox_annot, cbbox_hypo, ratio, IoU):
        """
        computes IoU (intersection over union) or centroids distance between 2 CustomBBox at a given time t
        :param cbbox_annot: CustomBBox from the ground_truth
        :param cbbox_hypo: CustomBBox from hypotheses
        :param ratio: threshold metric
        :param IoU: True if evaluation is done with IoU, False if distance between centroid
        :return : IoU/distance
        """
        if IoU:
            intersection_area = cbbox_annot.rectangle.intersection_area(cbbox_hypo.rectangle)
            union_area = cbbox_annot.rectangle.area() + cbbox_hypo.rectangle.area() - intersection_area
            if intersection_area > 0.0:
                return intersection_area / union_area
            return 0.0
        else:
            dist = cbbox_annot.rectangle.centroids_distance(cbbox_hypo.rectangle)
            if dist < ratio:
                return dist
            else:
                return ratio + 1

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
        :param IoU: True if evaluation is done with IoU, False if distance between centroid
        :param threshold: threshold value for metric (used for padding)
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
                col = np.full(shape=(max_dimension, 1), fill_value=threshold * 100)
            new_mat = np.hstack((new_mat, col))
            nb_cols += 1
        while nb_rows < max_dimension:
            if IoU:
                row = np.zeros(shape=(1, max_dimension))
            else:
                row = np.full(shape=(1, max_dimension), fill_value=threshold * 100)
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
        :param threshold: distance element as an overlap ratio between 2 boxes or distance between centroids
        :param IoU: True if evaluation is done with IoU, False if distance between centroid
        :return motp, mota: multiple object tracking precision/accuracy
        """
        # counters for different metrics to compute MOTP and MOTA
        correct_tracks = 0
        false_positives = 0
        misses = 0
        mismatches = 0
        gt = 0
        distance = 0.0

        # for hungarian algorithm
        munk = Munkres()

        previous_matches = {}

        for t in range(first_instant, last_instant + 1):
            matches = {}
            annotations = self.annotations[t]
            hypotheses = self.hypotheses[t]

            # find consistent matches between time t and t-1
            for pmatch in previous_matches:
                old_annot = pmatch
                old_hypo = previous_matches[pmatch]
                new_annot = MOTMetrics.find_correspondences(old_annot, annotations)
                new_hypo = MOTMetrics.find_correspondences(old_hypo, hypotheses)
                if new_annot is not None and new_hypo is not None:
                    if IoU:
                        if self.distance_between_objects(new_annot, new_hypo, threshold, IoU) >= threshold:
                            matches[new_annot] = new_hypo
                    else:
                        if self.distance_between_objects(new_annot, new_hypo, threshold, IoU) < threshold:
                            matches[new_annot] = new_hypo

            nb_annotations = len(annotations)
            nb_hypotheses = len(hypotheses)

            scores = np.zeros(shape=(nb_annotations, nb_hypotheses))
            i = 0
            if nb_hypotheses > 0:
                for a in annotations:
                    j = 0
                    for h in hypotheses:
                        scores[i, j] = self.distance_between_objects(a, h, threshold, IoU)
                        j += 1
                    i += 1

            # make sure to have square matrix for hungarian algo
            costs = MOTMetrics.make_matrix_square(scores, IoU, threshold)
            # hungarian algo minimizes assignments => take the complement of each value (1 - val)
            if IoU:
                costs = np.ones(costs.shape) - costs

            pairings = munk.compute(costs.copy())
            associations = []
            # only keep valid pairings (i.e. the ones not padding and respecting the threshold)
            for r, c in pairings:
                if r < nb_annotations and c < nb_hypotheses:
                    dist = self.distance_between_objects(annotations[r], hypotheses[c], threshold, IoU)
                    if IoU:
                        if dist >= threshold:
                            associations.append((r, c))
                    else:
                        if dist < threshold:
                            associations.append((r, c))

            good_associations = {}
            for r, c in associations:
                new_annot = annotations[r]
                new_hypo = hypotheses[c]

                invalid_matches = []
                already_exists = False
                for match in matches:
                    if new_annot == match and new_hypo == matches[match]:
                        already_exists = True
                    elif new_annot == match or new_hypo == matches[match]:
                        invalid_matches.append(match)
                        mismatches += 1

                if not already_exists:
                    good_associations[new_annot] = new_hypo

                for m in invalid_matches:
                    del matches[m]

            matches.update(good_associations)

            for match in matches:
                distance += self.distance_between_objects(match, matches[match], threshold, IoU)

            # counters update
            correct_tracks += len(matches)
            false_positives += nb_hypotheses - len(matches)
            misses += nb_annotations - len(matches)
            gt += nb_annotations

            previous_matches = matches.copy()

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
        print('Not enough arguments (filename_gt, filename_hypotheses, method [bboverlap or centroid]), distance')
        return

    file_annotations = sys.argv[1]
    if not os.path.exists(file_annotations):
        print('Annotation file does not exist')
        return

    file_hypotheses = sys.argv[2]
    if not os.path.exists(file_hypotheses):
        print('Hypotheses file does not exist')
        return

    method = None
    if sys.argv[3] == 'bboverlap':
        method = True
    elif sys.argv[3] == 'centroid':
        method = False

    if method is None:
        print('Third argument must either be bboverlap or centroid')
        return

    ratio = float(sys.argv[4])
    if method:
        if ratio < 0.0 or ratio > 1.0:
            print('Distance for bboverlap must be between 0.0 and 1.0')
            return
    else:
        if ratio < 0.0:
            print('Distance for centroid must be positive (x > 0.0)')
            return

    annotations_name, annotations_extension = os.path.splitext(file_annotations)
    hypotheses_name, hypotheses_extension = os.path.splitext(file_hypotheses)

    if annotations_extension == str(Extensions.POLYTRACK) and hypotheses_extension == str(Extensions.POLYTRACK):
        data_annotations = PolytrackData(file_annotations)
        data_hypotheses = PolytrackData(file_hypotheses)
    elif annotations_extension == str(Extensions.XML_PETS) and hypotheses_extension == str(Extensions.XML_PETS):
        data_annotations = XMLPetsData(file_annotations)
        data_hypotheses = XMLPetsData(file_hypotheses)
    elif annotations_extension == str(Extensions.XML_PETS) and hypotheses_extension == str(Extensions.POLYTRACK):
        data_annotations = XMLPetsData(file_annotations)
        data_hypotheses = PolytrackData(file_hypotheses)
    elif annotations_extension == str(Extensions.POLYTRACK) and hypotheses_extension == str(Extensions.XML_PETS):
        data_annotations = PolytrackData(file_annotations)
        data_hypotheses = XMLPetsData(file_hypotheses)
    else:
        print('File format not valid (Polytrack (.sqlite) or XML PETS (.xml))')
        return

    data_annotations.convert_annotations()
    data_hypotheses.convert_annotations()

    mot_metrics = MOTMetrics(data_annotations, data_hypotheses)
    motp, mota = mot_metrics.compute_metrics(data_annotations.min_frame, data_annotations.max_frame, ratio, method)

    if motp is not None:
        print('MOTP = %.4f' % motp)
    else:
        print('MOTP = None')
    if mota is not None:
        print('MOTA = %.4f' % mota)
    else:
        print('MOTA = None')


if __name__ == "__main__":
    main()
