# centroid_tracker.py
from scipy.spatial import distance as dist
from collections import OrderedDict, defaultdict
import numpy as np
import time
from tarsyer_packages.shared_logger import SharedLogger


class CentroidTracker():
    def __init__(self, store, line, entry_point, centroid_point, classes_list=None,
                 counting_axis='x', maxDisappeared=5, maxDistance=50, minDistance=5):

        self.nextObjectID = 1
        self.objects = OrderedDict()
        self.disappeared = OrderedDict()
        self.startTime = OrderedDict()

        # Store class information for each object
        self.objectClasses = OrderedDict()
        self.startCentroid = OrderedDict()
        self.lastCentroid = OrderedDict()
        self.objectCounted = OrderedDict()

        self.maxDisappeared = maxDisappeared
        self.maxDistance = maxDistance
        self.minDistance = minDistance
        self.count = 0

        self.line = line
        self.entry_point = entry_point
        self.centroid_point = centroid_point

        self.classes_list = classes_list if classes_list is not None else []
        self.counting_axis = counting_axis.lower()
        if self.counting_axis not in ['x', 'y']:
            raise ValueError("counting_axis must be 'x' or 'y'")

        self.axis_index = 0 if self.counting_axis == 'x' else 1
        self.entry_area_polarity = self._point_side_of_line(self.line, self.entry_point)

        # Overall counts
        self.positive_direction_count = 0
        self.negative_direction_count = 0
        self.ignore_count = 0
        self.extended_positive_direction_count = 0
        self.extended_negative_direction_count = 0
        self.extended_ignore_count = 0

        # Class-specific counts
        self.class_positive_direction_count = defaultdict(int)
        self.class_negative_direction_count = defaultdict(int)
        self.class_ignore_count = defaultdict(int)
        self.class_extended_positive_direction_count = defaultdict(int)
        self.class_extended_negative_direction_count = defaultdict(int)
        self.class_extended_ignore_count = defaultdict(int)

        # Store original bboxes for saving later
        self.objectBBoxes = OrderedDict()

        self.logger = SharedLogger(store, "centroid_tracker")

        self.logger.info(f'Centroid Started {store}')
        self.logger.info(f'Counting axis: {self.counting_axis.upper()}-axis')
        self.logger.info(f'Classes to track: {self.classes_list}')
        self.logger.info(f'Store is {self.entry_area_polarity} side [True means right and False means left]')
        self.logger.info(
            f'objectID \t startCentroid \t currentCentroid \t axis_dist \t dist. \t Direction \t Status \t Class \t Counted')

    def _point_side_of_line(self, line, point):
        x1, y1 = line[0]
        x2, y2 = line[1]
        x, y = point
        AB_x = x2 - x1
        AB_y = y2 - y1
        AP_x = x - x1
        AP_y = y - y1
        cross_product = (AB_x * AP_y) - (AB_y * AP_x)
        if cross_product < 0:
            return False
        elif cross_product >= 0:
            return True

    def _intersects(self, s0, s1):
        dx0 = s0[1][0] - s0[0][0]
        dx1 = s1[1][0] - s1[0][0]
        dy0 = s0[1][1] - s0[0][1]
        dy1 = s1[1][1] - s1[0][1]
        p0 = dy1 * (s1[1][0] - s0[0][0]) - dx1 * (s1[1][1] - s0[0][1])
        p1 = dy1 * (s1[1][0] - s0[1][0]) - dx1 * (s1[1][1] - s0[1][1])
        p2 = dy0 * (s0[1][0] - s1[0][0]) - dx0 * (s0[1][1] - s1[0][1])
        p3 = dy0 * (s0[1][0] - s1[1][0]) - dx0 * (s0[1][1] - s1[1][1])
        return (p0 * p1 <= 0) & (p2 * p3 <= 0)

    def _extended_intersects(self, line, line2):
        x1, y1 = line[0]
        x2, y2 = line[1]
        prev_pos = line2[0]
        curr_pos = line2[1]

        def position_to_line(px, py):
            det = (px - x1) * (y2 - y1) - (py - y1) * (x2 - x1)
            return det, np.sign(det)

        prev_det, prev_sign = position_to_line(*prev_pos)
        curr_det, curr_sign = position_to_line(*curr_pos)

        if prev_sign != curr_sign:
            return True
        elif prev_sign == curr_sign and curr_det == 0:
            return True
        else:
            return False

    def _check_line_crossing(self, objectID, new_centroid):
        """
        Check if object has crossed the line and update counts immediately
        Returns dict with crossing info if crossed, None otherwise
        """
        if objectID not in self.lastCentroid or self.objectCounted.get(objectID, False):
            return None

        obj_class = self.objectClasses.get(objectID, 'unknown')
        last_pos = self.lastCentroid[objectID]
        current_pos = new_centroid
        movement_line = (last_pos, current_pos)

        crossed = self._intersects(self.line, movement_line)
        extended_crossed = self._extended_intersects(self.line, movement_line)

        if crossed or extended_crossed:
            current_object_polarity = self._point_side_of_line(self.line, current_pos)

            # Determine direction
            if crossed:
                if current_object_polarity == self.entry_area_polarity:
                    self.positive_direction_count += 1
                    direction = 'POSITIVE'
                else:
                    self.negative_direction_count += 1
                    direction = 'NEGATIVE'

                if obj_class in self.classes_list:
                    if current_object_polarity == self.entry_area_polarity:
                        self.class_positive_direction_count[obj_class] += 1
                    else:
                        self.class_negative_direction_count[obj_class] += 1
            else:
                direction = 'IGNORED'
                self.ignore_count += 1
                if obj_class in self.classes_list:
                    self.class_ignore_count[obj_class] += 1

            if extended_crossed:
                if current_object_polarity == self.entry_area_polarity:
                    self.extended_positive_direction_count += 1
                    ext_direction = 'EXT_POSITIVE'
                else:
                    self.extended_negative_direction_count += 1
                    ext_direction = 'EXT_NEGATIVE'

                if obj_class in self.classes_list:
                    if current_object_polarity == self.entry_area_polarity:
                        self.class_extended_positive_direction_count[obj_class] += 1
                    else:
                        self.class_extended_negative_direction_count[obj_class] += 1
            else:
                ext_direction = 'EXT_IGNORED'
                self.extended_ignore_count += 1
                if obj_class in self.classes_list:
                    self.class_extended_ignore_count[obj_class] += 1

            self.objectCounted[objectID] = True

            axis_dist = current_pos[self.axis_index] - self.startCentroid[objectID][self.axis_index]
            dist_ = round(np.linalg.norm(np.array(self.startCentroid[objectID]) - np.array(current_pos)), 2)

            self.logger.info(
                f'{objectID} \t! {self.startCentroid[objectID]} \t! {current_pos} \t! {axis_dist} \t! {dist_} \t! {direction} \t! {ext_direction} \t! {obj_class} \t! COUNTED_NOW')

            # Return crossing info for bbox saving
            return {
                'objectID': objectID,
                'direction': 'pos' if current_object_polarity == self.entry_area_polarity else 'neg',
                'class': obj_class,
                'bbox': self.objectBBoxes.get(objectID)
            }

        return None

    def register(self, centroid, obj_class=None, bbox=None):
        self.objects[self.nextObjectID] = centroid
        self.startCentroid[self.nextObjectID] = centroid
        self.lastCentroid[self.nextObjectID] = centroid
        self.objectClasses[self.nextObjectID] = obj_class
        self.objectCounted[self.nextObjectID] = False
        self.objectBBoxes[self.nextObjectID] = bbox
        self.count += 1
        self.disappeared[self.nextObjectID] = 0
        self.startTime[self.nextObjectID] = int(time.time())
        self.nextObjectID += 1

    def deregister(self, objectID):
        dist_ = round(np.linalg.norm(np.array(self.startCentroid[objectID]) - np.array(self.objects[objectID])), 2)

        if dist_ < self.minDistance:
            self.count -= 1

        obj_class = self.objectClasses.get(objectID, 'unknown')
        was_counted = self.objectCounted.get(objectID, False)

        self.logger.info(f'{objectID} \t! DEREGISTERED \t! {obj_class} \t! Already_counted: {was_counted}')

        del self.objects[objectID]
        del self.disappeared[objectID]
        del self.startTime[objectID]
        del self.startCentroid[objectID]
        del self.lastCentroid[objectID]
        del self.objectClasses[objectID]
        del self.objectCounted[objectID]
        if objectID in self.objectBBoxes:
            del self.objectBBoxes[objectID]

    def update(self, rects, classes=None):
        """
        Update method now accepts classes parameter and returns crossing info
        Returns: (objects, startTime, crossing_events)
        crossing_events is a list of dicts with bbox info for objects that crossed
        """
        crossing_events = []

        if len(rects) == 0:
            try:
                for objectID in list(self.disappeared.keys()):
                    self.disappeared[objectID] += 1
                    if self.disappeared[objectID] > self.maxDisappeared:
                        self.deregister(objectID)
            except Exception as e:
                self.logger.error(f'Exception ******* {e}')

            return self.objects, self.startTime, crossing_events

        if classes is not None and len(classes) != len(rects):
            raise ValueError("Length of classes must match length of rects")

        if classes is None:
            classes = ['unknown'] * len(rects)

        inputCentroids = np.zeros((len(rects), 2), dtype="int")

        for (i, (startX, startY, endX, endY)) in enumerate(rects):
            cX = int((startX + endX) / 2.0)

            if self.centroid_point == 'head':
                cY = int(startY)
            elif self.centroid_point == 'foot':
                cY = int(endY)
            else:
                cY = int((startY + endY) / 2.0)

            inputCentroids[i] = (cX, cY)

        if len(self.objects) == 0:
            for i in range(0, len(inputCentroids)):
                self.register(inputCentroids[i], classes[i], rects[i])
        else:
            objectIDs = list(self.objects.keys())
            objectCentroids = list(self.objects.values())

            D = dist.cdist(np.array(objectCentroids), inputCentroids)
            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]

            usedRows = set()
            usedCols = set()

            for (row, col) in zip(rows, cols):
                if row in usedRows or col in usedCols:
                    continue

                if D[row, col] > self.maxDistance:
                    self.register(inputCentroids[col], classes[col], rects[col])
                    usedRows.add(row)
                    usedCols.add(col)
                    continue

                objectID = objectIDs[row]
                new_centroid = inputCentroids[col]

                # Check for line crossing
                crossing_info = self._check_line_crossing(objectID, new_centroid)
                if crossing_info:
                    crossing_events.append(crossing_info)

                # Update object information
                self.lastCentroid[objectID] = self.objects[objectID]
                self.objects[objectID] = new_centroid

                # Only update class if object hasn't been counted yet
                if not self.objectCounted.get(objectID, False):
                    self.objectClasses[objectID] = classes[col]

                self.objectBBoxes[objectID] = rects[col]
                self.disappeared[objectID] = 0

                usedRows.add(row)
                usedCols.add(col)

            unusedRows = set(range(0, D.shape[0])).difference(usedRows)
            unusedCols = set(range(0, D.shape[1])).difference(usedCols)

            for row in unusedRows:
                objectID = objectIDs[row]
                self.disappeared[objectID] += 1
                if self.disappeared[objectID] > self.maxDisappeared:
                    self.deregister(objectID)

            for col in unusedCols:
                self.register(inputCentroids[col], classes[col], rects[col])

        return self.objects, self.startTime, crossing_events

    def get_class_counts(self):
        """Get counts for each class"""
        class_counts = {}
        for class_name in self.classes_list:
            class_counts[class_name] = {
                'pos': self.class_positive_direction_count[class_name],
                'neg': self.class_negative_direction_count[class_name],
                'ign': self.class_ignore_count[class_name],
                'ext_pos': self.class_extended_positive_direction_count[class_name],
                'ext_neg': self.class_extended_negative_direction_count[class_name],
                'ext_ign': self.class_extended_ignore_count[class_name]
            }
        return {'classes': class_counts}

    def get_current_objects_with_classes(self):
        """Get currently tracked objects with their class information"""
        result = {}
        for obj_id, centroid in self.objects.items():
            result[obj_id] = {
                'centroid': centroid,
                'class': self.objectClasses.get(obj_id, 'unknown'),
                'start_time': self.startTime.get(obj_id),
                'start_centroid': self.startCentroid.get(obj_id),
                'already_counted': self.objectCounted.get(obj_id, False)
            }
        return result

    def reset_counts(self):
        """Reset all counts to zero"""
        self.positive_direction_count = 0
        self.negative_direction_count = 0
        self.ignore_count = 0
        self.extended_positive_direction_count = 0
        self.extended_negative_direction_count = 0
        self.extended_ignore_count = 0
        self.class_positive_direction_count.clear()
        self.class_negative_direction_count.clear()
        self.class_ignore_count.clear()
        self.class_extended_positive_direction_count.clear()
        self.class_extended_negative_direction_count.clear()
        self.class_extended_ignore_count.clear()