#!/usr/bin/env python3
from __future__ import division
import os
import sys
import numpy as np
from scipy.spatial.transform import Rotation as R
import scipy.signal as signal
from scipy import interpolate
from scipy.spatial.transform import Slerp
from scipy.spatial.transform import Rotation as R
import quaternion as Q
import quaternion.quaternion_time_series as qseries

import rosbag
import rospy
import tf
from std_msgs.msg import Header
from geometry_msgs.msg import TransformStamped, Transform, Vector3, Quaternion

from crowdbot_data.crowdbot_data import CrowdBotDatabase, bag_file_filter, processed_Crowdbot_bag_file_filter

from copy import deepcopy

class BagTfTransformer(object):
    """
    A transformer which transparently uses data recorded from rosbag on the /tf topic
    """

    def __init__(self, bag):
        """
        Create a new BagTfTransformer from an open rosbag or from a file path

        :param bag: an open rosbag or a file path to a rosbag file
        """
        if type(bag) == str:
            bag = rosbag.Bag(bag)
        self.tf_messages = sorted(
            (self._remove_slash_from_frames(tm) for m in bag if m.topic.strip("/") == 'tf' for tm in
             m.message.transforms),
            key=lambda tfm: tfm.header.stamp.to_nsec())
        self.tf_static_messages = sorted(
            (self._remove_slash_from_frames(tm) for m in bag if m.topic.strip("/") == 'tf_static' for tm in
             m.message.transforms),
            key=lambda tfm: tfm.header.stamp.to_nsec())

        self.tf_times = np.array(list((tfm.header.stamp.to_nsec() for tfm in self.tf_messages)))
        self.transformer = tf.TransformerROS()
        self.last_population_range = (rospy.Time(0), rospy.Time(0))
        self.all_frames = None
        self.all_transform_tuples = None
        self.static_transform_tuples = None

    @staticmethod
    def _remove_slash_from_frames(msg):
        msg.header.frame_id = msg.header.frame_id.strip("/")
        msg.child_frame_id = msg.child_frame_id.strip("/")
        return msg

    def getMessagesInTimeRange(self, min_time=None, max_time=None):
        """
        Returns all messages in the time range between two given ROS times

        :param min_time: the lower end of the desired time range (if None, the bag recording start time)
        :param max_time: the upper end of the desired time range (if None, the bag recording end time)
        :return: an iterator over the messages in the time range
        """
        import genpy
        if min_time is None:
            min_time = -float('inf')
        elif type(min_time) in (genpy.rostime.Time, rospy.rostime.Time):
            min_time = min_time.to_nsec()
        if max_time is None:
            max_time = float('inf')
        elif type(max_time) in (genpy.rostime.Time, rospy.rostime.Time):
            max_time = max_time.to_nsec()
        if max_time < min_time:
            raise ValueError('the minimum time should be lesser than the maximum time!')
        indices_in_range = np.where(np.logical_and(min_time < self.tf_times, self.tf_times < max_time))
        ret = (self.tf_messages[i] for i in indices_in_range[0])
        return ret

    def populateTransformerAtTime(self, target_time, buffer_length=10, lookahead=0.1):
        """
        Fills the buffer of the internal tf Transformer with the messages preceeding the given time

        :param target_time: the time at which the Transformer is going to be queried at next
        :param buffer_length: the length of the buffer, in seconds (default: 10, maximum for tf TransformerBuffer)
        """
        target_start_time = target_time - rospy.Duration(
            min(min(buffer_length, 10) - lookahead, target_time.to_sec()))  # max buffer length of tf Transformer
        target_end_time = target_time + rospy.Duration(lookahead)  # lookahead is there for numerical stability
        # otherwise, messages exactly around that time could be discarded
        previous_start_time, previous_end_time = self.last_population_range

        if target_start_time < previous_start_time:
            self.transformer.clear()  # or Transformer would ignore messages as old ones
            population_start_time = target_start_time
        else:
            population_start_time = max(target_start_time, previous_end_time)

        tf_messages_in_interval = self.getMessagesInTimeRange(population_start_time, target_end_time)
        for m in tf_messages_in_interval:
            self.transformer.setTransform(m)
        for st_tfm in self.tf_static_messages:
            st_tfm.header.stamp = target_time
            self.transformer._buffer.set_transform_static(st_tfm, "default_authority")

        self.last_population_range = (target_start_time, target_end_time)

    def getTimeAtPercent(self, percent):
        """
        Returns the ROS time at the given point in the time range

        :param percent: the point in the recorded time range for which the ROS time is desired
        :return:
        """
        start_time, end_time = self.getStartTime(), self.getEndTime()
        time_range = (end_time - start_time).to_sec()
        ret = start_time + rospy.Duration(time_range * float(percent / 100))
        return ret

    def _filterMessages(self, orig_frame=None, dest_frame=None, start_time=None, end_time=None, reverse=False):
        if reverse:
            messages = reversed(self.tf_messages)
        else:
            messages = self.tf_messages

        if orig_frame:
            messages = filter(lambda m: m.header.frame_id == orig_frame, messages)
        if dest_frame:
            messages = filter(lambda m: m.child_frame_id == dest_frame, messages)
        if start_time:
            messages = filter(lambda m: m.header.stamp > start_time, messages)
        if end_time:
            messages = filter(lambda m: m.header.stamp < end_time, messages)
        return messages

    def getTransformMessagesWithFrame(self, frame, start_time=None, end_time=None, reverse=False):
        """
        Returns all transform messages with given frame as source or target frame

        :param frame: the tf frame of interest
        :param start_time: the time at which the messages should start; if None, all recorded messages
        :param end_time: the time at which the messages should end; if None, all recorded messages
        :param reverse: if True, the messages will be provided in reversed order
        :return: an iterator over the messages respecting the criteria
        """
        for m in self._filterMessages(start_time=start_time, end_time=end_time, reverse=reverse):
            if m.header.frame_id == frame or m.child_frame_id == frame:
                yield m

    def getFrameStrings(self):
        """
        Returns the IDs of all tf frames

        :return: a set containing all known tf frame IDs
        """
        if self.all_frames is None:
            ret = set()
            for m in self.tf_messages:
                ret.add(m.header.frame_id)
                ret.add(m.child_frame_id)
            self.all_frames = ret

        return self.all_frames

    def getTransformFrameTuples(self):
        """
        Returns all pairs of directly connected tf frames

        :return: a set containing all known tf frame pairs
        """
        if self.all_transform_tuples is None:
            ret = set()
            for m in self.tf_messages:
                ret.add((m.header.frame_id, m.child_frame_id))
            for m in self.tf_static_messages:
                ret.add((m.header.frame_id, m.child_frame_id))
            self.all_transform_tuples = ret
            self.static_transform_tuples = {(m.header.frame_id, m.child_frame_id) for m in self.tf_static_messages}

        return self.all_transform_tuples

    def getTransformGraphInfo(self, time=None):
        """
        Returns the output of TfTransformer.allFramesAsDot() at a given point in time

        :param time: the ROS time at which tf should be queried; if None, it will be the buffer middle time
        :return: A string containing information about the tf tree
        """
        if time is None:
            time = self.getTimeAtPercent(50)
        self.populateTransformerAtTime(time)
        return self.transformer.allFramesAsDot()

    def getStartTime(self):
        """
        Returns the time of the first tf message in the buffer

        :return: the ROS time of the first tf message in the buffer
        """
        return self.tf_messages[0].header.stamp

    def getEndTime(self):
        """
        Returns the time of the last tf message in the buffer

        :return: the ROS time of the last tf message in the buffer
        """
        return self.tf_messages[-1].header.stamp

    @staticmethod
    def _getTimeFromTransforms(transforms):
        return (t.header.stamp for t in transforms)

    def getAverageUpdateFrequency(self, orig_frame, dest_frame, start_time=None, end_time=None):
        """
        Computes the average time between two tf messages directly connecting two given frames

        :param orig_frame: the source tf frame of the transform of interest
        :param dest_frame: the target tf frame of the transform of interest
        :param start_time: the first time at which the messages should be considered; if None, all recorded messages
        :param end_time: the last time at which the messages should be considered; if None, all recorded messages
        :return: the average transform update frequency
        """
        messages = self._filterMessages(orig_frame=orig_frame, dest_frame=dest_frame,
                                        start_time=start_time, end_time=end_time)
        message_times = BagTfTransformer._getTimeFromTransforms(messages)
        message_times = np.array(message_times)
        average_delta = (message_times[1:] - message_times[:-1]).mean()
        return average_delta

    def getTransformUpdateTimes(self, orig_frame, dest_frame, trigger_orig_frame=None, trigger_dest_frame=None,
                                start_time=None, end_time=None, reverse=False):
        """
        Returns the times at which the transform between two frames was updated.

        If the two frames are not directly connected, two directly connected "trigger frames" must be provided.
        The result will be then the update times of the transform between the two frames, but will start at the
        time when the entire transformation chain is complete.

        :param orig_frame: the source tf frame of the transform of interest
        :param dest_frame: the target tf frame of the transform of interest
        :param trigger_orig_frame: the source tf frame of a transform in the chain, directly connected to trigger_dest_frame
        :param trigger_dest_frame: the target tf frame of a transform in the chain, directly connected to trigger_orig_frame
        :param start_time: the first time at which the messages should be considered; if None, all recorded messages
        :param end_time: the last time at which the messages should be considered; if None, all recorded messages
        :param reverse: if True, the times will be provided in reversed order
        :return: an iterator over the times at which the transform is updated
        """
        trigger_frames_were_provided = trigger_orig_frame is not None or trigger_dest_frame is not None
        if trigger_orig_frame is None:
            trigger_orig_frame = orig_frame
        if trigger_dest_frame is None:
            trigger_dest_frame = dest_frame
        if (trigger_dest_frame, trigger_orig_frame) in self.getTransformFrameTuples():
            trigger_orig_frame, trigger_dest_frame = trigger_dest_frame, trigger_orig_frame
        updates = list(self._filterMessages(orig_frame=trigger_orig_frame, dest_frame=trigger_dest_frame,
                                            start_time=start_time, end_time=end_time, reverse=reverse))
        if not updates:
            if trigger_frames_were_provided:
                raise RuntimeError('the provided trigger frames ({}->{}) must be directly connected!'
                                   .format(trigger_orig_frame, trigger_dest_frame))
            else:
                raise RuntimeError('the two frames ({}->{}) are not directly connected! you must provide \
                 directly connected "trigger frames"'.format(trigger_orig_frame, trigger_dest_frame))
        first_update_time = self.waitForTransform(orig_frame, dest_frame, start_time=start_time)
        return (t for t in BagTfTransformer._getTimeFromTransforms(updates) if t > first_update_time)

    def waitForTransform(self, orig_frame, dest_frame, start_time=None):
        """
        Returns the first time for which at least a tf message is available for the whole chain between \
        the two provided frames

        :param orig_frame: the source tf frame of the transform of interest
        :param dest_frame: the target tf frame of the transform of interest
        :param start_time: the first time at which the messages should be considered; if None, all recorded messages
        :return: the ROS time at which the transform is available
        """
        if orig_frame == dest_frame:
            return self.tf_messages[0].header.stamp
        if start_time is not None:
            messages = filter(lambda m: m.header.stamp > start_time, self.tf_messages)
        else:
            messages = self.tf_messages
        missing_transforms = set(self.getChainTuples(orig_frame, dest_frame)) - self.static_transform_tuples
        message = messages.__iter__()
        ret = rospy.Time(0)
        try:
            while missing_transforms:
                m = next(message)
                if (m.header.frame_id, m.child_frame_id) in missing_transforms:
                    missing_transforms.remove((m.header.frame_id, m.child_frame_id))
                    ret = max(ret, m.header.stamp)
                if (m.child_frame_id, m.header.frame_id) in missing_transforms:
                    missing_transforms.remove((m.child_frame_id, m.header.frame_id))
                    ret = max(ret, m.header.stamp)
        except StopIteration:
            raise ValueError('Transform not found between {} and {}'.format(orig_frame, dest_frame))
        return ret

    def lookupTransform(self, orig_frame, dest_frame, time):
        """
        Returns the transform between the two provided frames at the given time

        :param orig_frame: the source tf frame of the transform of interest
        :param dest_frame: the target tf frame of the transform of interest
        :param time: the first time at which the messages should be considered; if None, all recorded messages
        :return: the ROS time at which the transform is available
        """
        if orig_frame == dest_frame:
            return (0, 0, 0), (0, 0, 0, 1)

        self.populateTransformerAtTime(time)
        try:
            common_time = self.transformer.getLatestCommonTime(orig_frame, dest_frame)
        except:
            raise RuntimeError('Could not find the transformation {} -> {} in the 10 seconds before time {}'
                               .format(orig_frame, dest_frame, time))

        return self.transformer.lookupTransform(orig_frame, dest_frame, common_time)

    def lookupTransformWhenTransformUpdates(self, orig_frame, dest_frame,
                                            trigger_orig_frame=None, trigger_dest_frame=None,
                                            start_time=None, end_time=None):
        """
        Returns the transform between two frames every time it updates

        If the two frames are not directly connected, two directly connected "trigger frames" must be provided.
        The result will be then sampled at the update times of the transform between the two frames.

        :param orig_frame: the source tf frame of the transform of interest
        :param dest_frame: the target tf frame of the transform of interest
        :param trigger_orig_frame: the source tf frame of a transform in the chain, directly connected to trigger_dest_frame
        :param trigger_dest_frame: the target tf frame of a transform in the chain, directly connected to trigger_orig_frame
        :param start_time: the first time at which the messages should be considered; if None, all recorded messages
        :param end_time: the last time at which the messages should be considered; if None, all recorded messages
        :return: an iterator over tuples containing the update time and the transform
        """
        update_times = self.getTransformUpdateTimes(orig_frame, dest_frame,
                                                    trigger_orig_frame=trigger_orig_frame,
                                                    trigger_dest_frame=trigger_dest_frame,
                                                    start_time=start_time, end_time=end_time)
        ret = ((t, self.lookupTransform(orig_frame=orig_frame, dest_frame=dest_frame, time=t)) for t in update_times)
        return ret

    def getFrameAncestors(self, frame, early_stop_frame=None):
        """
        Returns the ancestor frames of the given tf frame, until the tree root

        :param frame: ID of the tf frame of interest
        :param early_stop_frame: if not None, stop when this frame is encountered
        :return: a list representing the succession of frames from the tf tree root to the provided one
        """
        frame_chain = [frame]
        chain_link = list(filter(lambda tt: tt[1] == frame, self.getTransformFrameTuples()))
        while chain_link and frame_chain[-1] != early_stop_frame:
            frame_chain.append(chain_link[0][0])
            chain_link = list(filter(lambda tt: tt[1] == frame_chain[-1], self.getTransformFrameTuples()))
        return list(reversed(frame_chain))

    def getChain(self, orig_frame, dest_frame):
        """
        Returns the chain of frames between two frames

        :param orig_frame: the source tf frame of the transform of interest
        :param dest_frame: the target tf frame of the transform of interest
        :return: a list representing the succession of frames between the two passed as argument
        """
        # transformer.chain is apparently bugged
        orig_ancestors = self.getFrameAncestors(orig_frame, early_stop_frame=dest_frame)
        if orig_ancestors[0] == dest_frame:
            return orig_ancestors
        dest_ancestors = self.getFrameAncestors(dest_frame, early_stop_frame=orig_frame)
        if dest_ancestors[0] == orig_frame:
            return dest_ancestors
        if orig_ancestors[0] == dest_ancestors[-1]:
            return list(reversed(dest_ancestors)) + orig_ancestors[1:]
        if dest_ancestors[0] == orig_ancestors[-1]:
            return list(reversed(orig_ancestors)) + dest_ancestors[1:]
        while len(dest_ancestors) > 0 and orig_ancestors[0] == dest_ancestors[0]:
            if len(orig_ancestors) > 1 and len(dest_ancestors) > 1 and orig_ancestors[1] == dest_ancestors[1]:
                orig_ancestors.pop(0)
            dest_ancestors.pop(0)
        return list(reversed(orig_ancestors)) + dest_ancestors

    def getChainTuples(self, orig_frame, dest_frame):
        """
        Returns the chain of frame pairs representing the transforms connecting two frames

        :param orig_frame: the source tf frame of the transform chain of interest
        :param dest_frame: the target tf frame of the transform chain of interest
        :return: a list of frame ID pairs representing the succession of transforms between the frames passed as argument
        """
        chain = self.getChain(orig_frame, dest_frame)
        return zip(chain[:-1], chain[1:])

    @staticmethod
    def averageTransforms(transforms):
        """
        Computes the average transform over the ones passed as argument

        :param transforms: a list of transforms
        :return: a transform having the average value
        """
        if not transforms:
            raise RuntimeError('requested average of an empty vector of transforms')
        transforms = list(transforms)
        translations = np.array([t[0] for t in transforms])
        quaternions = np.array([t[1] for t in transforms])
        mean_translation = translations.mean(axis=0).tolist()
        mean_quaternion = quaternions.mean(axis=0)  # I know, it is horrible.. but for small rotations shouldn't matter
        mean_quaternion = (mean_quaternion / np.linalg.norm(mean_quaternion)).tolist()
        return mean_translation, mean_quaternion

    def averageTransformOverTime(self, orig_frame, dest_frame, start_time, end_time,
                                 trigger_orig_frame=None, trigger_dest_frame=None):
        """
        Computes the average value of the transform between two frames

        If the two frames are not directly connected, two directly connected "trigger frames" must be provided.
        The result will be then sampled at the update times of the transform between the two frames, but will start at the
        time when the entire transformation chain is complete.

        :param orig_frame: the source tf frame of the transform of interest
        :param dest_frame: the target tf frame of the transform of interest
        :param start_time: the start time of the averaging time range
        :param end_time: the end time of the averaging time range
        :param trigger_orig_frame: the source tf frame of a transform in the chain, directly connected to trigger_dest_frame
        :param trigger_dest_frame: the target tf frame of a transform in the chain, directly connected to trigger_orig_frame
        :return: the average value of the transformation over the specified time range
        """
        if orig_frame == dest_frame:
            return (0, 0, 0), (0, 0, 0, 1)
        update_times = self.getTransformUpdateTimes(orig_frame=orig_frame, dest_frame=dest_frame,
                                                    start_time=start_time, end_time=end_time,
                                                    trigger_orig_frame=trigger_orig_frame,
                                                    trigger_dest_frame=trigger_dest_frame)
        target_transforms = (self.lookupTransform(orig_frame=orig_frame, dest_frame=dest_frame, time=t)
                             for t in update_times)
        return self.averageTransforms(target_transforms)

    def replicateTransformOverTime(self, transf, orig_frame, dest_frame, frequency, start_time=None, end_time=None):
        """
        Adds a new transform to the graph with the specified value

        This can be useful to add calibration a-posteriori.

        :param transf: value of the transform
        :param orig_frame: the source tf frame of the transform of interest
        :param dest_frame: the target tf frame of the transform of interest
        :param frequency: frequency at which the transform should be published
        :param start_time: the time the transform should be published from
        :param end_time: the time the transform should be published until
        :return:
        """
        if start_time is None:
            start_time = self.getStartTime()
        if end_time is None:
            end_time = self.getEndTime()
        transl, quat = transf
        time_delta = rospy.Duration(1 / frequency)

        t_msg = TransformStamped(header=Header(frame_id=orig_frame),
                                 child_frame_id=dest_frame,
                                 transform=Transform(translation=Vector3(*transl), rotation=Quaternion(*quat)))

        def createMsg(time_nsec):
            time = rospy.Time(time_nsec / 1000000000)
            t_msg2 = copy.deepcopy(t_msg)
            t_msg2.header.stamp = time
            return t_msg2

        new_msgs = [createMsg(t) for t in range(start_time.to_nsec(), end_time.to_nsec(), time_delta.to_nsec())]
        self.tf_messages += new_msgs
        self.tf_messages.sort(key=lambda tfm: tfm.header.stamp.to_nsec())
        self.tf_times = np.array(list((tfm.header.stamp.to_nsec() for tfm in self.tf_messages)))
        self.all_transform_tuples.add((orig_frame, dest_frame))

    def processTransform(self, callback, orig_frame, dest_frame,
                         trigger_orig_frame=None, trigger_dest_frame=None, start_time=None, end_time=None):
        """
        Looks up the transform between two frames and forwards it to a callback at each update

        :param callback: a function taking two arguments (the time and the transform as a tuple of translation and rotation)
        :param orig_frame: the source tf frame of the transform of interest
        :param dest_frame: the target tf frame of the transform of interest
        :param start_time: the start time of the time range
        :param end_time: the end time of the time range
        :param trigger_orig_frame: the source tf frame of a transform in the chain, directly connected to trigger_dest_frame
        :param trigger_dest_frame: the target tf frame of a transform in the chain, directly connected to trigger_orig_frame
        :return: an iterator over the result of calling the callback with the looked up transform as argument
        """
        times = self.getTransformUpdateTimes(orig_frame, dest_frame, trigger_orig_frame, trigger_dest_frame,
                                             start_time=start_time, end_time=end_time)
        transforms = [(t, self.lookupTransform(orig_frame=orig_frame, dest_frame=dest_frame, time=t)) for t in times]
        for time, transform in transforms:
            yield callback(time, transform)

    def plotTranslation(self, orig_frame, dest_frame, axis=None,
                        trigger_orig_frame=None, trigger_dest_frame=None, start_time=None, end_time=None,
                        fig=None, ax=None, color='blue'):
        """
        Creates a 2D or 3D plot of the trajectory described by the values of the translation of the transform over time

        :param orig_frame: the source tf frame of the transform of interest
        :param dest_frame: the target tf frame of the transform of interest
        :param axis: if None, the plot will be 3D; otherwise, it should be 'x', 'y', or 'z': the value will be plotted over time
        :param trigger_orig_frame: the source tf frame of a transform in the chain, directly connected to trigger_dest_frame
        :param trigger_dest_frame: the target tf frame of a transform in the chain, directly connected to trigger_orig_frame
        :param start_time: the start time of the time range
        :param end_time: the end time of the time range
        :param fig: if provided, the Matplotlib figure will be reused; otherwise a new one will be created
        :param ax: if provided, the Matplotlib axis will be reused; otherwise a new one will be created
        :param color: the color of the line
        :return:
        """
        import matplotlib.pyplot as plt
        if axis is None:
            # 3D
            from mpl_toolkits.mplot3d import Axes3D
            translation_data = np.array(list(self.processTransform(lambda t, tr: (tr[0]),
                                                                   orig_frame=orig_frame, dest_frame=dest_frame,
                                                                   trigger_orig_frame=trigger_orig_frame,
                                                                   trigger_dest_frame=trigger_dest_frame,
                                                                   start_time=start_time, end_time=end_time)))
            if fig is None:
                fig = plt.figure()
            if ax is None:
                ax = fig.add_subplot(111, projection='3d')

            ax.scatter(
                translation_data[:, 0],
                translation_data[:, 1],
                translation_data[:, 2],
                c=color
            )
            return ax, fig
        else:
            translation_data = np.array(list(self.processTransform(lambda t, tr: (t.to_nsec(), tr[0][axis]),
                                                                   orig_frame=orig_frame, dest_frame=dest_frame,
                                                                   trigger_orig_frame=trigger_orig_frame,
                                                                   trigger_dest_frame=trigger_dest_frame,
                                                                   start_time=start_time, end_time=end_time)))
            if fig is None:
                fig = plt.figure()
            if ax is None:
                ax = fig.add_subplot(111)
            ax.plot(translation_data[:, 0], translation_data[:, 1], color=color)
            return ax, fig

class Transform:
    def __init__(self, rotation=None, translation=None):
        if rotation is None:
            rotation = [0, 0, 0, 1]  # Identity quaternion (x, y, z, w)
        if translation is None:
            translation = [0, 0, 0]  # No translation
        self.rotation = R.from_quat(rotation)
        self.translation = np.array(translation)
    
    @classmethod
    def from_matrix(cls, matrix):
        rotation = R.from_matrix(matrix[:3, :3])
        translation = matrix[:3, 3]
        return cls(rotation.as_quat(), translation)
    
    def to_matrix(self):
        matrix = np.eye(4)
        matrix[:3, :3] = self.rotation.as_matrix()
        matrix[:3, 3] = self.translation
        return matrix

    def __mul__(self, other):
        combined_rotation = self.rotation * other.rotation
        rotated_translation = self.rotation.apply(other.translation)
        combined_translation = self.translation + rotated_translation
        return Transform(combined_rotation.as_quat(), combined_translation)
    
def interp_rotation(source_ts, interp_ts, source_ori):
    """Apply slerp interpolatation to list of rotation variable"""

    slerp = Slerp(source_ts, R.from_quat(source_ori))
    interp_ori = slerp(interp_ts)
    return interp_ori.as_quat()


def interp_translation(source_ts, interp_ts, source_trans):
    """Apply linear interpolatation to list of translation variable"""

    # The length of y along the interpolation axis must be equal to the length of x.
    # source_ts (N,) source_trans (...,N)
    f = interpolate.interp1d(source_ts, np.transpose(source_trans))
    interp_pos = f(interp_ts)
    return np.transpose(interp_pos)


def compute_motion_derivative(motion_stamped_dict, subset=None):
    """Compute derivative for robot state, such as vel, acc, or jerk"""

    ts = motion_stamped_dict.get("timestamp")
    dval_dt_dict = {"timestamp": ts}

    if subset == None:
        subset = motion_stamped_dict.keys()
    for val in subset:
        if val == "timestamp":
            pass
        else:
            dval_dt = np.gradient(motion_stamped_dict.get(val), ts, axis=0)
            dval_dt_dict.update({val: dval_dt})
    return dval_dt_dict


def smooth1d(
    data, filter='savgol', window=9, polyorder=1, check_thres=False, thres=[-1.2, 1.5], mode='interp'
):
    """
    Smooth datapoints with Savitzky-Golay or moving-average

    filter:
        Type of filter to use when smoothing the velocities.
        Default is Savitzky-Golay, which fits a polynomial of order 'polyorder' to the data within each window
    window:
        Smoothing window size in # of frames
    polyorder:
        Order of the polynomial for the Savitzky-Golay filter.
    thres:
        Speed threshold of qolo.
    """

    if check_thres:
        # method1: zero
        # data[data < thres[0]] = 0
        # data[data > thres[1]] = 0
        # method2: assign valid value that not exceeds threshold to the noisy datapoints
        curr_nearest_valid = 0
        for idx in range(1, len(data)):
            if thres[0] < data[idx] < thres[1]:
                curr_nearest_valid = idx
            else:
                data[idx] = data[curr_nearest_valid]

    if filter == 'savgol':
        data_smoothed = signal.savgol_filter(
            data, window_length=window, polyorder=polyorder, mode=mode
        )
    elif filter == 'moving_average':
        ma_window = np.ones(window) / window
        data_smoothed = np.convolve(data, ma_window, mode='same')
    return data_smoothed


def smooth(
    nd_data,
    filter='savgol',
    window=9,
    polyorder=1,
    check_thres=False,
    thres=[-1.2, 1.5],
):
    """
    Smooth multi-dimension datapoints with Savitzky-Golay or moving-average

    filter:
        Type of filter to use when smoothing the velocities.
        Default is Savitzky-Golay, which fits a polynomial of order 'polyorder' to the data within each window
    window:
        Smoothing window size in # of frames
    polyorder:
        Order of the polynomial for the Savitzky-Golay filter.
    thres:
        Speed threshold of qolo.
    """

    nd_data_smoothed = np.zeros_like(nd_data)
    for dim in range(np.shape(nd_data)[1]):
        nd_data_smoothed[:, dim] = smooth1d(
            nd_data[:, dim],
            filter=filter,
            window=window,
            polyorder=polyorder,
            check_thres=check_thres,
            thres=thres,
        )
    return nd_data_smoothed

def extract_and_combine_transforms(bag_path, frames, dynamic_index):
    """
    Extracts and combines transforms from a ROS bag file to form a single 
    transformation from the first frame to the last frame in a specified chain.
    
    Parameters:
    - bag_path (str): The file path to the ROS bag file.
    - frames (list of str): A list of frame names representing the transformation chain.
                            Example: ['odom', 'base_link', 'base_chassis_link'].
                            The function will look for transforms between consecutive frames.
    - dynamic_index (int): The index in the `frames` list indicating which segment of the 
                           transform chain is dynamic (i.e., varying over time). All other 
                           segments are assumed to be static. Example: if `frames` is 
                           ['odom', 'base_link', 'base_chassis_link'] and `dynamic_index` is 1, 
                           then 'odom' to 'base_link' is dynamic and the rest are static.

    Returns:
    - dict: A dictionary containing three keys:
        - 'timestamp': A numpy array of timestamps corresponding to the dynamic transforms.
        - 'position': A numpy array of shape (N, 3) containing the combined position (x, y, z) 
                      for each timestamp.
        - 'orientation': A numpy array of shape (N, 4) containing the combined orientation as 
                         quaternions (x, y, z, w) for each timestamp.
    """
    
    if dynamic_index < 0 or dynamic_index >= len(frames) - 1:
        raise ValueError("Dynamic index is out of range for the frames list.")
    
    # Initialize containers for dynamic and static transforms
    dynamic_transforms = []
    static_transforms = {}
    
    # Prepare placeholders for static transforms based on frames list
    for i in range(len(frames) - 1):
        source_frame = frames[i]
        target_frame = frames[i + 1]
        if i != dynamic_index:
            static_transforms[(source_frame, target_frame)] = None

    # Read rosbag to collect the required transforms
    with rosbag.Bag(bag_path) as bag:
        for topic, msg, t in bag.read_messages(topics=["/tf", "/tf_static"]):
            for transform in msg.transforms:
                src_frame = transform.header.frame_id
                dst_frame = transform.child_frame_id
                
                # Check if the transform matches any of the static segments
                if (src_frame, dst_frame) in static_transforms:
                    if topic == "/tf_static":
                        static_transforms[(src_frame, dst_frame)] = Transform(
                            [transform.transform.rotation.x,
                             transform.transform.rotation.y,
                             transform.transform.rotation.z,
                             transform.transform.rotation.w],
                            [transform.transform.translation.x,
                             transform.transform.translation.y,
                             transform.transform.translation.z]
                        )
                
                # Check if the transform matches the dynamic segment
                if (src_frame == frames[dynamic_index] and dst_frame == frames[dynamic_index + 1]):
                    if topic == "/tf":
                        timestamp = transform.header.stamp.to_sec()
                        dynamic_transforms.append((timestamp, Transform(
                            [transform.transform.rotation.x,
                             transform.transform.rotation.y,
                             transform.transform.rotation.z,
                             transform.transform.rotation.w],
                            [transform.transform.translation.x,
                             transform.transform.translation.y,
                             transform.transform.translation.z]
                        )))

    # Ensure all static transforms are present
    for key, value in static_transforms.items():
        if value is None:
            raise ValueError(f"Static transform {key} not found in the bag.")

    # Ensure there is at least one dynamic transform
    if not dynamic_transforms:
        raise ValueError("No dynamic transforms found for specified frames in the bag.")
    
    # Combine transforms
    combined_timestamps = []
    combined_positions = []
    combined_orientations = []

    for timestamp, dynamic_transform in dynamic_transforms:
        combined_transform = dynamic_transform
        
        # Apply the static transforms before the dynamic one
        for i in range(dynamic_index):
            combined_transform = static_transforms[(frames[i], frames[i + 1])] * combined_transform

        # Apply the static transforms after the dynamic one
        for i in range(dynamic_index + 1, len(frames) - 1):
            combined_transform = combined_transform * static_transforms[(frames[i], frames[i + 1])]
        
        combined_timestamps.append(timestamp)
        combined_positions.append(combined_transform.translation)
        combined_orientations.append(combined_transform.rotation.as_quat())
    
    # Convert lists to numpy arrays
    result = {
        "timestamp": np.array(combined_timestamps),
        "position": np.array(combined_positions),
        "orientation": np.array(combined_orientations)
    }

    return result

#%% Utility function for extraction tf from rosbag and apply interpolation
def extract_pose_from_rosbag(bag_file_path, dataset='JRDB'):
    """Esxtract pose_stamped from rosbag without rosbag play"""

    # load rosbag and BagTfTransformer
    bag = rosbag.Bag(bag_file_path)
    bag_transformer = BagTfTransformer(bag)

    if dataset == 'JRDB':
        trans_iter = bag_transformer.lookupTransformWhenTransformUpdates(
            "odom",
            "base_chassis_link",
            trigger_orig_frame="odom",
            trigger_dest_frame="base_link",)
    elif dataset == 'Crowdbot':
        trans_iter = bag_transformer.lookupTransformWhenTransformUpdates(
        "odom",
        "tf_qolo",)
    else:
        trans_iter = bag_transformer.lookupTransformWhenTransformUpdates(
        "camera_init",
        "body",)
    
    t_list, p_list, o_list = [], [], []
    for timestamp, transformation in trans_iter:
        (position, orientation) = transformation
        # timestamp in genpy.Time type
        t_list.append(timestamp.to_sec())
        p_list.append(position)
        o_list.append(orientation)

    t_np = np.asarray(t_list, dtype=np.float64)
    p_np = np.asarray(p_list, dtype=np.float64)
    o_np = np.asarray(o_list, dtype=np.float64)

    pose_stamped_dict = {"timestamp": t_np, "position": p_np, "orientation": o_np}
    return pose_stamped_dict


# deduplicate tf 
def deduplicate_tf(dataset_tf):
    """Delete duplicate tf in recorded rosbag"""
    ts = dataset_tf.get("timestamp")
    tf_dataset_pos = dataset_tf.get("position")
    tf_dataset_ori = dataset_tf.get("orientation")
    print(
        "Raw input {} frames, about {:.1f} Hz".format(
            len(ts), len(ts) / (max(ts) - min(ts))
        )
    )

    tf_dataset_pos_ = np.vstack(([0.0, 0.0, 0.0], tf_dataset_pos))
    tf_dataset_pos_delta = np.diff(tf_dataset_pos_, axis=0)
    print(tf_dataset_pos)

    dataset_x = np.nonzero(tf_dataset_pos_delta[:, 0])
    dataset_y = np.nonzero(tf_dataset_pos_delta[:, 1])
    dataset_z = np.nonzero(tf_dataset_pos_delta[:, 2])
    tf_dataset_unique_idx = tuple(set(dataset_x[0]) & set(dataset_y[0]) & set(dataset_z[0]))

    new_idx = sorted(tf_dataset_unique_idx)
    

    ts_new = ts[list(new_idx)]
    tf_dataset_pos_new = tf_dataset_pos[new_idx, :]
    tf_dataset_ori_ori = tf_dataset_ori[new_idx, :]
    print(
        "Reduplicated Output {} frames, about {:.1f} Hz".format(
            len(ts_new), len(ts_new) / (max(ts_new) - min(ts_new))
        )
    )
    return {
        "timestamp": ts_new,
        "position": tf_dataset_pos_new,
        "orientation": tf_dataset_ori_ori,
    }


def interp_pose(source_dict, interp_ts):
    """Calculate interpolations for all states with scipy"""
    source_ts = source_dict.get("timestamp")
    source_pos = source_dict.get("position")
    source_ori = source_dict.get("orientation")

    interp_dict = {}
    interp_dict["timestamp"] = deepcopy(interp_ts)

    # method1: saturate the timestamp outside the range
    if np.min(interp_ts) < np.min(source_ts):
        interp_ts[interp_ts < min(source_ts)] = min(source_ts)
    if np.max(interp_ts) > np.max(source_ts):
        interp_ts[interp_ts > max(source_ts)] = max(source_ts)

    # method2: discard timestamps smaller or bigger than source
    # start_idx, end_idx = 0, -1
    # if min(interp_ts) < min(source_ts):
    #     start_idx = np.argmax(interp_ts[interp_ts - source_ts.min() < 0]) + 1
    # if max(interp_ts) > max(source_ts):
    #     end_idx = np.argmax(interp_ts[interp_ts - source_ts.max() <= 0]) + 1
    # interp_ts = interp_ts[start_idx:end_idx]

    # print(interp_ts.min(), interp_ts.max(), source_ts.min(), source_ts.max())

    # Slerp -> interp_rotation -> ValueError: Times must be in strictly increasing order.
    interp_dict["orientation"] = interp_rotation(source_ts, interp_ts, source_ori)
    interp_dict["position"] = interp_translation(source_ts, interp_ts, source_pos)
    return interp_dict, interp_ts


# calculate velocity
def quat_mul(quat0, quat1):
    x0, y0, z0, w0 = quat0
    x1, y1, z1, w1 = quat1
    return np.array(
        [
            w0 * x1 + x0 * w1 + y0 * z1 - z0 * y1,
            w0 * y1 - x0 * z1 + y0 * w1 + z0 * x1,
            w0 * z1 + x0 * y1 - y0 * x1 + z0 * w1,
            w0 * w1 - x0 * x1 - y0 * y1 - z0 * z1,
        ],
        dtype=np.float64,
    )


def quat_norm(quat_):
    quat_sum = np.linalg.norm(quat_)
    for i, val in enumerate(quat_):
        quat_[i] = val / quat_sum
    return quat_


def quat_conjugate(quat_):
    x0, y0, z0, w0 = quat_
    return np.array(
        [-x0, -y0, -z0, w0],
        dtype=np.float64,
    )


def qv_mult(quat_, vec_):
    # vec -> list; quat -> list
    quat_ = quat_norm(quat_)
    temp = quat_mul(quat_, vec_)
    res_vec = quat_mul(temp, quat_conjugate(quat_))
    return res_vec[:3]

class Settings:
    """
    Settings class to store script parameters.

    Attributes:
        dataset (str): Dataset that is being processed
        config_path (str): Path to the configuration file
        folder (str): Different subfolder in rosbag/ dir
        hz (float): Desired interpolated high frequency
        smooth (bool): Filter datapoints with Savitzky-Golay or moving-average filter
        overwrite (bool): Overwrite existing rosbags
    """

    def __init__(self, dataset, config_path, 
                 folder, hz=200.0, smooth=True, overwrite=False):
        self.dataset = dataset
        self.config_path = config_path
        self.folder = folder
        self.hz = hz
        self.smooth = smooth
        self.overwrite = overwrite

if __name__ == '__main__':
    # Instantiate the Settings object with custom or default values
    # folders = ['Cafeteria_1', 'Cafeteria_2', 'Cafeteria_3', 'Cafeteria_5', 'Cafeteria_6', 
    #           'Cafe_street_1-002', 'Cafe_street_2-001', 
    #           'Corridor_1', 'Corridor_10', 
    #           'Hallway_1', 'Hallway_2', 'Hallway_3', 'Hallway_4', 'Hallway_6', 'Hallway_7', 'Hallway_8', 'Hallway_9', 'Hallway_10', 'Hallway_11', 
    #           'Lobby_2', 'Lobby_3', 'Lobby_4', 'Lobby_5', 'Lobby_6', 'Lobby_7', 'Lobby_8', 
    #           'Corridor_2', 'Corridor_3', 'Corridor_5', 'Corridor_7', 'Corridor_8', 'Corridor_9','Corridor_11',  
    #           'Courtyard_1', 'Courtyard_2', 'Courtyard_4', 'Courtyard_5', 'Courtyard_6', 'Courtyard_8', 'Courtyard_9',
    #           'Outdoor_Alley_2', 'Outdoor_Alley_3', 
    #           'Subway_Entrance_2', 'Subway_Entrance_4', 
    #           'Three_way_Intersection_3', 'Three_way_Intersection_4', 'Three_way_Intersection_5', 'Three_way_Intersection_8', 
    #           'Crossroad_1-001',]
    # folders = ['0325_rds_defaced', 
    #           '0325_shared_control_defaced', 
    #           '0327_shared_control_defaced', 
    #           '0410_mds_defaced', 
    #           '0410_rds_defaced', 
    #           '0410_shared_control_defaced', 
    #           '0424_mds_defaced', 
    #           '0424_rds_defaced', 
    #           '0424_shared_control_defaced', 
    #           '1203_manual_defaced', 
    #           '1203_shared_control_defaced']
    folders = ['JRDB_whole',]
    if len(sys.argv) > 1:
        subdir_arg = sys.argv[1]  # Get the single argument
        folders = [subdir_arg]
        
    for folder in folders:
        # args = Settings(dataset='Crowdbot', folder=folder, overwrite=False,
        #                 config_path='./datasets_configs/data_path_Crowdbot.yaml')
        args = Settings(dataset='JRDB', folder=folder, overwrite=True,
                        config_path='./datasets_configs/data_path_JRDB.yaml')

        assert args.dataset in ['JRDB', 'Crowdbot']


        cb_data = CrowdBotDatabase(args.folder, args.config_path)

        rosbag_dir = os.path.join(cb_data.bagbase_dir, args.folder)
        if args.dataset == 'Crowdbot':
            bag_files = list(filter(processed_Crowdbot_bag_file_filter, os.listdir(rosbag_dir)))
        elif args.dataset == 'JRDB':
            bag_files = list(filter(bag_file_filter, os.listdir(rosbag_dir)))
        else:
            bag_files = list(filter(bag_file_filter, os.listdir(rosbag_dir)))

        # destination: pose data in data/xxxx_processed/source_data/tf_JRDB
        if args.dataset == 'JRDB':
            tf_suffix = 'tf_JRDB'
        elif args.dataset == 'Crowdbot':
            tf_suffix = 'tf_qolo'
        else:
            raise RuntimeError

        tf_dataset_dir = os.path.join(cb_data.source_data_dir, tf_suffix)
        if not os.path.exists(tf_dataset_dir):
            os.makedirs(tf_dataset_dir)

        print(
            "Starting extracting pose_stamped files from {} rosbags!".format(len(bag_files))
        )

        counter = 0
        for bf in bag_files:
            bag_path = os.path.join(rosbag_dir, bf)
            seq = bf.split(".")[0]
            counter += 1
            print("({}/{}): {}".format(counter, len(bag_files), bag_path))

            if args.dataset == 'JRDB':
                # sample with lidar frame
                all_stamped_filepath = os.path.join(tf_dataset_dir, seq + "_tfJRDB_raw.npy")
                lidar_stamped_filepath = os.path.join(tf_dataset_dir, seq + "_tfJRDB_sampled.npy")
                # sample at high frequency (200Hz)
                state_filepath = os.path.join(tf_dataset_dir, seq + "_JRDB_state.npy")
            elif args.dataset == 'Crowdbot':
                all_stamped_filepath = os.path.join(tf_dataset_dir, seq + "_tfqolo_raw.npy")
                lidar_stamped_filepath = os.path.join(tf_dataset_dir, seq + "_tfqolo_sampled.npy")
                state_filepath = os.path.join(tf_dataset_dir, seq + "_qolo_state.npy")
            else:
                raise RuntimeError


            if (
                (not os.path.exists(lidar_stamped_filepath))
                or (not os.path.exists(state_filepath))
                or (args.overwrite)
            ):
                if (not os.path.exists(all_stamped_filepath)) or (args.overwrite):
                    
                    # For JRDB offset
                    # _stamped.npy
                    lidar_stamp_dir = os.path.join(cb_data.source_data_dir, "timestamp")
                    stamp_file_path = os.path.join(lidar_stamp_dir, seq + "_stamped.npy")
                    lidar_stamped = np.load(
                        stamp_file_path,
                        allow_pickle=True,
                    ).item()

                    # pose_stamped_dict = extract_pose_from_rosbag(bag_path, dataset=args.dataset)
                    if args.dataset == 'JRDB':
                        frames = ['odom', 'base_link', 'base_chassis_link']
                    elif args.dataset == 'Crowdbot':
                        frames = ['odom', 'tf_qolo']
                    else:
                        frames = ['camera_init', 'body']
                    pose_stamped_dict = extract_and_combine_transforms(bag_path, frames, dynamic_index=0)
                    _, ts_unique = np.unique(pose_stamped_dict.get("timestamp"), return_index=True)
                    ts, pos, orient = pose_stamped_dict.get("timestamp")[ts_unique], pose_stamped_dict.get("position")[ts_unique], pose_stamped_dict.get("orientation")[ts_unique],
                    pose_stamped_dict_ = {
                        "timestamp": ts,
                        "position": pos,
                        "orientation": orient,
                    }

                    if args.dataset == 'JRDB':
                        lidar_ts_offset = lidar_stamped.get("timestamp")[0]
                        timestamps = pose_stamped_dict_['timestamp']
                        differences = np.abs(timestamps - lidar_ts_offset)
                        closest_index = np.argmin(differences)
                        pos_offset = pose_stamped_dict_['position'][closest_index]
                        pose_stamped_dict_['position'] = pose_stamped_dict_['position'] - pos_offset

                    # pose_stamped_dict_ = deduplicate_tf(pose_stamped_dict) # Deleted for JRDB
                    print("Raw input {} frames, about {:.1f} Hz".format(len(pose_stamped_dict_["timestamp"]), len(pose_stamped_dict_["timestamp"]) / (max(pose_stamped_dict_["timestamp"]) - min(pose_stamped_dict_["timestamp"]))))
                    np.save(all_stamped_filepath, pose_stamped_dict_)
                else:
                    print(
                        "Detecting the generated {} already existed!".format(
                            all_stamped_filepath
                        )
                    )
                    print("If you want to overwrite, use flag --overwrite")
                    pose_stamped_dict_ = np.load(
                        all_stamped_filepath, allow_pickle=True
                    ).item()

                # _JRDB_state.npy
                init_ts = pose_stamped_dict_.get("timestamp")
                start_ts, end_ts = init_ts.min(), init_ts.max()
                interp_dt = 1 / args.hz
                high_interp_ts = np.arange(
                    start=start_ts, step=interp_dt, stop=end_ts, dtype=np.float64
                )
                # position & orientation
                state_dict, high_interp_ts = interp_pose(pose_stamped_dict_, high_interp_ts)
                print(
                    "Interpolated output {} frames, about {:.1f} Hz".format(
                        len(high_interp_ts),
                        len(high_interp_ts) / (max(high_interp_ts) - min(high_interp_ts)),
                    )
                )

                # tfdataset {xyz, quat, ts} -> {x, y, z, roll, pitch, yaw, ts}
                # vel {x_vel, y_vel, z_vel, xrot_vel, yrot_vel, zrot_vel, ts}
                from scipy.spatial.transform import Rotation as R

                quat_xyzw = state_dict["orientation"]
                state_pose_r = R.from_quat(quat_xyzw)

                # rotate to local frame
                # TODO: check if needing reduce compared to first frame?
                state_r_aligned = state_pose_r.reduce(
                    left=R.from_quat(state_dict["orientation"][0, :]).inv()
                )
                # rot_mat_list_aligned (frame, 3, 3) robot -> world
                r2w_rot_mat_aligned_list = state_r_aligned.as_matrix()
                # world -> robot
                w2r_rot_mat_aligned_list = state_r_aligned.inv().as_matrix()

                print("Computing linear velocity!")
                position_g = state_dict["position"]

                # smooth with Savitzky-Golay filter
                print("Using Savitzky-Golay filter to smooth position!")
                if args.smooth:
                    smoothed_position_g = smooth(
                        position_g,
                        filter='savgol',
                        window=201,
                        polyorder=2,
                    )
                    position_g = smoothed_position_g

                state_pose_g = {
                    "x": position_g[:, 0],
                    "y": position_g[:, 1],
                    "z": position_g[:, 2],
                    "timestamp": high_interp_ts,
                }
                state_vel_g = compute_motion_derivative(
                    state_pose_g, subset=["x", "y", "z"]
                )

                # https://math.stackexchange.com/a/2030281
                # https://answers.ros.org/question/196149/how-to-rotate-vector-by-quaternion-in-python/
                xyz_vel_g = np.vstack(
                    (state_vel_g["x"], state_vel_g["y"], state_vel_g["z"])
                ).T
                xyz_vel = np.zeros_like(xyz_vel_g)
                for idx in range(xyz_vel_g.shape[0]):
                    vel = xyz_vel_g[idx, :]
                    quat = quat_xyzw[idx, :]
                    vel_ = np.zeros(4, dtype=np.float64)
                    vel_[:3] = vel
                    # xyz_vel[idx, :] = qv_mult(quat, vel_)
                    xyz_vel[idx, :] = qv_mult(quat_conjugate(quat), vel_) # Transform from global to local so by inverse of the quaternion

                print("Computing angular velocity!")

                # calculate difference
                high_interp_ts_delta = np.diff(high_interp_ts)

                # find the index with zero variation
                result = np.where(high_interp_ts_delta == 0)[0]

                if len(result) > 0:
                    print("non-increasing elements number:", len(result))
                    nonzero_idx = np.nonzero(high_interp_ts_delta != 0)
                    start_zidx = np.min(nonzero_idx)
                    end_zidx = np.max(nonzero_idx) + 1

                    # extract the non-increasing point
                    new_high_interp_ts = high_interp_ts[start_zidx : end_zidx + 1]
                    new_quat_xyzw = quat_xyzw[start_zidx : end_zidx + 1, :]

                    quat_wxyz = new_quat_xyzw[:, [3, 0, 1, 2]]
                    quat_wxyz_ = Q.as_quat_array(quat_wxyz)
                    ang_vel = qseries.angular_velocity(quat_wxyz_, new_high_interp_ts)

                    if start_zidx > 0:
                        before = np.zeros((3, start_zidx), dtype=ang_vel.dtype)
                        ang_vel = np.concatenate((before, ang_vel), axis=0)
                    after = len(result) - start_zidx
                    if after > 0:
                        ang_vel = np.pad(ang_vel, ((0, after), (0, 0)), 'edge')
                else:
                    quat_wxyz = quat_xyzw[:, [3, 0, 1, 2]]
                    quat_wxyz_ = Q.as_quat_array(quat_wxyz)
                    ang_vel = qseries.angular_velocity(quat_wxyz_, high_interp_ts)

                assert ang_vel.shape[0] == quat_xyzw.shape[0]
                ## ensure the use of CubicSpline inside `angular_velocity``

                state_vel = {
                    "timestamp": high_interp_ts,
                    "x": xyz_vel[:, 0],
                    "zrot": ang_vel[:, 2],
                }
                if args.smooth:
                    print("Using Savitzky-Golay filter to smooth vel!")

                    # unfiltered data
                    # state_dict.update({"x_vel_uf": xyz_vel[:, 0]})
                    # state_dict.update({"zrot_vel_uf": ang_vel[:, 2]})

                    # apply filter to computed vel
                    smoothed_x_vel = smooth1d(
                        xyz_vel[:, 0],
                        filter='savgol',
                        window=201,
                        polyorder=2,
                        check_thres=True,
                    )
                    smoothed_zrot_vel = smooth1d(
                        ang_vel[:, 2],
                        filter='savgol',
                        window=201,
                        polyorder=2,
                        check_thres=True,
                        thres=[-4.124, 4.124],
                    )

                    # update filtered velocity
                    xyz_vel[:, 0] = smoothed_x_vel
                    ang_vel[:, 2] = smoothed_zrot_vel

                    state_dict.update({"x_vel": smoothed_x_vel})
                    state_dict.update({"zrot_vel": smoothed_zrot_vel})
                else:
                    state_dict.update({"x_vel": xyz_vel[:, 0]})
                    state_dict.update({"zrot_vel": ang_vel[:, 2]})

                # acc
                print("Computing acc!")
                state_acc = compute_motion_derivative(state_vel)
                if args.smooth:
                    print("Using Savitzky-Golay filter to smooth acc!")
                    # 211116: unfiltered data
                    # state_dict.update({"x_acc_uf": state_acc["x"]})
                    # state_dict.update({"zrot_acc_uf": state_acc["zrot"]})
                    # 211116: apply filter to computed acc
                    smoothed_x_acc = smooth1d(
                        state_acc['x'],
                        filter='savgol',
                        window=201,
                        polyorder=2,
                        check_thres=True,
                        thres=[-1.5, 1.5],  # limit value
                    )
                    smoothed_zrot_acc = smooth1d(
                        state_acc['zrot'],
                        filter='savgol',
                        window=201,
                        polyorder=2,
                        check_thres=True,
                        thres=[-4.5, 4.5],  # limit value
                    )

                    # update filtered acceleration
                    state_acc['x'] = smoothed_x_acc
                    state_acc['zrot'] = smoothed_zrot_acc

                    state_dict.update({"x_acc": smoothed_x_acc})
                    state_dict.update({"zrot_acc": smoothed_zrot_acc})
                else:
                    state_dict.update({"x_acc": smoothed_x_acc})
                    state_dict.update({"zrot_acc": smoothed_zrot_acc})

                # jerk
                print("Computing jerk!")
                state_jerk = compute_motion_derivative(state_acc)
                if args.smooth:
                    print("Using Savitzky-Golay filter to smooth jerk!")
                    smoothed_x_jerk = smooth1d(
                        state_jerk['x'],
                        filter='savgol',
                        window=201,
                        polyorder=2,
                        check_thres=True,
                        thres=[-20, 40],  # limit value
                    )
                    smoothed_zrot_jerk = smooth1d(
                        state_jerk['zrot'],
                        filter='savgol',
                        window=201,
                        polyorder=2,
                        check_thres=True,
                        thres=[-40, 40],  # limit value
                    )

                    # update filtered acceleration
                    state_jerk['x'] = smoothed_x_jerk
                    state_jerk['zrot'] = smoothed_zrot_jerk

                state_dict.update({"x_jerk": state_jerk["x"]})
                state_dict.update({"zrot_jerk": state_jerk["zrot"]})
                state_dict.update({"avg_x_jerk": np.average(state_jerk["x"])})
                state_dict.update({"avg_zrot_jerk": np.average(state_jerk["zrot"])})
                np.save(state_filepath, state_dict)

                lidar_ts = lidar_stamped.get("timestamp")
                lidar_stamped_dict, lidar_ts = interp_pose(pose_stamped_dict_, lidar_ts)
                
                # interpolate velocity
                if min(lidar_ts) < min(high_interp_ts):
                    lidar_ts[lidar_ts < min(high_interp_ts)] = min(high_interp_ts)
                if max(lidar_ts) > max(high_interp_ts):
                    lidar_ts[lidar_ts > max(high_interp_ts)] = max(high_interp_ts)
                x_vel_sampled = interp_translation(
                    high_interp_ts, lidar_ts, state_dict["x_vel"]
                )
                zrot_vel_sampled = interp_translation(
                    high_interp_ts, lidar_ts, state_dict["zrot_vel"]
                )
                lidar_stamped_dict.update({"x_vel": x_vel_sampled})
                lidar_stamped_dict.update({"zrot_vel": zrot_vel_sampled})
                np.save(lidar_stamped_filepath, lidar_stamped_dict)

        print("Finish extracting all twist and compute state msg!")