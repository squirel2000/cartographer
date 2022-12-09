/*
 * Copyright 2016 The Cartographer Authors
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "cartographer/mapping/trajectory_builder_interface.h"

#include "cartographer/mapping/internal/2d/local_trajectory_builder_options_2d.h"
#include "cartographer/mapping/internal/3d/local_trajectory_builder_options_3d.h"
#include "cartographer/mapping/internal/local_slam_result_data.h"

// TrajectoryBuilder是用来创建一个trajectory的。所以，该类首先要保存trajectory上的传感器数据，从一个连续的多帧传感器数据中会抽取若干关键帧, 一帧关键帧数据被称为trajectory上的一个节点(注意与ROS中的节点概念相区分)，
// 一条trajectory由一串儿节点组成，所以TrajectoryBuilder要维护一个节点列表，同时，每一帧时的传感器数据相对于该Submap的局部坐标变换要已知；
// 其次，该类要创建并维护一个栅格化的Submap列表，以便在MapBuilder中对所有的submap进行优化，形成一个总的Map。
// 每个Submap相对于世界坐标系的位姿要估计出来，这样PoseGraph才能依据不同的submap各自的位姿变换矩阵把他们tie在一起并做全局优化。
namespace cartographer {
namespace mapping {
namespace {

void PopulatePureLocalizationTrimmerOptions(
    proto::TrajectoryBuilderOptions* const trajectory_builder_options,
    common::LuaParameterDictionary* const parameter_dictionary) {
  constexpr char kDictionaryKey[] = "pure_localization_trimmer";
  if (!parameter_dictionary->HasKey(kDictionaryKey)) return;

  auto options_dictionary = parameter_dictionary->GetDictionary(kDictionaryKey);
  auto* options =
      trajectory_builder_options->mutable_pure_localization_trimmer();
  options->set_max_submaps_to_keep(
      options_dictionary->GetInt("max_submaps_to_keep"));
}

void PopulatePoseGraphOdometryMotionFilterOptions(
    proto::TrajectoryBuilderOptions* const trajectory_builder_options,
    common::LuaParameterDictionary* const parameter_dictionary) {
  constexpr char kDictionaryKey[] = "pose_graph_odometry_motion_filter";
  if (!parameter_dictionary->HasKey(kDictionaryKey)) return;

  auto options_dictionary = parameter_dictionary->GetDictionary(kDictionaryKey);
  auto* options =
      trajectory_builder_options->mutable_pose_graph_odometry_motion_filter();
  options->set_max_time_seconds(
      options_dictionary->GetDouble("max_time_seconds"));
  options->set_max_distance_meters(
      options_dictionary->GetDouble("max_distance_meters"));
  options->set_max_angle_radians(
      options_dictionary->GetDouble("max_angle_radians"));
}

}  // namespace

proto::TrajectoryBuilderOptions CreateTrajectoryBuilderOptions(
    common::LuaParameterDictionary* const parameter_dictionary) {
  proto::TrajectoryBuilderOptions options;
  *options.mutable_trajectory_builder_2d_options() =
      CreateLocalTrajectoryBuilderOptions2D(
          parameter_dictionary->GetDictionary("trajectory_builder_2d").get());
  *options.mutable_trajectory_builder_3d_options() =
      CreateLocalTrajectoryBuilderOptions3D(
          parameter_dictionary->GetDictionary("trajectory_builder_3d").get());
  options.set_collate_fixed_frame(
      parameter_dictionary->GetBool("collate_fixed_frame"));
  options.set_collate_landmarks(
      parameter_dictionary->GetBool("collate_landmarks"));
  PopulatePureLocalizationTrimmerOptions(&options, parameter_dictionary);
  PopulatePoseGraphOdometryMotionFilterOptions(&options, parameter_dictionary);
  return options;
}

proto::SensorId ToProto(const TrajectoryBuilderInterface::SensorId& sensor_id) {
  proto::SensorId sensor_id_proto;
  switch (sensor_id.type) {
    case TrajectoryBuilderInterface::SensorId::SensorType::RANGE:
      sensor_id_proto.set_type(proto::SensorId::RANGE);
      break;
    case TrajectoryBuilderInterface::SensorId::SensorType::IMU:
      sensor_id_proto.set_type(proto::SensorId::IMU);
      break;
    case TrajectoryBuilderInterface::SensorId::SensorType::ODOMETRY:
      sensor_id_proto.set_type(proto::SensorId::ODOMETRY);
      break;
    case TrajectoryBuilderInterface::SensorId::SensorType::FIXED_FRAME_POSE:
      sensor_id_proto.set_type(proto::SensorId::FIXED_FRAME_POSE);
      break;
    case TrajectoryBuilderInterface::SensorId::SensorType::LANDMARK:
      sensor_id_proto.set_type(proto::SensorId::LANDMARK);
      break;
    case TrajectoryBuilderInterface::SensorId::SensorType::LOCAL_SLAM_RESULT:
      sensor_id_proto.set_type(proto::SensorId::LOCAL_SLAM_RESULT);
      break;
    default:
      LOG(FATAL) << "Unsupported sensor type.";
  }
  sensor_id_proto.set_id(sensor_id.id);
  return sensor_id_proto;
}

TrajectoryBuilderInterface::SensorId FromProto(
    const proto::SensorId& sensor_id_proto) {
  TrajectoryBuilderInterface::SensorId sensor_id;
  switch (sensor_id_proto.type()) {
    case proto::SensorId::RANGE:
      sensor_id.type = TrajectoryBuilderInterface::SensorId::SensorType::RANGE;
      break;
    case proto::SensorId::IMU:
      sensor_id.type = TrajectoryBuilderInterface::SensorId::SensorType::IMU;
      break;
    case proto::SensorId::ODOMETRY:
      sensor_id.type =
          TrajectoryBuilderInterface::SensorId::SensorType::ODOMETRY;
      break;
    case proto::SensorId::FIXED_FRAME_POSE:
      sensor_id.type =
          TrajectoryBuilderInterface::SensorId::SensorType::FIXED_FRAME_POSE;
      break;
    case proto::SensorId::LANDMARK:
      sensor_id.type =
          TrajectoryBuilderInterface::SensorId::SensorType::LANDMARK;
      break;
    case proto::SensorId::LOCAL_SLAM_RESULT:
      sensor_id.type =
          TrajectoryBuilderInterface::SensorId::SensorType::LOCAL_SLAM_RESULT;
      break;
    default:
      LOG(FATAL) << "Unsupported sensor type.";
  }
  sensor_id.id = sensor_id_proto.id();
  return sensor_id;
}

}  // namespace mapping
}  // namespace cartographer
