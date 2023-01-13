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

#ifndef CARTOGRAPHER_MAPPING_MAP_BUILDER_H_
#define CARTOGRAPHER_MAPPING_MAP_BUILDER_H_

#include <memory>

#include "cartographer/common/thread_pool.h"
#include "cartographer/mapping/map_builder_interface.h"
#include "cartographer/mapping/pose_graph.h"
#include "cartographer/mapping/proto/map_builder_options.pb.h"
#include "cartographer/sensor/collator_interface.h"

// trajectory是机器人跑一圈时的轨迹，在这其中需要记录和维护传感器的数据。根据这个trajectory上传感器收集的数据，我们可以逐步构建出栅格化的地图Submap，但这个submap会随着时间或trajectory的增长而产生误差累积，但trajectory增长到超过一个阈值，则会新增一个submap。而PoseGraph是用来进行全局优化，将所有的Submap紧紧tie在一起，构成一个全局的Map，消除误差累积。
namespace cartographer {
namespace mapping {

// MapBuilder是对MapBuilderInterface的继承和实现，MapBuilder中的方法都已经在MapBuilderInterface中定义
// Wires up the complete SLAM stack with TrajectoryBuilders (for local submaps) and a PoseGraph for loop closure.
// 包含前端(TrajectoryBuilders,scan to submap) 与 后端(用于查找回环的PoseGraph) 的完整的SLAM
class MapBuilder : public MapBuilderInterface {
 public:
  explicit MapBuilder(const proto::MapBuilderOptions &options);
  ~MapBuilder() override {}

  MapBuilder(const MapBuilder &) = delete;
  MapBuilder &operator=(const MapBuilder &) = delete;

  // 用于建立子图的轨迹跟踪器的对象则需要通过调用接口AddTrajectoryBuilder来完成构建
  int AddTrajectoryBuilder(
      const std::set<SensorId> &expected_sensor_ids,
      const proto::TrajectoryBuilderOptions &trajectory_options,
      LocalSlamResultCallback local_slam_result_callback) override;

  int AddTrajectoryForDeserialization(
      const proto::TrajectoryBuilderOptionsWithSensorIds
          &options_with_sensor_ids_proto) override;

  void FinishTrajectory(int trajectory_id) override;

  std::string SubmapToProto(const SubmapId &submap_id,
                            proto::SubmapQuery::Response *response) override;

  void SerializeState(bool include_unfinished_submaps,
                      io::ProtoStreamWriterInterface *writer) override;

  bool SerializeStateToFile(bool include_unfinished_submaps,
                            const std::string &filename) override;

  std::map<int, int> LoadState(io::ProtoStreamReaderInterface *reader,
                               bool load_frozen_state) override;

  std::map<int, int> LoadStateFromFile(const std::string &filename,
                                       const bool load_frozen_state) override;

  mapping::PoseGraphInterface *pose_graph() override {
    return pose_graph_.get();
  }

  int num_trajectory_builders() const override {
    return trajectory_builders_.size();
  }

  // 返回指向CollatedTrajectoryBuilder的指针
  mapping::TrajectoryBuilderInterface *GetTrajectoryBuilder(
      int trajectory_id) const override {
    return trajectory_builders_.at(trajectory_id).get();
  }

  const std::vector<proto::TrajectoryBuilderOptionsWithSensorIds>
      &GetAllTrajectoryBuilderOptions() const override {
    return all_trajectory_builder_options_;
  }

 private:
  const proto::MapBuilderOptions options_;
  // Cartographer使用类ThreadPool对C++11的线程进行了封装，用于方便高效的管理多线程
  common::ThreadPool thread_pool_;  
  // 对于同一个Map，只需要有一个全局的PoseGraph来维护即可,所以，在MapBuilder的成员变量中，定义了一个PoseGraph类型的智能指针：
  std::unique_ptr<PoseGraph> pose_graph_; // Used for loop closure and global optimization

  std::unique_ptr<sensor::CollatorInterface> sensor_collator_;
  // 在系统运行的过程中，可能有不止一条轨迹，针对每一条轨迹Cartographer都建立了一个轨迹跟踪器。
  std::vector<std::unique_ptr<mapping::TrajectoryBuilderInterface>>
      trajectory_builders_;
  std::vector<proto::TrajectoryBuilderOptionsWithSensorIds>
      all_trajectory_builder_options_;
};

std::unique_ptr<MapBuilderInterface> CreateMapBuilder(
    const proto::MapBuilderOptions& options);

}  // namespace mapping
}  // namespace cartographer

#endif  // CARTOGRAPHER_MAPPING_MAP_BUILDER_H_
