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

#include "cartographer/mapping/map_builder.h"

#include "absl/memory/memory.h"
#include "absl/types/optional.h"
#include "cartographer/common/time.h"
#include "cartographer/io/internal/mapping_state_serialization.h"
#include "cartographer/io/proto_stream.h"
#include "cartographer/io/proto_stream_deserializer.h"
#include "cartographer/io/serialization_format_migration.h"
#include "cartographer/mapping/internal/2d/local_trajectory_builder_2d.h"
#include "cartographer/mapping/internal/2d/pose_graph_2d.h"
#include "cartographer/mapping/internal/3d/local_trajectory_builder_3d.h"
#include "cartographer/mapping/internal/3d/pose_graph_3d.h"
#include "cartographer/mapping/internal/collated_trajectory_builder.h"
#include "cartographer/mapping/internal/global_trajectory_builder.h"
#include "cartographer/mapping/internal/motion_filter.h"
#include "cartographer/sensor/internal/collator.h"
#include "cartographer/sensor/internal/trajectory_collator.h"
#include "cartographer/sensor/internal/voxel_filter.h"
#include "cartographer/transform/rigid_transform.h"
#include "cartographer/transform/transform.h"
#include "google/protobuf/io/zero_copy_stream_impl.h"  // Needed by TextFormat::print()
#include "google/protobuf/text_format.h"  // Needed by TextFormat::PrintToString(mrlist, &str)

// 一个MapBuilder的类对应了一次建图过程，在整个建图过程中，用于全局优化的PoseGraph的对象只有一个，即pose_graph_，而这个变量是在构造函数中就生成了。
// 在AddTrajectorybuilder函数中只需要检查一下pose_graph_是否符合PoseGraph2D或PoseGraph3D的情况
// 一个trajectory对应了机器人运行一圈。在图建好后机器人可能多次运行。每一次运行都是新增一条trajectory，因此，需要动态地维护一个trajectory的列表。
// 每生成一个trajectory时都是调用AddTrajectoryBuilder来创建的。
namespace cartographer {
namespace mapping {
// c++11: 匿名命名空间, 作用域被限制在本文件内
namespace {

using mapping::proto::SerializedData;

// 只返回传感器类型是RANGE的topic的集合
std::vector<std::string> SelectRangeSensorIds(
    const std::set<MapBuilder::SensorId>& expected_sensor_ids) {
  std::vector<std::string> range_sensor_ids;
  for (const MapBuilder::SensorId& sensor_id : expected_sensor_ids) {
    if (sensor_id.type == MapBuilder::SensorId::SensorType::RANGE) {
      range_sensor_ids.push_back(sensor_id.id);
    }
  }
  return range_sensor_ids;
}

// 检查是否是纯定位模式,支持2种纯定位的参数名字,老参数已经弃用,会报警告但程序不会终止
void MaybeAddPureLocalizationTrimmer(
    const int trajectory_id,
    const proto::TrajectoryBuilderOptions& trajectory_options,
    PoseGraph* pose_graph) {
  if (trajectory_options.pure_localization()) {
    LOG(WARNING)
        << "'TrajectoryBuilderOptions::pure_localization' field is deprecated. "
           "Use 'TrajectoryBuilderOptions::pure_localization_trimmer' instead.";
    pose_graph->AddTrimmer(absl::make_unique<PureLocalizationTrimmer>(
        trajectory_id, 3 /* max_submaps_to_keep */));
    return;
  }
  if (trajectory_options.has_pure_localization_trimmer()) {
    pose_graph->AddTrimmer(absl::make_unique<PureLocalizationTrimmer>(
        trajectory_id,
        trajectory_options.pure_localization_trimmer().max_submaps_to_keep()));
  }
}

}  // namespace

// Construct thread_pool_、pose_graph_ 、sensor_collator_
/**
 * @brief 保存配置参数, 根据给定的参数初始化线程池, 并且初始化pose_graph_与sensor_collator_
 * 
 * @param[in] options proto::MapBuilderOptions格式的 map_builder参数
 */
MapBuilder::MapBuilder(const proto::MapBuilderOptions& options)
    : options_(options), thread_pool_(options.num_background_threads()) {

  // 1. Create the pose_graph_ based on the 2d or 3d builder
  CHECK(options.use_trajectory_builder_2d() ^
        options.use_trajectory_builder_3d());
  // 2d位姿图(后端)的初始化
  if (options.use_trajectory_builder_2d()) {
    pose_graph_ = absl::make_unique<PoseGraph2D>(
        options_.pose_graph_options(),
        absl::make_unique<optimization::OptimizationProblem2D>(
            options_.pose_graph_options().optimization_problem_options()),
        &thread_pool_);
  }
  // 3d位姿图(后端)的初始化
  if (options.use_trajectory_builder_3d()) {
    pose_graph_ = absl::make_unique<PoseGraph3D>(
        options_.pose_graph_options(),
        absl::make_unique<optimization::OptimizationProblem3D>(
            options_.pose_graph_options().optimization_problem_options()),
        &thread_pool_);
  }
  // 2. sensor_collator_是一个接口sensor::CollatorInterface的智能指针; sensor::CollatorInterface定义在/sensor/collator_interface.h中
  if (options.collate_by_trajectory()) {
    sensor_collator_ = absl::make_unique<sensor::TrajectoryCollator>(); // sensor::TrajectoryCollator定义在/sensor/internal/collator.h
  } else {
    // sensor_collator_初始化, 实际使用这个
    sensor_collator_ = absl::make_unique<sensor::Collator>(); // sensor::Collator定义在/sensor/internal/collator.h中
  }
}

/**
 * @brief 创建一个新的 TrajectoryBuilder 并返回它的 trajectory_id
 * 
 * @param[in] expected_sensor_ids 所有需要的topic的名字的集合
 * @param[in] trajectory_options 轨迹的参数配置
 * @param[in] local_slam_result_callback 需要传入的回调函数
 *        实际上是map_builder_bridge.cc 中的 OnLocalSlamResult() 函数
 * @return int 新生成的轨迹的id
 */
int MapBuilder::AddTrajectoryBuilder(
    const std::set<SensorId>& expected_sensor_ids,
    const proto::TrajectoryBuilderOptions& trajectory_options,
    LocalSlamResultCallback local_slam_result_callback) {
  // 1. Create a new trajectory_id ; id是从零开始的, 所以新trajectory_id就是trajectory_builders_的size()
  const int trajectory_id = trajectory_builders_.size();

  // 运动过滤器, 运动太小没必要进行更新
  // 配置文件中没有 pose_graph_odometry_motion_filte
  absl::optional<MotionFilter> pose_graph_odometry_motion_filter;
  // LOG(INFO) << "pose_graph odometry_motion_filter is " << trajectory_options.has_pose_graph_odometry_motion_filter();
  // 上面会打印出0, 所以没有使用后端的里程计的motion_filter
  if (trajectory_options.has_pose_graph_odometry_motion_filter()) {
    LOG(INFO) << "Using a motion filter for adding odometry to the pose graph.";
    pose_graph_odometry_motion_filter.emplace(
        MotionFilter(trajectory_options.pose_graph_odometry_motion_filter()));
  }

  // LocalTrajectoryBuilder 就是前端, 不带 Loop Closure 
  // 包含了 Pose Extrapolator, Scan Matching, 生成submap 等

  /*
  * 2. Create a new LocalTrajectoryBuild2D object, local_trajectory_build, and 
  *    push back to the vector of trajectory_builders_
  * 其中CollatedTrajectoryBuilder继承了接口TrajectoryBuilder;而前面生成的local_trajectory_builder
  * 则用于CreateGlobalTrajectoryBuilder2D函数的第一个参数，用于生成一个CollatedTrajectoryBuilder的智能指针
  * CreateGlobalTrajectoryBuilder2D函数定义在/mapping/internal/global_trajectory_builder.h中。
  */
  if (options_.use_trajectory_builder_3d()) {
    // local_trajectory_builder(前端)的初始化
    std::unique_ptr<LocalTrajectoryBuilder3D> local_trajectory_builder;
    if (trajectory_options.has_trajectory_builder_3d_options()) {
      local_trajectory_builder = absl::make_unique<LocalTrajectoryBuilder3D>(
          trajectory_options.trajectory_builder_3d_options(),
          SelectRangeSensorIds(expected_sensor_ids));
    }
    /**
     * c++11: static_cast关键字（编译时类型检查）: static_cast < type-id > ( expression )
     * 该运算符把expression转换为type-id类型, 但没有运行时类型检查来保证转换的安全性
      （1）用于基本数据类型之间的转换, 如把int转换为char, 把int转换成enum, 
      （2）把空指针转换成目标类型的空指针
      （3）把任何类型的表达式类型转换成void类型
      （4）用于类层次结构中父类和子类之间指针和引用的转换.

     * c++11: dynamic_cast关键字（运行时类型检查）: dynamic_cast < type-id > ( expression )
        该运算符把 expression 转换成 type-id 类型的对象. Type-id必须是类的指针、类的引用或者void *
        如果type-id是类指针类型, 那么expression也必须是一个指针
        如果type-id是一个引用, 那么expression也必须是一个引用

        dynamic_cast主要用于类层次间的上行转换（子类到父类）和下行转换（父类到子类）, 还可以用于类之间的交叉转换.
        在类层次间进行上行转换时, dynamic_cast和static_cast的效果是一样的；
        在进行下行转换时, dynamic_cast具有类型检查的功能, 比static_cast更安全.
     */
    DCHECK(dynamic_cast<PoseGraph3D*>(pose_graph_.get()));
    trajectory_builders_.push_back(absl::make_unique<CollatedTrajectoryBuilder>(
        trajectory_options, sensor_collator_.get(), trajectory_id,
        expected_sensor_ids,
        // 将3D前端与3D位姿图打包在一起, 传入CollatedTrajectoryBuilder
        CreateGlobalTrajectoryBuilder3D(
            std::move(local_trajectory_builder), trajectory_id,
            static_cast<PoseGraph3D*>(pose_graph_.get()),
            local_slam_result_callback, pose_graph_odometry_motion_filter)));
  } else {  // 2d的轨迹
    std::unique_ptr<LocalTrajectoryBuilder2D> local_trajectory_builder;
    // 1. 配置项以及传感器配置具例化对象local_trajectory_builder
    if (trajectory_options.has_trajectory_builder_2d_options()) {
      // local_trajectory_builder(前端)的初始化
      local_trajectory_builder = absl::make_unique<LocalTrajectoryBuilder2D>(
          trajectory_options.trajectory_builder_2d_options(),
          SelectRangeSensorIds(expected_sensor_ids));
    }
    // 通过dynamic_cast将pose_graph_对象强制转换为PoseGraph2D，并检查数据类型是否正确
    DCHECK(dynamic_cast<PoseGraph2D*>(pose_graph_.get()));
    // 2. Create a vector of real TrajectoryBuilder including local_trajectory_builder and global pose_graph_
    trajectory_builders_.push_back(absl::make_unique<CollatedTrajectoryBuilder>(
        trajectory_options, sensor_collator_.get(), trajectory_id,
        expected_sensor_ids,
        // 将2D前端与2D位姿图打包在一起, 传入CollatedTrajectoryBuilder
        CreateGlobalTrajectoryBuilder2D(   // local_trajectory_builder or global pose graph?
            std::move(local_trajectory_builder), trajectory_id,
            static_cast<PoseGraph2D*>(pose_graph_.get()),
            local_slam_result_callback, pose_graph_odometry_motion_filter)));
  }
  // 是否是纯定位模式, 如果是则只保存最近3个submap
  MaybeAddPureLocalizationTrimmer(trajectory_id, trajectory_options,
                                  pose_graph_.get());  // 决定是否为pose_graph_对象添加一个OverlappingSubmapsTrimmer2D类型的修饰器， 用于根据子图之间重叠的部分修饰地图

  // 如果该轨迹有初始pose；开始一条轨迹前我们是否已知初始位姿。
  // 这对应的情况就是比如说，我们检测到了一个Landmark。那么这时，我们可以新增加一条trajectory，
  // 增加新的trajectory时设置has.initial_trajectory_pose为真，然后根据机器人与Landmark之间的相对位姿推算机器人相对于世界坐标系的相对位姿。
  // 以该位姿作为新增加的trajectory的初始位姿。这样情况下，在检测到Landmark时就能有效降低累积误差。
  if (trajectory_options.has_initial_trajectory_pose()) {
    const auto& initial_trajectory_pose =
        trajectory_options.initial_trajectory_pose();
    // 在位姿图中设置初始位姿
    pose_graph_->SetInitialTrajectoryPose(
        trajectory_id, initial_trajectory_pose.to_trajectory_id(),
        transform::ToRigid3(initial_trajectory_pose.relative_pose()),
        common::FromUniversal(initial_trajectory_pose.timestamp()));
  }
  // 保存轨迹的配置文件; Add "sensor_id" and "trajectory_options" to the container: all_trajectory_builder_options_
  proto::TrajectoryBuilderOptionsWithSensorIds options_with_sensor_ids_proto;
  for (const auto& sensor_id : expected_sensor_ids) {
    *options_with_sensor_ids_proto.add_sensor_id() = ToProto(sensor_id);
  }
  *options_with_sensor_ids_proto.mutable_trajectory_builder_options() =
      trajectory_options;
  all_trajectory_builder_options_.push_back(options_with_sensor_ids_proto);

  CHECK_EQ(trajectory_builders_.size(), all_trajectory_builder_options_.size());
  return trajectory_id;
}

// 从序列化的数据中构造一条 trajectory
int MapBuilder::AddTrajectoryForDeserialization(
    const proto::TrajectoryBuilderOptionsWithSensorIds&
        options_with_sensor_ids_proto) {
  const int trajectory_id = trajectory_builders_.size();
  //emplace_back和push_back都是向容器内添加数据. 对于在容器中添加类的对象时, 相比于push_back,emplace_back可以避免额外类的复制和移动操作.
  trajectory_builders_.emplace_back();  // emplace an empty object? Just for increase the size()?
  all_trajectory_builder_options_.push_back(options_with_sensor_ids_proto); // Why here use "push_back()", but the last line use emplace_back()? Maybe options_with... is an input value, should be kept.
  // Check the size of both builders and options
  CHECK_EQ(trajectory_builders_.size(), all_trajectory_builder_options_.size());
  return trajectory_id;
}

// 结束指定id的轨迹, 分别进行 传感器数据处理的结束 与 位姿图的结束
void MapBuilder::FinishTrajectory(const int trajectory_id) {
  //结束一条轨迹；通知sensor_collator_和pose_graph_两个对象终止trajectory_id所对应的轨迹处理
  sensor_collator_->FinishTrajectory(trajectory_id);  // local?
  pose_graph_->FinishTrajectory(trajectory_id); // global?
}

// 返回压缩后的地图数据
std::string MapBuilder::SubmapToProto(
    const SubmapId& submap_id, proto::SubmapQuery::Response* const response) {
  // 进行id的检查
  if (submap_id.trajectory_id < 0 ||
      submap_id.trajectory_id >= num_trajectory_builders()) {
    return "Requested submap from trajectory " +
           std::to_string(submap_id.trajectory_id) + " but there are only " +
           std::to_string(num_trajectory_builders()) + " trajectories.";
  }

  // 获取地图数据; Get the submapData and put the submap_data.pose to the "response" 
  const auto submap_data = pose_graph_->GetSubmapData(submap_id); //pose_graph_中应该是维护着一张submap的列表。通过pose_graph_获取指定id的子图
  if (submap_data.submap == nullptr) {
    return "Requested submap " + std::to_string(submap_id.submap_index) +
           " from trajectory " + std::to_string(submap_id.trajectory_id) +
           " but it does not exist: maybe it has been trimmed.";
  }
  // 将压缩后的地图数据放入response
  submap_data.submap->ToResponseProto(submap_data.pose, response);
  return "";
}

// 调用 io::WritePbStream 保存所有数据, 没有使用
void MapBuilder::SerializeState(bool include_unfinished_submaps,
                                io::ProtoStreamWriterInterface* const writer) {
  io::WritePbStream(*pose_graph_, all_trajectory_builder_options_, writer,
                    include_unfinished_submaps);
}

// 将数据进行压缩,并保存到文件中
bool MapBuilder::SerializeStateToFile(bool include_unfinished_submaps,
                                      const std::string& filename) {
  io::ProtoStreamWriter writer(filename);
  io::WritePbStream(*pose_graph_, all_trajectory_builder_options_, &writer,
                    include_unfinished_submaps);
  return (writer.Close());
}

// 从pbstream文件向位姿图添加信息
std::map<int, int> MapBuilder::LoadState(
    io::ProtoStreamReaderInterface* const reader, bool load_frozen_state) {

LOG(WARNING) << "************* LoadState() ****************";  // 1388, 378
  io::ProtoStreamDeserializer deserializer(reader);

  // Create a copy of the pose_graph_proto, such that we can re-write the
  // trajectory ids.
  proto::PoseGraph pose_graph_proto = deserializer.pose_graph();
  const auto& all_builder_options_proto =
      deserializer.all_trajectory_builder_options();

LOG(WARNING) << "Size of trajectory: " << pose_graph_proto.trajectory_size() << "; constraints: " <<
      pose_graph_proto.constraint_size() << "; nodes: " <<
      pose_graph_proto.trajectory()[0].node_size() << "; submap: " <<
      pose_graph_proto.trajectory()[0].submap_size() << "; landmark_pose: "
      << pose_graph_proto.landmark_poses_size() ;
{

    // Write the new address book back to disk in text format
    std::ofstream fw;
    fw.open("/home/asus/Downloads/pose_graph.txt", std::ios::out | std::ios::binary);
    google::protobuf::io::OstreamOutputStream* output = new google::protobuf::io::OstreamOutputStream(&fw);
    google::protobuf::TextFormat::Print(pose_graph_proto, output);
    delete output;
    fw.close();

    // std::string str;
    // google::protobuf::TextFormat::PrintToString(pose_graph_proto, &str);
    // LOG(INFO) << "PrintToString():\n" << str;

    std::string str;
    google::protobuf::TextFormat::PrintToString( pose_graph_proto.constraint(800), &str);
    LOG(INFO) << "pose_graph_proto.constraint(800):\n" << str;
}


// Recover each trajectory containing trajectory_proto, constraint_proto,
// submap_proto, node_proto, landmark
std::map<int, int> trajectory_remapping;
for (int i = 0; i < pose_graph_proto.trajectory_size(); ++i) {
    auto& trajectory_proto = *pose_graph_proto.mutable_trajectory(i);
    const auto& options_with_sensor_ids_proto =
        all_builder_options_proto.options_with_sensor_ids(i);
    // 添加新轨迹
    const int new_trajectory_id =
        AddTrajectoryForDeserialization(options_with_sensor_ids_proto);
    // 原始轨迹id与新生成的轨迹id组成map,放入trajectory_remapping中
    CHECK(trajectory_remapping
              .emplace(trajectory_proto.trajectory_id(), new_trajectory_id)
              .second)
        << "Duplicate trajectory ID: " << trajectory_proto.trajectory_id();

    // 将轨迹id设置为新生成的id
    trajectory_proto.set_trajectory_id(new_trajectory_id);
    if (load_frozen_state) {
      // 将指定轨迹id设置为FROZEN状态
      pose_graph_->FreezeTrajectory(new_trajectory_id);
    }

LOG(WARNING) << "trajectory_proto: " << trajectory_proto.trajectory_id() << "/" << new_trajectory_id << ": " << trajectory_proto.submap_size() << "; " <<
    trajectory_remapping.at(0)<< "/" << trajectory_remapping.size() ; // trajectory_proto: 0/0: 5; 0/1

  }

std::ofstream fw;
fw.open("/home/asus/Downloads/constraints.txt",          std::ios::out | std::ios::binary);
google::protobuf::io::OstreamOutputStream* output = new      google::protobuf::io::OstreamOutputStream(&fw);

  // Apply the calculated remapping to constraints in the pose graph proto.
  // 更新约束中节点与子图的轨迹id
  for (auto& constraint_proto : *pose_graph_proto.mutable_constraint()) {
    constraint_proto.mutable_submap_id()->set_trajectory_id(
        trajectory_remapping.at(constraint_proto.submap_id().trajectory_id()));
    constraint_proto.mutable_node_id()->set_trajectory_id(
        trajectory_remapping.at(constraint_proto.node_id().trajectory_id()));

    google::protobuf::TextFormat::Print(constraint_proto, output);
    }
delete output;
fw.close();


  // 从获取到的位姿图中生成submap_poses
  MapById<SubmapId, transform::Rigid3d> submap_poses;
  for (const proto::Trajectory& trajectory_proto :
       pose_graph_proto.trajectory()) {
    for (const proto::Trajectory::Submap& submap_proto :
         trajectory_proto.submap()) {
      submap_poses.Insert(SubmapId{trajectory_proto.trajectory_id(),
                                   submap_proto.submap_index()},
                          transform::ToRigid3(submap_proto.pose()));

std::string str;
google::protobuf::TextFormat::PrintToString(submap_proto, &str);
LOG(INFO) << "Trajectory: " << trajectory_proto.trajectory_id() << "; submap_proto: " << submap_proto.submap_index() << ":\n" << str;
    }
  }


// 378 nodes recorded in the trajectory ID="0"
// std::ofstream fw;
fw.open("/home/asus/Downloads/node.txt", std::ios::out | std::ios::binary);
google::protobuf::io::OstreamOutputStream* output_node = new google::protobuf::io::OstreamOutputStream(&fw);
// 从获取到的位姿图中生成node_poses
MapById<NodeId, transform::Rigid3d> node_poses;
for (const proto::Trajectory& trajectory_proto :
     pose_graph_proto.trajectory()) {
    for (const proto::Trajectory::Node& node_proto : trajectory_proto.node()) {
      node_poses.Insert(
          NodeId{trajectory_proto.trajectory_id(), node_proto.node_index()},
          transform::ToRigid3(node_proto.pose()));

google::protobuf::TextFormat::Print(node_proto, output_node);

    }
  }
delete output_node;
fw.close();


  // Set global poses of landmarks.
  // 将landmark_poses添加到位姿图中
  for (const auto& landmark : pose_graph_proto.landmark_poses()) {
    pose_graph_->SetLandmarkPose(landmark.landmark_id(),
                                 transform::ToRigid3(landmark.global_pose()),
                                 true);
  }

  if (options_.use_trajectory_builder_3d()) {
    CHECK_NE(deserializer.header().format_version(),
             io::kFormatVersionWithoutSubmapHistograms)
        << "The pbstream file contains submaps without rotational histograms. "
           "This can be converted with the 'pbstream migrate' tool, see the "
           "Cartographer documentation for details. ";
  }

  // The deserializer only contains Submap, Node, TrajectoryData, ImuData, without PoseGraph nor AllTrajectoryBuilderOptions?
  // Extract data from the proto and add to the pose_graph_ ?? 
  // What's difference between proto.xxx and proto.mutable_xxx ?
  // Only imu, odometry, fixedframepose and landmark w/o "scan" ?? scan integrated in submap??
  SerializedData proto;
  int count = 0, count_node = 0;
  while (deserializer.ReadNextSerializedData(&proto)) {
    count++;
    switch (proto.data_case()) {
      case SerializedData::kPoseGraph:
        LOG(ERROR) << "Found multiple serialized `PoseGraph`. Serialized "
                      "stream likely corrupt!.";
        break;
      case SerializedData::kAllTrajectoryBuilderOptions:
        LOG(ERROR) << "Found multiple serialized "
                      "`AllTrajectoryBuilderOptions`. Serialized stream likely "
                      "corrupt!.";
        break;
      case SerializedData::kSubmap: {
        // 为submap设置新的轨迹id
        proto.mutable_submap()->mutable_submap_id()->set_trajectory_id(
            trajectory_remapping.at(
                proto.submap().submap_id().trajectory_id()));
        const SubmapId submap_id(proto.submap().submap_id().trajectory_id(),
                                 proto.submap().submap_id().submap_index());
        // 将submap添加到位姿图中
        pose_graph_->AddSubmapFromProto(submap_poses.at(submap_id),
                                        proto.submap());
        break;
      }
      case SerializedData::kNode: {
        count_node++;
        proto.mutable_node()->mutable_node_id()->set_trajectory_id(
            trajectory_remapping.at(proto.node().node_id().trajectory_id()));
        const NodeId node_id(proto.node().node_id().trajectory_id(),
                             proto.node().node_id().node_index());
        const transform::Rigid3d& node_pose = node_poses.at(node_id);
        // 将node_pose添加到位姿图中
        pose_graph_->AddNodeFromProto(node_pose, proto.node());
        break;
      }
      case SerializedData::kTrajectoryData: {
        proto.mutable_trajectory_data()->set_trajectory_id(
            trajectory_remapping.at(proto.trajectory_data().trajectory_id()));
        // 将TrajectoryData添加到位姿图中
        pose_graph_->SetTrajectoryDataFromProto(proto.trajectory_data());
        break;
      }
      case SerializedData::kImuData: {
        if (load_frozen_state) break;
        // 将IMU数据添加到位姿图中
        pose_graph_->AddImuData(
            trajectory_remapping.at(proto.imu_data().trajectory_id()),
            sensor::FromProto(proto.imu_data().imu_data()));
        break;
      }
      case SerializedData::kOdometryData: {
        if (load_frozen_state) break;
        // 将Odom数据添加到位姿图中
        pose_graph_->AddOdometryData(
            trajectory_remapping.at(proto.odometry_data().trajectory_id()),
            sensor::FromProto(proto.odometry_data().odometry_data()));
        break;
      }
      case SerializedData::kFixedFramePoseData: {
        if (load_frozen_state) break;
        // 将GPS数据添加到位姿图中
        pose_graph_->AddFixedFramePoseData(
            trajectory_remapping.at(
                proto.fixed_frame_pose_data().trajectory_id()),
            sensor::FromProto(
                proto.fixed_frame_pose_data().fixed_frame_pose_data()));
        break;
      }
      case SerializedData::kLandmarkData: {
        if (load_frozen_state) break;
        // 将landmark数据添加到位姿图中
        pose_graph_->AddLandmarkData(
            trajectory_remapping.at(proto.landmark_data().trajectory_id()),
            sensor::FromProto(proto.landmark_data().landmark_data()));
        break;
      }
      default:
        LOG(WARNING) << "Skipping unknown message type in stream: "
                     << proto.GetTypeName();
    }
  }
LOG(WARNING) << "count: " << count << "; node: " << count_node; // 1388, 378

  // 添加子图的附属的节点
  if (load_frozen_state) {
    // Add information about which nodes belong to which submap.
    // This is required, even without constraints.
    for (const proto::PoseGraph::Constraint& constraint_proto :
         pose_graph_proto.constraint()) {
      // 如果子图外约束就跳过, 只向子图添加子图内约束的节点
      if (constraint_proto.tag() !=
          proto::PoseGraph::Constraint::INTRA_SUBMAP) {
        continue;
      }
      // 添加子图的附属的节点
      pose_graph_->AddNodeToSubmap(
          NodeId{constraint_proto.node_id().trajectory_id(),
                 constraint_proto.node_id().node_index()},
          SubmapId{constraint_proto.submap_id().trajectory_id(),
                   constraint_proto.submap_id().submap_index()});
    }
  } else {
    // When loading unfrozen trajectories, 'AddSerializedConstraints' will
    // take care of adding information about which nodes belong to which
    // submap.
    pose_graph_->AddSerializedConstraints(
        FromProto(pose_graph_proto.constraint()));
  }
  CHECK(reader->eof());
  return trajectory_remapping;
}

// 从pbstream文件读取信息
std::map<int, int> MapBuilder::LoadStateFromFile(
    const std::string& state_filename, const bool load_frozen_state) {
  // 检查后缀名
  const std::string suffix = ".pbstream";
  if (state_filename.substr(
          std::max<int>(state_filename.size() - suffix.size(), 0)) != suffix) {
    LOG(WARNING) << "The file containing the state should be a "
                    ".pbstream file.";
  }
  LOG(INFO) << "Loading saved state '" << state_filename << "'...";
LOG(INFO) << "************* Before LoadStateFromFile() ****************";
  io::ProtoStreamReader stream(state_filename);
LOG(INFO) << "************* LoadStateFromFile() ****************";
  return LoadState(&stream, load_frozen_state);
}

std::unique_ptr<MapBuilderInterface> CreateMapBuilder(
    const proto::MapBuilderOptions& options) {
  return absl::make_unique<MapBuilder>(options);
}

}  // namespace mapping
}  // namespace cartographer
