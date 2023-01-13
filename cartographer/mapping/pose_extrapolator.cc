/*
 * Copyright 2017 The Cartographer Authors
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

#include "cartographer/mapping/pose_extrapolator.h"

#include <algorithm>

#include "absl/memory/memory.h"
#include "cartographer/transform/transform.h"
#include "glog/logging.h"

namespace cartographer {
namespace mapping {

/**
 * @brief 构造函数
 * 
 * @param[in] pose_queue_duration 时间差 0.001s
 * @param[in] imu_gravity_time_constant 10
 */
PoseExtrapolator::PoseExtrapolator(const common::Duration pose_queue_duration,
                                   double imu_gravity_time_constant)
    : pose_queue_duration_(pose_queue_duration),
      gravity_time_constant_(imu_gravity_time_constant),
      cached_extrapolated_pose_{common::Time::min(),
                                transform::Rigid3d::Identity()} {}

// 使用imu数据进行PoseExtrapolator的初始化
std::unique_ptr<PoseExtrapolator> PoseExtrapolator::InitializeWithImu(
    const common::Duration pose_queue_duration,
    const double imu_gravity_time_constant, const sensor::ImuData& imu_data) {
  auto extrapolator = absl::make_unique<PoseExtrapolator>(
      pose_queue_duration, imu_gravity_time_constant);
  extrapolator->AddImuData(imu_data);
  extrapolator->imu_tracker_ =
      absl::make_unique<ImuTracker>(imu_gravity_time_constant, imu_data.time);
  extrapolator->imu_tracker_->AddImuLinearAccelerationObservation(
      imu_data.linear_acceleration);
  extrapolator->imu_tracker_->AddImuAngularVelocityObservation(
      imu_data.angular_velocity);
  extrapolator->imu_tracker_->Advance(imu_data.time);
  extrapolator->AddPose(
      imu_data.time,
      transform::Rigid3d::Rotation(extrapolator->imu_tracker_->orientation()));
  return extrapolator;
}

// 返回上次校准位姿的时间
common::Time PoseExtrapolator::GetLastPoseTime() const {
  // 如果尚未添加任何位姿, 则返回Time::min()
  if (timed_pose_queue_.empty()) {
    return common::Time::min();
  }
  return timed_pose_queue_.back().time;
}

// 获取上一次预测位姿的时间
common::Time PoseExtrapolator::GetLastExtrapolatedTime() const {
  if (!extrapolation_imu_tracker_) {
    return common::Time::min();
  }
  return extrapolation_imu_tracker_->time();
}

// 将扫描匹配后的pose加入到pose队列中,计算线速度与角速度,并将imu_tracker_的状态更新到time时刻
void PoseExtrapolator::AddPose(const common::Time time,
                               const transform::Rigid3d& pose) {
  // 如果imu_tracker_没有初始化就先进行初始化
  if (imu_tracker_ == nullptr) {
    common::Time tracker_start = time;
    if (!imu_data_.empty()) {// 如果IMU数据队列不为空，则以当前时间和IMU数据中的最早时刻的较小值为初始时刻建立一个ImuTracker
      tracker_start = std::min(tracker_start, imu_data_.front().time);
    }
    // imu_tracker_的初始化
    imu_tracker_ =
        absl::make_unique<ImuTracker>(gravity_time_constant_, tracker_start);
  }
  // 在timed_pose_queue_中保存pose
  timed_pose_queue_.push_back(TimedPose{time, pose});
  //Pose队列大于2，并且时间间隔已经大于我们设定的pose_queue_duration_时，则把队列之前的元素删除
  while (timed_pose_queue_.size() > 2 &&
         timed_pose_queue_[1].time <= time - pose_queue_duration_) {
    timed_pose_queue_.pop_front();
  }
  // 根据加入的pose计算线速度与角速度
  UpdateVelocitiesFromPoses();
  // 将imu_tracker_更新到time时刻
  AdvanceImuTracker(time, imu_tracker_.get());
  // pose队列更新了,之前imu及里程计数据已经过时了
  // 因为pose是匹配的结果,之前的imu及里程计数据是用于预测的,现在结果都有了,之前的用于预测的数据肯定不需要了
  TrimImuData();
  TrimOdometryData();
  // 里程计和融合结果都以当前IMU的tracker为准。
  odometry_imu_tracker_ = absl::make_unique<ImuTracker>(*imu_tracker_);
  // 用于位姿预测时的姿态预测
  extrapolation_imu_tracker_ = absl::make_unique<ImuTracker>(*imu_tracker_);
}

// 把新的IMU数据添加到队列中，删去队列中的过期数据
void PoseExtrapolator::AddImuData(const sensor::ImuData& imu_data) {
  CHECK(timed_pose_queue_.empty() ||
        imu_data.time >= timed_pose_queue_.back().time);
  imu_data_.push_back(imu_data);
  TrimImuData();
}

// 向odom数据队列中添加odom数据,并进行数据队列的修剪,并计算角速度与线速度
void PoseExtrapolator::AddOdometryData(
    const sensor::OdometryData& odometry_data) {
  CHECK(timed_pose_queue_.empty() ||
        odometry_data.time >= timed_pose_queue_.back().time);
  odometry_data_.push_back(odometry_data);
  // 修剪odom的数据队列
  TrimOdometryData();
  // 数据队列中至少有2个数据
  if (odometry_data_.size() < 2) {
    return;
  }
  // TODO(whess): Improve by using more than just the last two odometry poses.
  // Compute extrapolation in the tracking frame.
  // 取最新与最老的两个里程计数据
  const sensor::OdometryData& odometry_data_oldest = odometry_data_.front();  // Why not just compute the last two odometry poses
  const sensor::OdometryData& odometry_data_newest = odometry_data_.back();
  // 最新与最老odom数据间的时间差
  const double odometry_time_delta =
      common::ToSeconds(odometry_data_oldest.time - odometry_data_newest.time);
  // 计算两个位姿间的坐标变换
  const transform::Rigid3d odometry_pose_delta =
      odometry_data_newest.pose.inverse() * odometry_data_oldest.pose;
  // 两个位姿间的旋转量除以时间得到 tracking frame 的角速度
  angular_velocity_from_odometry_ =
      transform::RotationQuaternionToAngleAxisVector(
          odometry_pose_delta.rotation()) /
      odometry_time_delta;
  if (timed_pose_queue_.empty()) {
    return;
  }
  // 平移量除以时间得到 tracking frame 的线速度, 只在x方向有数值
  const Eigen::Vector3d
      linear_velocity_in_tracking_frame_at_newest_odometry_time =
          odometry_pose_delta.translation() / odometry_time_delta;
  // 根据位姿队列中最后一个位姿 乘以 上次添加位姿时的姿态预测到time时刻的姿态变化量
  // 得到预测的 最新里程计数据时刻 tracking frame 在 local 坐标系下的姿态
  const Eigen::Quaterniond orientation_at_newest_odometry_time =
      timed_pose_queue_.back().pose.rotation() *
      ExtrapolateRotation(odometry_data_newest.time,
                          odometry_imu_tracker_.get());
  // 将tracking frame的线速度进行旋转, 得到 local 坐标系下 tracking frame 的线速度
  linear_velocity_from_odometry_ =
      orientation_at_newest_odometry_time *
      linear_velocity_in_tracking_frame_at_newest_odometry_time;
}

// 预测得到time时刻 tracking frame 在 local 坐标系下的位姿
transform::Rigid3d PoseExtrapolator::ExtrapolatePose(const common::Time time) {
  const TimedPose& newest_timed_pose = timed_pose_queue_.back();
  CHECK_GE(time, newest_timed_pose.time);
  // 如果本次预测时间与上次计算时间相同 就不再重复计算
  if (cached_extrapolated_pose_.time != time) {
    // 预测tracking frame在local坐标系下time时刻的位置
    const Eigen::Vector3d translation =
        ExtrapolateTranslation(time) + newest_timed_pose.pose.translation();
    // 预测tracking frame在local坐标系下time时刻的姿态
    const Eigen::Quaterniond rotation =
        newest_timed_pose.pose.rotation() *
        ExtrapolateRotation(time, extrapolation_imu_tracker_.get());
    cached_extrapolated_pose_ =
        TimedPose{time, transform::Rigid3d{translation, rotation}};
  }
  return cached_extrapolated_pose_.pose;
}

// 预测得到time时刻 tracking frame 在 local 坐标系下的姿态
Eigen::Quaterniond PoseExtrapolator::EstimateGravityOrientation(
    const common::Time time) {
  ImuTracker imu_tracker = *imu_tracker_;
  // 使得 imu_tracker 预测到time时刻
  AdvanceImuTracker(time, &imu_tracker);
  // 返回 imu_tracker 预测到的time时刻 的姿态
  return imu_tracker.orientation();
}

// 根据pose队列计算tracking frame 在 local坐标系下的线速度与角速度
void PoseExtrapolator::UpdateVelocitiesFromPoses() {
  if (timed_pose_queue_.size() < 2) {
    // We need two poses to estimate velocities.
    return;
  }
  CHECK(!timed_pose_queue_.empty());
  // 取出队列最末尾的一个Pose,也就是最新时间点的Pose,并记录相应的时间
  const TimedPose& newest_timed_pose = timed_pose_queue_.back();
  const auto newest_time = newest_timed_pose.time;
  // 取出队列最开头的一个 Pose, 也就是最旧时间点的 Pose,并记录相应的时间
  const TimedPose& oldest_timed_pose = timed_pose_queue_.front();
  const auto oldest_time = oldest_timed_pose.time;
  // 计算两者的时间差
  const double queue_delta = common::ToSeconds(newest_time - oldest_time);
  // 如果时间差小于pose_queue_duration_(1ms), 不进行计算
  if (queue_delta < common::ToSeconds(pose_queue_duration_)) {
    LOG(WARNING) << "Queue too short for velocity estimation. Queue duration: "
                 << queue_delta << " s";
    return;
  }
  // 取出timed_pose_queue_这个队列中最早和最新的两个Pose做差，然后除以时间得到机器人的速度。
  const transform::Rigid3d& newest_pose = newest_timed_pose.pose;
  const transform::Rigid3d& oldest_pose = oldest_timed_pose.pose;
  // 平移量除以时间得到 tracking frame 在 local坐标系下的线速度
  linear_velocity_from_poses_ =
      (newest_pose.translation() - oldest_pose.translation()) / queue_delta;
  // 角度变化量除以时间得到角速度得到 tracking frame 在 local坐标系下的角速度
  angular_velocity_from_poses_ =
      transform::RotationQuaternionToAngleAxisVector(
          oldest_pose.rotation().inverse() * newest_pose.rotation()) /
      queue_delta;
}

// 修剪imu的数据队列,丢掉过时的imu数据
void PoseExtrapolator::TrimImuData() {
  // 删去队列中无用的IMU数据, 需要满足三个条件：IMU数据队列大于1，Pose的队列不为空，IMU数据队列的第一个元素时间小于Pose队列的最后一个元素的时间
  // 最后一个条件意味着当IMU数据的时间比一个最新的Pose的时间要早时，说明这个IMU数据已经过期了。所以从队列中删掉就可以了。
  // 知道IMU数据的时间要比最新的Pose时间晚，那么说明这时候这个数据还有用。这种情况就不再删了，跳出循环，等待其他程序取出队列最开头的IMU数据进行融合
  while (imu_data_.size() > 1 && !timed_pose_queue_.empty() &&
         imu_data_[1].time <= timed_pose_queue_.back().time) {
    imu_data_.pop_front();
  }
}

// 修剪odom的数据队列,丢掉过时的odom数据
void PoseExtrapolator::TrimOdometryData() {
  // 保持odom队列中第二个数据的时间要大于最后一个位姿的时间, odometry_data_最少是2个
  while (odometry_data_.size() > 2 && !timed_pose_queue_.empty() &&
         odometry_data_[1].time <= timed_pose_queue_.back().time) {
    odometry_data_.pop_front();
  }
}

// ExtrapolatePose: 使用码盘计算出的线速度进行位置估计，使用IMU计算出的角度增量进行姿态估计
// AdvanceImuTracker
//    使用IMU的陀螺仪和加速度计进行姿态估计，估计方法是比较简单的互补滤波算法。用此函数进行的姿态估计共有三个，分别是 imu_tracker_、odometry_imu_tracker_和extrapolation_imu_tracker_
//    imu_tracker_
//      直接使用IMU数据进行，只进行一次初始化
//    odometry_imu_tracker_
//      表示AddPose到AddOdometryData之间（即 最新优化的位姿时刻 到 最新码盘数据时刻 之间的时间段）的姿态变化
//    extrapolation_imu_tracker_
//      表示AddPose到ExtrapolatePose之间（即 最新优化的位姿时刻 到 最新估计位姿时刻 之间的时间段）的姿态变化
/**
 * @brief 更新imu_tracker的状态, 并将imu_tracker的状态预测到time时刻
 * 
 * @param[in] time 要预测到的时刻
 * @param[in] imu_tracker 给定的先验状态
 */
void PoseExtrapolator::AdvanceImuTracker(const common::Time time,
                                         ImuTracker* const imu_tracker) const {
  // 检查指定时间是否大于等于 ImuTracker 的时间
  CHECK_GE(time, imu_tracker->time());
  // 不使用imu 或者 预测时间之前没有imu数据的情况
  if (imu_data_.empty() || time < imu_data_.front().time) {
    // There is no IMU data until 'time', so we advance the ImuTracker and use
    // the angular velocities from poses and fake gravity to help 2D stability.
    // 在time之前没有IMU数据, 因此我们推进ImuTracker, 并使用姿势和假重力产生的角速度来帮助2D稳定
    
    // 预测当前时刻的姿态与重力方向
    imu_tracker->Advance(time);
    // 使用 假的重力数据对加速度的测量进行更新
    imu_tracker->AddImuLinearAccelerationObservation(Eigen::Vector3d::UnitZ());
    // 只能依靠其他方式得到的角速度进行测量值的更新
    imu_tracker->AddImuAngularVelocityObservation(
        odometry_data_.size() < 2 ? angular_velocity_from_poses_
                                  : angular_velocity_from_odometry_);
    return;
  }
  // imu_tracker的时间比imu数据队列中第一个数据的时间早, 就先预测到imu数据队列中第一个数据的时间
  if (imu_tracker->time() < imu_data_.front().time) {
    // Advance to the beginning of 'imu_data_'. // 先把ImuTracker更新到IMU数据来临的那一刻
    imu_tracker->Advance(imu_data_.front().time);
  }

  // c++11: std::lower_bound() 是在区间内找到第一个大于等于 value 的值的位置并返回, 如果没找到就返回 end() 位置
  // 在第四个参数位置可以自定义比较规则,在区域内查找第一个 **不符合** comp 规则的元素

  // 在imu数据队列中找到第一个时间上 大于等于 imu_tracker->time() 的数据的索引
  //然后依次取出IMU数据队列中的数据，更新ImuTracker，直到IMU数据的时间比指定时间time要晚。
  auto it = std::lower_bound(
      imu_data_.begin(), imu_data_.end(), imu_tracker->time(),
      [](const sensor::ImuData& imu_data, const common::Time& time) {
        return imu_data.time < time;
      });
  // 然后依次对imu数据进行预测, 以及添加观测, 直到imu_data_的时间大于等于time截止
  while (it != imu_data_.end() && it->time < time) {
    // 预测出当前时刻的姿态与重力方向
    imu_tracker->Advance(it->time);
    // 根据线速度的观测,更新重力的方向,并根据重力的方向对上一时刻预测的姿态进行校准
    imu_tracker->AddImuLinearAccelerationObservation(it->linear_acceleration);
    // 更新角速度观测
    imu_tracker->AddImuAngularVelocityObservation(it->angular_velocity);
    ++it;
  }
  // 最后将imu_tracker的状态预测到time时刻
  imu_tracker->Advance(time);
}

// 计算从imu_tracker到time时刻的姿态变化量
Eigen::Quaterniond PoseExtrapolator::ExtrapolateRotation(
    const common::Time time, ImuTracker* const imu_tracker) const {
  CHECK_GE(time, imu_tracker->time());
  // 更新imu_tracker的状态到time时刻
  AdvanceImuTracker(time, imu_tracker);
  // 通过imu_tracker_获取上一次位姿校准时的姿态
  const Eigen::Quaterniond last_orientation = imu_tracker_->orientation();
  // 求取上一帧到当前时刻预测出的姿态变化量：上一帧姿态四元数的逆 乘以 当前时刻预测出来的姿态四元数
  return last_orientation.inverse() * imu_tracker->orientation();
}

// 返回从最后一个位姿的时间 到time时刻 的tracking frame在local坐标系下的平移量
Eigen::Vector3d PoseExtrapolator::ExtrapolateTranslation(common::Time time) {
  const TimedPose& newest_timed_pose = timed_pose_queue_.back();
  const double extrapolation_delta =
      common::ToSeconds(time - newest_timed_pose.time);
  // 如果有里程计数据，则更信任里程计速度，直接把从里程计处获得的线速度乘以时间
  if (odometry_data_.size() < 2) {
    return extrapolation_delta * linear_velocity_from_poses_;
  }
  // 没有里程计数据的话则把从Pose队列中估计的线速度乘以时间
  return extrapolation_delta * linear_velocity_from_odometry_;
}

// 获取一段时间内的预测位姿的结果
PoseExtrapolator::ExtrapolationResult
PoseExtrapolator::ExtrapolatePosesWithGravity(
    const std::vector<common::Time>& times) {
  std::vector<transform::Rigid3f> poses;
  // c++11: std::prev 获取一个距离指定迭代器 n 个元素的迭代器,而不改变输入迭代器的值
  // 默认 n 为1,当 n 为正数时, 其返回的迭代器将位于 it 左侧；
  // 反之, 当 n 为负数时, 其返回的迭代器位于 it 右侧

  // 获取 [0, n-1] 范围的预测位姿
  for (auto it = times.begin(); it != std::prev(times.end()); ++it) {
    poses.push_back(ExtrapolatePose(*it).cast<float>());
  }

  // 进行当前线速度的预测
  const Eigen::Vector3d current_velocity = odometry_data_.size() < 2
                                               ? linear_velocity_from_poses_
                                               : linear_velocity_from_odometry_;
  return ExtrapolationResult{poses, ExtrapolatePose(times.back()),
                             current_velocity,
                             EstimateGravityOrientation(times.back())};
}

}  // namespace mapping
}  // namespace cartographer
