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

#ifndef CARTOGRAPHER_MAPPING_2D_SUBMAP_2D_H_
#define CARTOGRAPHER_MAPPING_2D_SUBMAP_2D_H_

#include <memory>
#include <vector>

#include "Eigen/Core"
#include "cartographer/common/lua_parameter_dictionary.h"
#include "cartographer/mapping/2d/grid_2d.h"
#include "cartographer/mapping/2d/map_limits.h"
#include "cartographer/mapping/proto/serialization.pb.h"
#include "cartographer/mapping/proto/submap_visualization.pb.h"
#include "cartographer/mapping/proto/submaps_options_2d.pb.h"
#include "cartographer/mapping/range_data_inserter_interface.h"
#include "cartographer/mapping/submaps.h"
#include "cartographer/mapping/trajectory_node.h"
#include "cartographer/mapping/value_conversion_tables.h"
#include "cartographer/sensor/range_data.h"
#include "cartographer/transform/rigid_transform.h"

namespace cartographer {
namespace mapping {

proto::SubmapsOptions2D CreateSubmapsOptions2D(
    common::LuaParameterDictionary* parameter_dictionary);

class Submap2D : public Submap {
 public:
  // 第一个参数是原点坐标，第二个参数是一个Grid2D变量，存储栅格化坐标和坐标上的概率值 Grid2D继承了GridInterface(/mapping/grid_interface.h)，
  // Grid2D又被ProbabilityGrid继承，定义在/mapping/2d/probability_grid.h中, 基本数据都存储在Grid2D的成员变量grid_中。
  Submap2D(const Eigen::Vector2f& origin, std::unique_ptr<Grid2D> grid,
           ValueConversionTables* conversion_tables);
  explicit Submap2D(const proto::Submap2D& proto,
                    ValueConversionTables* conversion_tables);

  // For implement the virtual functions in TrajectoryBuildInterface
  proto::Submap ToProto(bool include_grid_data) const override;
  void UpdateFromProto(const proto::Submap& proto) override;

  void ToResponseProto(const transform::Rigid3d& global_submap_pose,
                       proto::SubmapQuery::Response* response) const override;

  const Grid2D* grid() const { return grid_.get(); }

  // Insert 'range_data' into this submap using 'range_data_inserter'. The
  // submap must not be finished yet.
  void InsertRangeData(const sensor::RangeData& range_data,
                       const RangeDataInserterInterface* range_data_inserter);
  void Finish();

 private:
  std::unique_ptr<Grid2D> grid_;
  ValueConversionTables* conversion_tables_;
};

// 在cartographer中总是同时存在着两个Submap：Old Submap和New Submap. Old Submap是用来做matching, New submap则用来matching next. 每一帧RangeData数据都要同时插入到两个submap中。
// 当插入Old Submap中的传感器帧数达到一定数量（在配置文件/src/cartographer/configuration_files/trajectory_builder_2d.lua中设置-submap/num_range_data）时，
// old submap就不再改变，这时Old Submap开始进行Loop Closure，被加入到submap的list中，设置matching_submap_index增加1，然后被ActiveSubmap这个object所以往，
// 而原先的new submap则变成新的Old Submap，同时通过AddSubmap函数重新初始化一个submap。

//除了刚开始构建该对象的时候，只有一个子图(Submap2D)，其他时候它都维护着两个子图对象。 一个子图用于进行扫描匹配，称为旧图。另一个子图被称为新图用于插入扫描数据。当新图中插入一定数量的数据完成了初始化操作之后，它就会被当作旧图，用于扫描匹配

// The first active submap will be created on the insertion of the first range
// data. Except during this initialization when no or only one single submap
// exists, there are always two submaps into which range data is inserted: an
// old submap that is used for matching, and a new one, which will be used for
// matching next, that is being initialized.
//
// Once a certain number of range data have been inserted, the new submap is
// considered initialized: the old submap is no longer changed, the "new" submap
// is now the "old" submap and is used for scan-to-map matching. Moreover, a
// "new" submap gets created. The "old" submap is forgotten by this object.
class ActiveSubmaps2D {
 public:
  explicit ActiveSubmaps2D(const proto::SubmapsOptions2D& options);

  ActiveSubmaps2D(const ActiveSubmaps2D&) = delete;
  ActiveSubmaps2D& operator=(const ActiveSubmaps2D&) = delete;

  // Inserts 'range_data' into the Submap collection.
  std::vector<std::shared_ptr<const Submap2D>> InsertRangeData(
      const sensor::RangeData& range_data);

  // LocalTrajectoryBuilder2D::ScanMatch uses this function
  std::vector<std::shared_ptr<const Submap2D>> submaps() const;

 private:
  std::unique_ptr<RangeDataInserterInterface> CreateRangeDataInserter();
  std::unique_ptr<GridInterface> CreateGrid(const Eigen::Vector2f& origin);
  void FinishSubmap();
  void AddSubmap(const Eigen::Vector2f& origin);

  const proto::SubmapsOptions2D options_;
  std::vector<std::shared_ptr<Submap2D>> submaps_;
  std::unique_ptr<RangeDataInserterInterface> range_data_inserter_;
  ValueConversionTables conversion_tables_;
};

}  // namespace mapping
}  // namespace cartographer

#endif  // CARTOGRAPHER_MAPPING_2D_SUBMAP_2D_H_
