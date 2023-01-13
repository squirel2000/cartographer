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
#include "cartographer/mapping/2d/probability_grid.h"

#include <limits>

#include "absl/memory/memory.h"
#include "cartographer/mapping/probability_values.h"
#include "cartographer/mapping/submaps.h"

namespace cartographer {
namespace mapping {

/**
 * @brief ProbabilityGrid的构造函数
 * 
 * @param[in] limits 地图坐标信息
 * @param[in] conversion_tables 转换表
 */
ProbabilityGrid::ProbabilityGrid(const MapLimits& limits,
                                 ValueConversionTables* conversion_tables)
    : Grid2D(limits, kMinCorrespondenceCost, kMaxCorrespondenceCost,
             conversion_tables),
      conversion_tables_(conversion_tables) {}

ProbabilityGrid::ProbabilityGrid(const proto::Grid2D& proto,
                                 ValueConversionTables* conversion_tables)
    : Grid2D(proto, conversion_tables), conversion_tables_(conversion_tables) {
  CHECK(proto.has_probability_grid_2d());
}

// Sets the probability of the cell at 'cell_index' to the given
// 'probability'. Only allowed if the cell was unknown before.
// 将 索引 处单元格的概率设置为给定的概率, 仅当单元格之前处于未知状态时才允许
void ProbabilityGrid::SetProbability(const Eigen::Array2i& cell_index,
                                     const float probability) {
  // 获取对应栅格的引用
  uint16& cell =
      (*mutable_correspondence_cost_cells())[ToFlatIndex(cell_index)];
  CHECK_EQ(cell, kUnknownProbabilityValue);
  // 为栅格赋值 value
  cell =
      CorrespondenceCostToValue(ProbabilityToCorrespondenceCost(probability));
  // 更新bounding_box
  mutable_known_cells_box()->extend(cell_index.matrix());
}

// 根据传感器返回的数据是Hit还是Miss这两种情况，我们会对所有的 [公式] 计算出两张表，hit_table和miss_table. 这样，假设在Grid2D这个栅格图中的其中一个pixel或cell，已预先有一个value值，然后现在又有一个传感器的测量结果，hit或miss，那么我们不需要计算，只需要以value为索引，通过查hit_table或miss_table中的值就可以得到更新后的value应该是多少。所以这个函数中第一个参数就是一个点的坐标，第二个参数就是一张table。
// ApplyLookupTable是用来通过查表来更新栅格单元的占用概率的。 这也就是为什么Cartographer会费那么多精力把浮点数转换为uint16
// Applies the 'odds' specified when calling ComputeLookupTableToApplyOdds() to the probability of the cell at 'cell_index',
// if the cell has not already been updated. Multiple updates of the same cell will be ignored until FinishUpdate() is called. 
// Returns true if the cell was updated.
// 如果单元格尚未更新,则将调用 ComputeLookupTableToApplyOdds() 时指定的 'odds' 应用于单元格在 'cell_index' 处的概率
// 在调用 FinishUpdate() 之前，将忽略同一单元格的多次更新。如果单元格已更新，则返回 true
//
// If this is the first call to ApplyOdds() for the specified cell, its value
// will be set to probability corresponding to 'odds'.
// 如果这是对指定单元格第一次调用 ApplyOdds(),则其值将设置为与 'odds' 对应的概率

// 使用查找表对指定栅格进行栅格值的更新
bool ProbabilityGrid::ApplyLookupTable(const Eigen::Array2i& cell_index,
                                       const std::vector<uint16>& table) {
  DCHECK_EQ(table.size(), kUpdateMarker);
  const int flat_index = ToFlatIndex(cell_index); //把pixel坐标转化为一维索引值
  // 获取对应栅格的指针; mutable_correspondence_cost_cells()是Grid2D的成员函数，返回存放概率值的一维向量, 根据cell坐标，返回该cell中原本的value值
  uint16* cell = &(*mutable_correspondence_cost_cells())[flat_index]; //根据索引值求该cell的值
  // 对处于更新状态的栅格, 不再进行更新了
  if (*cell >= kUpdateMarker) {
    return false;
  }
  // 标记这个索引的栅格已经被更新过; 已更新的信息都存储在update_indices_这个向量中，所以该cell被处理过后它的index要加入到这个向量中
  mutable_update_indices()->push_back(flat_index);
  // 更新栅格值; 根据该pixel返回的值cell来查表，获取更新后应该是什么值。然后把这个值放入到cell原先的地址中。实际就是更新该值
  *cell = table[*cell];
  DCHECK_GE(*cell, kUpdateMarker);
  // 更新bounding_box; mutable_known_cells_box()是Grid2D的成员函数，返回存放已知概率值的一个子区域的盒子。现在就是把该cell放入已知概率值的盒子中
  mutable_known_cells_box()->extend(cell_index.matrix());
  return true;
}

GridType ProbabilityGrid::GetGridType() const {
  return GridType::PROBABILITY_GRID;
}

// Returns the probability of the cell with 'cell_index'.
// 获取 索引 处单元格的占用概率
float ProbabilityGrid::GetProbability(const Eigen::Array2i& cell_index) const {
  if (!limits().Contains(cell_index)) return kMinProbability;
  return CorrespondenceCostToProbability(ValueToCorrespondenceCost(
      correspondence_cost_cells()[ToFlatIndex(cell_index)]));
}

proto::Grid2D ProbabilityGrid::ToProto() const {
  proto::Grid2D result;
  result = Grid2D::ToProto();
  result.mutable_probability_grid_2d();
  return result;
}

// 根据bounding_box对栅格地图进行裁剪到正好包含点云; 在更新子图的过程中，并不能保证更新的数据能够完整覆盖整个子图的所有栅格。该函数就是以最小的矩形框出已经更新的栅格
std::unique_ptr<Grid2D> ProbabilityGrid::ComputeCroppedGrid() const {
  Eigen::Array2i offset;
  CellLimits cell_limits;
  // 根据bounding_box对栅格地图进行裁剪; 接着获取子图的分辨率和最大的xy索引，并构建一个新的ProbabilityGrid对象cropped_grid。
  ComputeCroppedLimits(&offset, &cell_limits);
  const double resolution = limits().resolution();
  // 重新计算最大值坐标
  const Eigen::Vector2d max =
      limits().max() - resolution * Eigen::Vector2d(offset.y(), offset.x());
  // 重新定义概率栅格地图的大小
  std::unique_ptr<ProbabilityGrid> cropped_grid =
      absl::make_unique<ProbabilityGrid>(
          MapLimits(resolution, max, cell_limits), conversion_tables_);
  // 给新栅格地图赋值
  for (const Eigen::Array2i& xy_index : XYIndexRangeIterator(cell_limits)) {
    if (!IsKnown(xy_index + offset)) continue;
    cropped_grid->SetProbability(xy_index, GetProbability(xy_index + offset));
  }

  // 返回新地图的指针
  return std::unique_ptr<Grid2D>(cropped_grid.release());
}

// 获取压缩后的地图栅格数据
bool ProbabilityGrid::DrawToSubmapTexture(
    proto::SubmapQuery::Response::SubmapTexture* const texture,
    transform::Rigid3d local_pose) const {
  Eigen::Array2i offset;
  CellLimits cell_limits;
  // 根据bounding_box对栅格地图进行裁剪
  ComputeCroppedLimits(&offset, &cell_limits);

  std::string cells;
  // 遍历地图, 将栅格数据存入cells
  for (const Eigen::Array2i& xy_index : XYIndexRangeIterator(cell_limits)) {
    if (!IsKnown(xy_index + offset)) {
      cells.push_back(0 /* unknown log odds value */);
      cells.push_back(0 /* alpha */);
      continue;
    }
    // We would like to add 'delta' but this is not possible using a value and
    // alpha. We use premultiplied alpha, so when 'delta' is positive we can
    // add it by setting 'alpha' to zero. If it is negative, we set 'value' to
    // zero, and use 'alpha' to subtract. This is only correct when the pixel
    // is currently white, so walls will look too gray. This should be hard to
    // detect visually for the user, though.
    // 我们想添加 'delta'，但使用值和 alpha 是不可能的
    // 我们使用预乘 alpha，因此当 'delta' 为正时，我们可以通过将 'alpha' 设置为零来添加它。 
    // 如果它是负数，我们将 'value' 设置为零，并使用 'alpha' 进行减法。 这仅在像素当前为白色时才正确，因此墙壁看起来太灰。 
    // 但是，这对于用户来说应该很难在视觉上检测到。
    
    // delta处于[-127, 127]
    const int delta =
        128 - ProbabilityToLogOddsInteger(GetProbability(xy_index + offset));
    const uint8 alpha = delta > 0 ? 0 : -delta;
    const uint8 value = delta > 0 ? delta : 0;
    // 存数据时存了2个值, 一个是栅格值value, 另一个是alpha透明度
    cells.push_back(value);
    cells.push_back((value || alpha) ? alpha : 1);
  }

  // 保存地图栅格数据时进行压缩
  common::FastGzipString(cells, texture->mutable_cells());
  // 填充地图描述信息
  texture->set_width(cell_limits.num_x_cells);
  texture->set_height(cell_limits.num_y_cells);
  const double resolution = limits().resolution();
  texture->set_resolution(resolution);
  const double max_x = limits().max().x() - resolution * offset.y();
  const double max_y = limits().max().y() - resolution * offset.x();
  *texture->mutable_slice_pose() = transform::ToProto(
      local_pose.inverse() *
      transform::Rigid3d::Translation(Eigen::Vector3d(max_x, max_y, 0.)));

  return true;
}

}  // namespace mapping
}  // namespace cartographer
