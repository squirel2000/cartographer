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

#include "cartographer/mapping/2d/probability_grid_range_data_inserter_2d.h"

#include <cstdlib>

#include "Eigen/Core"
#include "Eigen/Geometry"
#include "cartographer/mapping/2d/xy_index.h"
#include "cartographer/mapping/internal/2d/ray_to_pixel_mask.h"
#include "cartographer/mapping/probability_values.h"
#include "glog/logging.h"

namespace cartographer {
namespace mapping {
namespace {

// Factor for subpixel accuracy of start and end point for ray casts.
constexpr int kSubpixelScale = 1000;

// 根据点云的bounding box, 看是否需要对地图进行扩张
void GrowAsNeeded(const sensor::RangeData& range_data,
                  ProbabilityGrid* const probability_grid) {
  // 找到点云的bounding_box
  Eigen::AlignedBox2f bounding_box(range_data.origin.head<2>());
  // Padding around bounding box to avoid numerical issues at cell boundaries.
  constexpr float kPadding = 1e-6f;
  for (const sensor::RangefinderPoint& hit : range_data.returns) {
    bounding_box.extend(hit.position.head<2>());
  }
  for (const sensor::RangefinderPoint& miss : range_data.misses) {
    bounding_box.extend(miss.position.head<2>());
  }
  // 是否对地图进行扩张
  probability_grid->GrowLimits(bounding_box.min() -
                               kPadding * Eigen::Vector2f::Ones());
  probability_grid->GrowLimits(bounding_box.max() +
                               kPadding * Eigen::Vector2f::Ones());
}

// http://gaoyichao.com/Xiaotu/?book=Cartographer%E6%BA%90%E7%A0%81%E8%A7%A3%E8%AF%BB&title=%E6%9F%A5%E6%89%BE%E8%A1%A8%E4%B8%8E%E5%8D%A0%E7%94%A8%E6%A0%85%E6%A0%BC%E6%9B%B4%E6%96%B0
// CastRay中就是把RangeData中包含的一系列点 [公式] ，计算出一条从原点到 [公式] 的射线，射线端点处的点是Hit，射线中间的点是Free。把所有这些点要在地图上把相应的cell进行更新。
// 在CastRays中处理RangeData的数据时，又用到了一个超分辨率像素的概念，即将一个cell又进一步划分成kSubpixelScale × kSubpixelScale个cell，其中，kSubpixelScale = 1000。这样，在计算origin到end之间的射线方程时可以做到更精确。
/**
 * @brief 根据雷达点对栅格地图进行更新
 * 
 * @param[in] range_data 
 * @param[in] hit_table 更新占用栅格时的查找表
 * @param[in] miss_table 更新空闲栅格时的查找表
 * @param[in] insert_free_space 
 * @param[in] probability_grid 栅格地图
 */
void CastRays(const sensor::RangeData& range_data,
              const std::vector<uint16>& hit_table,
              const std::vector<uint16>& miss_table,
              const bool insert_free_space, ProbabilityGrid* probability_grid) {
  // 根据雷达数据调整地图范围
  GrowAsNeeded(range_data, probability_grid);

  const MapLimits& limits = probability_grid->limits();
  // 定义一个超分辨率像素，把当前的分辨率又划分成了kSubpixelScale份，这里int kSubpixelScale = 1000 (为了提高RayCasting的精度)
  const double superscaled_resolution = limits.resolution() / kSubpixelScale;
  // 根据超分辨率像素又生成了一个新的MapLimits
  const MapLimits superscaled_limits(
      superscaled_resolution, limits.max(),
      CellLimits(limits.cell_limits().num_x_cells * kSubpixelScale,
                 limits.cell_limits().num_y_cells * kSubpixelScale));
  // 雷达原点在地图中的像素坐标, 作为画线的起始坐标
  const Eigen::Array2i begin =
      superscaled_limits.GetCellIndex(range_data.origin.head<2>());
  // Compute and add the end points.
  std::vector<Eigen::Array2i> ends;
  // reserve 函数用来给vector预分配存储区大小，即capacity的值 ，但是没有给这段内存进行初始化。
  // reserve 的参数n是推荐预分配内存的大小，实际分配的可能等于或大于这个值，即n大于capacity的值，就会reallocate内存 capacity的值会大于或者等于n 。
  // 这样，当ector调用push_back函数使得size 超过原来的默认分配的capacity值时 避免了内存重分配开销。
  // 这里就是根据returns集合的大小，给ends预分配一块存储区
  ends.reserve(range_data.returns.size());
  for (const sensor::RangefinderPoint& hit : range_data.returns) {
    // 计算hit点在地图中的像素坐标, 作为画线的终止点坐标; 遍历returns这个集合，把每个点先压入ends中，
    ends.push_back(superscaled_limits.GetCellIndex(hit.position.head<2>()));
    // ens.back()返回的是vector中的最末尾项，也就是我们刚刚压入vector中的那一项；
    // 这里我猜测，hit_table就是预先计算好的。如果一个cell，原先的值是value，那么在检测到hit后应该更新为多少
    // 更新hit点的栅格值
    probability_grid->ApplyLookupTable(ends.back() / kSubpixelScale, hit_table);  // 调用了占用栅格对象的ApplyLookupTable函数查表更新占用概率。这次调用的传参有两个， 第一个参数将精细栅格下hit点索引重新转换成原始栅格分辨率下的索引，第二个参数就是待查的hit表
  }

  // 如果配置项里设置是不考虑free space。那么函数到这里结束，只处理完hit后返回即可
  // 否则的话，需要计算那条射线，射线中间的点都是free space，同时，没有检测到hit的misses集里也都是free
  if (!insert_free_space) {
    return;
  }

  // Now add the misses. // 处理origin跟hit之间的射线中间的点 (处理射线起点到hit点之间的栅格，把它们都看做是发生了miss事件的栅格，查找miss_table更新占用概率。但是需要注意这里的begin和end都是精细栅格下的索引)
  for (const Eigen::Array2i& end : ends) {
    std::vector<Eigen::Array2i> ray =
        RayToPixelMask(begin, end, kSubpixelScale);
    for (const Eigen::Array2i& cell_index : ray) {
      // 从起点到end点之前, 更新miss点的栅格值
      probability_grid->ApplyLookupTable(cell_index, miss_table);
    }
  }

  // Finally, compute and add empty rays based on misses in the range data.
  for (const sensor::RangefinderPoint& missing_echo : range_data.misses) {
    std::vector<Eigen::Array2i> ray = RayToPixelMask(
        begin, superscaled_limits.GetCellIndex(missing_echo.position.head<2>()),
        kSubpixelScale);
    for (const Eigen::Array2i& cell_index : ray) {
      // 从起点到misses点之前, 更新miss点的栅格值
      probability_grid->ApplyLookupTable(cell_index, miss_table);
    }
  }
}
}  // namespace

proto::ProbabilityGridRangeDataInserterOptions2D
CreateProbabilityGridRangeDataInserterOptions2D(
    common::LuaParameterDictionary* parameter_dictionary) {
  proto::ProbabilityGridRangeDataInserterOptions2D options;
  options.set_hit_probability(
      parameter_dictionary->GetDouble("hit_probability"));
  options.set_miss_probability(
      parameter_dictionary->GetDouble("miss_probability"));
  options.set_insert_free_space(
      parameter_dictionary->HasKey("insert_free_space")
          ? parameter_dictionary->GetBool("insert_free_space")
          : true);
  CHECK_GT(options.hit_probability(), 0.5);
  CHECK_LT(options.miss_probability(), 0.5);
  return options;
}

// 写入器的构造, 新建了2个查找表; 栅格地图以差异比(Odd)的形式表示占用概率，那么更新过程就只是一个乘法运算，效率也还可以。但Cartographer还是想要进一步的提高效率，它以空间换取时间，构建了hit表和miss表。 如果发生了hit事件，就从hit表中查找更新后的数据。发生了miss事件则查找miss表
ProbabilityGridRangeDataInserter2D::ProbabilityGridRangeDataInserter2D(
    const proto::ProbabilityGridRangeDataInserterOptions2D& options)
    : options_(options),
      // 生成更新占用栅格时的查找表 // param: hit_probability
      hit_table_(ComputeLookupTableToApplyCorrespondenceCostOdds(
          Odds(options.hit_probability()))),    // 0.55
      // 生成更新空闲栅格时的查找表 // param: miss_probability
      miss_table_(ComputeLookupTableToApplyCorrespondenceCostOdds(
          Odds(options.miss_probability()))) {} // 0.49

/**
 * @brief 将点云写入栅格地图
 * 
 * @param[in] range_data 要写入地图的点云
 * @param[in] grid 栅格地图
 */
void ProbabilityGridRangeDataInserter2D::Insert(const sensor::RangeData& range_data, GridInterface* const grid) const {
  ProbabilityGrid* const probability_grid = static_cast<ProbabilityGrid*>(grid);  // 将grid强制转换为ProbabilityGrid类型的数据， 所以这个插入器只适配ProbabilityGrid类型的栅格地图
  CHECK(probability_grid != nullptr);
  // By not finishing the update after hits are inserted, we give hits priority (i.e. no hits will be ignored because of a miss in the same cell).
  // param: insert_free_space
  CastRays(range_data, hit_table_, miss_table_, options_.insert_free_space(), probability_grid);  //调用CastRays函数更新Grid
  probability_grid->FinishUpdate();
}

}  // namespace mapping
}  // namespace cartographer
