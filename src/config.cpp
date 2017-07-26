/*****************************************************************************
**       PL-SVO: Semi-direct Monocular Visual Odometry by Combining       	**
**                        Points and Line Segments                          **
******************************************************************************
**																			**
**	Copyright(c) 2016, Ruben Gomez-Ojeda, University of Malaga              **
**	Copyright(c) 2016, MAPIR group, University of Malaga					**
**																			**
**  This library extends SVO to the case of also using line-segments, thus  **
**  it is highly based on the previous implementation of SVO:               **
**  https://github.com/uzh-rpg/rpg_svo                                      **
**                                                                          **
**  This program is free software: you can redistribute it and/or modify	**
**  it under the terms of the GNU General Public License (version 3) as		**
**	published by the Free Software Foundation.								**
**																			**
**  This program is distributed in the hope that it will be useful, but		**
**	WITHOUT ANY WARRANTY; without even the implied warranty of				**
**  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the			**
**  GNU General Public License for more details.							**
**																			**
**  You should have received a copy of the GNU General Public License		**
**  along with this program.  If not, see <http://www.gnu.org/licenses/>.	**
**																			**
*****************************************************************************/


#ifdef SVO_USE_ROS
#include <vikit/params_helper.h>
#endif
#include <plsvo/config.h>

namespace plsvo {

Config::Config() :
#ifdef SVO_USE_ROS
    has_pt(vk::getParam<bool>(                          "plsvo/has_pt",true)),
    has_ls(vk::getParam<bool>(                          "plsvo/has_ls",true)),
    init_pt(vk::getParam<bool>(                         "plsvo/init_pt",true)),
    init_ls(vk::getParam<bool>(                         "plsvo/init_ls",true)),
    use_imu(vk::getParam<bool>(                         "plsvo/use_imu",false)),
    trace_name(vk::getParam<string>(                    "plsvo/trace_name","svo")),
    trace_dir(vk::getParam<string>(                     "plsvo/trace_dir","/tmp")),
    n_pyr_levels(vk::getParam<int>(                     "plsvo/n_pyr_levels",3)),
    n_pyr_levels_segs(vk::getParam<int>(                "plsvo/n_pyr_levels_segs",1)),
    core_n_kfs(vk::getParam<int>(                       "plsvo/core_n_kfs",5)),
    map_scale(vk::getParam<double>(                     "plsvo/map_scale",1.0)),
    grid_size(vk::getParam<int>(                        "plsvo/grid_size",25)),
    grid_size_segs(vk::getParam<int>(                   "plsvo/grid_size_segs",25)),
    init_min_disparity(vk::getParam<double>(            "plsvo/init_min_disparity",40.0)),
    init_min_tracked(vk::getParam<int>(                 "plsvo/init_min_tracked",40)),
    init_min_inliers(vk::getParam<int>(                 "plsvo/init_min_inliers",30)),
    klt_max_level(vk::getParam<int>(                    "plsvo/klt_max_level",4)),
    klt_min_level(vk::getParam<int>(                    "plsvo/klt_min_level",2)),
    has_refinement(vk::getParam<bool>(                  "plsvo/has_refinement",true)),
    reproj_thresh(vk::getParam<double>(                 "plsvo/reproj_thresh",2.0)),
    poseoptim_thresh(vk::getParam<double>(              "plsvo/poseoptim_thresh",2.0)),
    poseoptim_num_iter(vk::getParam<int>(               "plsvo/poseoptim_num_iter",10)),
    poseoptim_num_iter_ref(vk::getParam<int>(           "plsvo/poseoptim_num_iter_ref",3)),
    structureoptim_max_pts(vk::getParam<int>(           "plsvo/structureoptim_max_pts",20)),
    structureoptim_num_iter(vk::getParam<int>(          "plsvo/structureoptim_num_iter",5)),
    structureoptim_max_segs(vk::getParam<int>(          "plsvo/structureoptim_max_segs",20)),
    structureoptim_num_iter_segs(vk::getParam<int>(     "plsvo/structureoptim_num_iter_segs",5)),
    loba_thresh(vk::getParam<double>(                   "plsvo/loba_thresh",2.0)),
    loba_robust_huber_width(vk::getParam<double>(       "plsvo/loba_robust_huber_width",1.0)),
    loba_num_iter(vk::getParam<int>(                    "plsvo/loba_num_iter",0)),
    kfselect_mindist_t(vk::getParam<double>(            "plsvo/kfselect_mindist_t",0.06)),
    kfselect_mindist_r(vk::getParam<double>(            "plsvo/kfselect_mindist_r",3.0)),
    triang_min_corner_score(vk::getParam<double>(       "plsvo/triang_min_corner_score",20.0)),
    lsd_min_length(vk::getParam<double>(                "plsvo/lsd_min_length",0.15)),
    img_imu_delay(vk::getParam<double>(                 "plsvo/img_imu_delay",0.0)),
    triang_half_patch_size(vk::getParam<int>(           "plsvo/triang_half_patch_size",4)),
    subpix_n_iter(vk::getParam<int>(                    "plsvo/subpix_n_iter",10)),
    max_n_kfs(vk::getParam<int>(                        "plsvo/max_n_kfs",0)),
    max_fts(vk::getParam<int>(                          "plsvo/max_fts",100)),
    max_fts_segs(vk::getParam<int>(                     "plsvo/max_fts_segs",100)),
    quality_min_fts(vk::getParam<int>(                  "plsvo/quality_min_fts",20)),
    quality_max_drop_fts(vk::getParam<int>(             "plsvo/quality_max_drop_fts",50)),
    quality_min_fts_segs(vk::getParam<int>(             "plsvo/quality_min_fts_segs",20)),
    quality_max_drop_fts_segs(vk::getParam<int>(        "plsvo/quality_max_drop_fts_segs",20))
#else
    has_pt(true),
    has_ls(true),
    init_pt(true),
    init_ls(true),
    trace_name("svo"),
    trace_dir("/tmp"),
    n_pyr_levels(3),
    n_pyr_levels_segs(1),
    use_imu(false),
    core_n_kfs(5),
    map_scale(1.0),
    grid_size(25),
    grid_size_segs(25),
    init_min_disparity(40.0),
    init_min_tracked(40),
    init_min_inliers(30),
    klt_max_level(4),
    klt_min_level(2),
    has_refinement(true),
    reproj_thresh(2.0),
    poseoptim_thresh(2.0),
    poseoptim_num_iter(10),
    poseoptim_num_iter_ref(3),
    structureoptim_max_pts(20),
    structureoptim_num_iter(5),
    structureoptim_max_segs(20),
    structureoptim_num_iter_segs(5),
    loba_thresh(2.0),
    loba_robust_huber_width(1.0),
    loba_num_iter(0),
    kfselect_mindist_t(0.06),
    kfselect_mindist_r(3.0),
    triang_min_corner_score(20.0),
    lsd_min_length(0.15),
    triang_half_patch_size(4),
    subpix_n_iter(10),
    max_n_kfs(0),
    img_imu_delay(0.0),
    max_fts(100),
    max_fts_segs(100),
    quality_min_fts(20),
    quality_max_drop_fts(50),
    quality_min_fts_segs(20),
    quality_max_drop_fts_segs(50)
#endif
{}

Config& Config::getInstance()
{
  static Config instance; // Instantiated on first use and guaranteed to be destroyed
  return instance;
}

} // namespace plsvo

