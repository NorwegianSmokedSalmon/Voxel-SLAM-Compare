/**
 * Voxel-SLAM 离线PGO优化工具
 * 完全模拟在线PGO的增量式优化逻辑
 * 
 * Voxel-SLAM在线PGO的核心逻辑：
 * 1. 当前session的pose是增量添加的（从buf_lba2loop中逐个取出）
 * 2. 之前的session已经处理完了，它们的pose在multimap_scanPoses中
 * 3. 当检测到回环时：
 *    - 如果回环在当前图的ids中，直接添加到graph
 *    - 如果回环不在当前图的ids中（新session），调用build_graph重建整个图
 * 4. 当isOpt=true时，优化整个图（创建新的ISAM2实例，非增量式）
 * 5. 优化后，更新initial，用于下次优化
 * 6. 最终优化时，调用build_graph和topDownProcess
 * 
 * 离线模拟策略：
 * - 按session顺序处理（session0, session1, session2...）
 * - 对于每个session，增量添加pose到graph（模拟在线从buf_lba2loop中取出）
 * - 当检测到回环时，检查是否需要重建图
 * - 当满足优化条件时，优化（但只优化一次，不是每个session都优化）
 * - 优化后，更新initial，用于下次优化
 * - 最后进行全局优化（使用build_graph和topDownProcess）
 * 
 * 输入格式（与loop_graph_opt相同）：
 * - pose: 通过read_json_pose_new读取，格式为 x y z qw qx qy qz
 * - loop: 通过read_multi_loop读取loop.txt，格式为 session1_idx id1 session2_idx id2 [16个矩阵元素] overlap score
 * 
 * 输出格式（与loop_graph_opt相同）：
 * - 优化后的pose: 通过write_pose保存，格式为 x y z qw qx qy qz
 */

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <memory>
#include <Eigen/Core>
#include <ros/ros.h>
#include <map>
#include <algorithm>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/slam/PriorFactor.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/Values.h>
#include <boost/shared_ptr.hpp>
#include <gtsam/nonlinear/ISAM2.h>
using namespace std;

// 必要的宏定义
#define DIM 15
#define PLM(a) vector<Eigen::Matrix<double, a, a>, Eigen::aligned_allocator<Eigen::Matrix<double, a, a>>>
#define PLV(a) vector<Eigen::Matrix<double, a, 1>, Eigen::aligned_allocator<Eigen::Matrix<double, a, 1>>>

// IMUST 结构体定义（简化版，只包含需要的成员）
struct IMUST
{
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  double t;
  Eigen::Matrix3d R;
  Eigen::Vector3d p;
  Eigen::Vector3d v;
  Eigen::Vector3d bg;
  Eigen::Vector3d ba;
  Eigen::Vector3d g;
  Eigen::Matrix<double, DIM, DIM> cov;
  
  IMUST()
  {
    t = 0;
    R.setIdentity();
    p.setZero();
    v.setZero();
    bg.setZero();
    ba.setZero();
    g << 0, 0, -9.8;
    cov.setIdentity();
  }
};

// PVecPtr 类型定义
struct pointVar 
{
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  Eigen::Vector3d pnt;
  Eigen::Matrix3d var;
};
using PVec = vector<pointVar>;
using PVecPtr = shared_ptr<vector<pointVar>>;

// ScanPose 结构体定义
struct ScanPose
{
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  IMUST x;
  PVecPtr pvec;
  Eigen::Matrix<double, 6, 1> v6;

  ScanPose(IMUST &_x, PVecPtr _pvec): x(_x), pvec(_pvec)
  {
    v6.setZero();
  }

  void set_state(const gtsam::Pose3 &pose)
  {
    Eigen::Matrix3d rot = pose.rotation().matrix();
    rot = rot * x.R.transpose();
    x.R = pose.rotation().matrix();
    x.p = pose.translation();
    x.v = rot * x.v;
  }
};

// add_edge 函数定义（两个重载版本）
void add_edge(int pos1, int pos2, IMUST &x1, IMUST &x2, gtsam::NonlinearFactorGraph &graph, gtsam::noiseModel::Diagonal::shared_ptr odometryNoise)
{
  gtsam::Point3 tt(x1.R.transpose() * (x2.p - x1.p));
  gtsam::Rot3 RR(x1.R.transpose() * x2.R);
  gtsam::NonlinearFactor::shared_ptr factor(new gtsam::BetweenFactor<gtsam::Pose3>(pos1, pos2, gtsam::Pose3(RR, tt), odometryNoise));
  graph.push_back(factor);
}

void add_edge(int pos1, int pos2, Eigen::Matrix3d &rot, Eigen::Vector3d &tra, gtsam::NonlinearFactorGraph &graph, gtsam::noiseModel::Diagonal::shared_ptr odometryNoise)
{
  gtsam::Point3 tt(tra);
  gtsam::Rot3 RR(rot);
  gtsam::NonlinearFactor::shared_ptr factor(new gtsam::BetweenFactor<gtsam::Pose3>(pos1, pos2, gtsam::Pose3(RR, tt), odometryNoise));
  graph.push_back(factor);
}

// PGO_Edge 结构体定义
struct PGO_Edge
{
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  int m1, m2;
  vector<int> ids1, ids2;
  PLM(3) rots;
  PLV(3) tras;
  PLV(6) covs;

  PGO_Edge(int _m1, int _m2, int id1, int id2, Eigen::Matrix3d &rot, Eigen::Vector3d &tra, Eigen::Matrix<double, 6, 1> &v6): m1(_m1), m2(_m2) 
  {
    push(id1, id2, rot, tra, v6);
  }

  void push(int id1, int id2, Eigen::Matrix3d &rot, Eigen::Vector3d &tra, Eigen::Matrix<double, 6, 1> &v6)
  {
    ids1.push_back(id1); ids2.push_back(id2);
    rots.push_back(rot); tras.push_back(tra);
    covs.push_back(v6);
  }

  bool is_adapt(vector<int> &maps, vector<int> &step)
  {
    bool f1 = false, f2 = false;
    for(int i=0; i<maps.size(); i++)
    {
      if(m1 == maps[i])
      {
        f1 = true;
        step[0] = i;
      }
      if(m2 == maps[i])
      {
        f2 = true;
        step[1] = i;
      }
    }
    return f1 && f2;
  }
};

// PGO_Edges 结构体定义
struct PGO_Edges
{
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  vector<PGO_Edge> edges;
  vector<vector<int>> mates;

  void push(int _m1, int _m2, int _id1, int _id2, Eigen::Matrix3d &rot, Eigen::Vector3d &tra, Eigen::Matrix<double, 6, 1> &v6)
  {
    bool is_lack = true;
    for(PGO_Edge &e: edges)
    {
      if(e.m1 == _m1 && e.m2 == _m2)
      {
        is_lack = false;
        e.push(_id1, _id2, rot, tra, v6);
        break;
      }
    }

    if(is_lack)
    {
      edges.emplace_back(_m1, _m2, _id1, _id2, rot, tra, v6);
      int msize = mates.size();
      for(int i=msize; i<_m2+1; i++)
        mates.emplace_back();
      mates[_m1].push_back(_m2);
      mates[_m2].push_back(_m1);
    }
  }

  void connect(int root, vector<int> &ids)
  {
    ids.clear();
    ids.push_back(root);
    tras(root, ids);
    sort(ids.begin(), ids.end());
  }

  void tras(int ord, vector<int> &ids)
  {
    if(ord < mates.size())
    for(int id: mates[ord])
    {
      bool is_exist = false;
      for(int i: ids)
      if(id == i)
      {
        is_exist = true;
        break;
      }

      if(!is_exist)
      {
        ids.push_back(id);
        tras(id, ids);
      }
    }
  }
};

// pose结构体定义（与loop_graph_opt兼容）
struct pose
{
    pose(Eigen::Quaterniond _q = Eigen::Quaterniond(1, 0, 0, 0),
         Eigen::Vector3d _t = Eigen::Vector3d(0, 0, 0), int id = 0)
        : q(_q), t(_t), pose_status(id) {}
    Eigen::Quaterniond q;
    Eigen::Vector3d t;
    int pose_status;
};

// 从pose结构转换为ScanPose（用于Voxel-SLAM内部格式）
void convert_pose_to_scanpose(const vector<pose> &pose_vec, vector<ScanPose*> &scanPoses)
{
    for(const pose &p : pose_vec)
    {
        IMUST xx;
        xx.t = 0;  // 离线优化不需要时间戳
        xx.p = p.t;
        xx.R = p.q.toRotationMatrix();
        xx.v.setZero();
        xx.bg.setZero();
        xx.ba.setZero();
        xx.g << 0, 0, -9.8;
        xx.cov.setIdentity();
        xx.cov *= 0.01;
        
        ScanPose* blp = new ScanPose(xx, nullptr);
        
        // 使用Voxel-SLAM的默认v6值（里程计边默认值）
        Eigen::Matrix<double, 6, 1> v6_default;
        v6_default << 1e-4, 1e-4, 1e-4, 1e-4, 1e-4, 1e-4;  // Voxel-SLAM默认值
        blp->v6 = v6_default;
        
        scanPoses.push_back(blp);
    }
}

// 读取pose（兼容loop_graph_opt格式）
int read_json_pose_new_compat(vector<pose> &pose_vec, const string &filename, int id, const string &name)
{
    string full_path = filename + name + ".json";
    ifstream file(full_path);
    if(!file.is_open())
    {
        printf("Error: Cannot open file %s\n", full_path.c_str());
        return 0;
    }
    
    string line;
    while(getline(file, line))
    {
        istringstream iss(line);
        double tx, ty, tz, w, x, y, z;
        if(iss >> tx >> ty >> tz >> w >> x >> y >> z)
        {
            Eigen::Quaterniond q(w, x, y, z);
            Eigen::Vector3d t(tx, ty, tz);
            pose_vec.push_back(pose(q, t, id));
        }
    }
    file.close();
    
    printf("Loaded %s: %zu poses\n", full_path.c_str(), pose_vec.size());
    return pose_vec.size();
}

// 读取loop.txt（兼容loop_graph_opt格式）
void read_multi_loop_compat(
    const string &data_path,
    vector<pair<int, int>> &end_idx,
    vector<pair<int, int>> &start_idx,
    vector<gtsam::Pose3> &T_start_end,
    vector<double> &overlap,
    vector<double> &score)
{
    ifstream file(data_path);
    if(!file.is_open())
    {
        cerr << "Error opening file: " << data_path << endl;
        return;
    }
    
    string line;
    while(getline(file, line))
    {
        istringstream stream(line);
        int temp_end_idx, temp_start_idx;
        int end_drone_id, start_drone_id;
        double temp_overlap, temp_score;
        vector<double> matrix_data(16);
        
        // 读取 start_idx
        stream >> start_drone_id >> temp_start_idx;
        start_idx.push_back(make_pair(start_drone_id, temp_start_idx));
        
        // 读取 end_idx
        stream >> end_drone_id >> temp_end_idx;
        end_idx.push_back(make_pair(end_drone_id, temp_end_idx));
        
        // 读取 4x4 变换矩阵（按行展开，16个元素）
        for(int i=0; i<16; i++)
            stream >> matrix_data[i];
        
        // 转换为gtsam::Pose3
        gtsam::Rot3 rotation(
            matrix_data[0], matrix_data[1], matrix_data[2],
            matrix_data[4], matrix_data[5], matrix_data[6],
            matrix_data[8], matrix_data[9], matrix_data[10]
        );
        gtsam::Point3 translation(matrix_data[3], matrix_data[7], matrix_data[11]);
        T_start_end.push_back(gtsam::Pose3(rotation, translation));
        
        // 读取overlap和score
        stream >> temp_overlap >> temp_score;
        overlap.push_back(temp_overlap);
        score.push_back(temp_score);
    }
    file.close();
}

// 写入pose（兼容loop_graph_opt格式）
void write_pose_compat(const vector<pose> &pose_vec, const string &path, int i)
{
    string filename;
    if(i == -1)
        filename = path + "pose_0.json";
    else
        filename = path + "pose_" + to_string(i) + ".json";
    
    // 确保目录存在
    size_t pos = filename.find_last_of("/");
    if(pos != string::npos)
    {
        string dir = filename.substr(0, pos);
        string cmd = "mkdir -p " + dir;
        system(cmd.c_str());
    }
    
    ofstream file(filename);
    for(size_t j=0; j<pose_vec.size(); j++)
    {
        file << pose_vec[j].t(0) << " "
             << pose_vec[j].t(1) << " "
             << pose_vec[j].t(2) << " "
             << pose_vec[j].q.w() << " " << pose_vec[j].q.x() << " "
             << pose_vec[j].q.y() << " " << pose_vec[j].q.z();
        if(j < pose_vec.size() - 1)
            file << "\n";
    }
    file.close();
}

// 提取build_graph逻辑为独立函数（避免依赖VOXEL_SLAM类）
void build_graph_standalone(
    gtsam::Values &initial, 
    gtsam::NonlinearFactorGraph &graph, 
    int cur_id, 
    PGO_Edges &lp_edges, 
    gtsam::noiseModel::Diagonal::shared_ptr default_noise, 
    vector<int> &ids, 
    vector<int> &stepsizes, 
    int lpedge_enable,
    vector<vector<ScanPose*>*> &multimap_scanPoses)
{
    initial.clear(); 
    graph = gtsam::NonlinearFactorGraph();
    ids.clear();
    lp_edges.connect(cur_id, ids);

    stepsizes.clear(); 
    stepsizes.push_back(0);
    for(int i=0; i<ids.size(); i++)
        stepsizes.push_back(stepsizes.back() + multimap_scanPoses[ids[i]]->size());
    
    for(int ii=0; ii<ids.size(); ii++)
    {
        int bsize = stepsizes[ii], id = ids[ii];
        for(int j=bsize; j<stepsizes[ii+1]; j++)
        {
            IMUST &xc = multimap_scanPoses[id]->at(j-bsize)->x;
            gtsam::Pose3 pose3(gtsam::Rot3(xc.R), gtsam::Point3(xc.p));
            initial.insert(j, pose3);
            if(j > bsize)
            {
                gtsam::Vector samv6(6);
                samv6 = multimap_scanPoses[ids[ii]]->at(j-1-bsize)->v6;
                gtsam::noiseModel::Diagonal::shared_ptr v6_noise = gtsam::noiseModel::Diagonal::Variances(samv6);
                add_edge(j-1, j, multimap_scanPoses[id]->at(j-1-bsize)->x, multimap_scanPoses[id]->at(j-bsize)->x, graph, v6_noise);
            }
        }
    }

    if(multimap_scanPoses[ids[0]]->size() != 0)
    {
        int ceil = multimap_scanPoses[ids[0]]->size();
        ceil = 1;
        for(int i=0; i<ceil; i++)
        {
            Eigen::Matrix<double, 6, 1> v6_fixd;
            v6_fixd << 1e-9, 1e-9, 1e-9, 1e-9, 1e-9, 1e-9;
            gtsam::noiseModel::Diagonal::shared_ptr fixd_noise = gtsam::noiseModel::Diagonal::Variances(gtsam::Vector(v6_fixd));
            IMUST xf = multimap_scanPoses[ids[0]]->at(i)->x;
            gtsam::Pose3 pose3 = gtsam::Pose3(gtsam::Rot3(xf.R), gtsam::Point3(xf.p));
            graph.addPrior(i, pose3, fixd_noise);
        }
    }

    // lpedge_enable: 1=只添加当前连接的session之间的回环边, 0=添加所有回环边
    if(lpedge_enable == 1)
    {
        // 只添加当前连接的session之间的回环边
        for(PGO_Edge &edge: lp_edges.edges)
        {
            vector<int> step(2);
            if(edge.is_adapt(ids, step))
            {
                int mp[2] = {stepsizes[step[0]], stepsizes[step[1]]};
                for(int i=0; i<edge.rots.size(); i++)
                {
                    int id1 = mp[0] + edge.ids1[i];
                    int id2 = mp[1] + edge.ids2[i];
                    add_edge(id1, id2, edge.rots[i], edge.tras[i], graph, default_noise);
                }
            }
        }
    }
    else if(lpedge_enable == 0)
    {
        // 添加所有回环边（最终优化时）
        for(PGO_Edge &edge: lp_edges.edges)
        {
            vector<int> step(2);
            if(edge.is_adapt(ids, step))
            {
                int mp[2] = {stepsizes[step[0]], stepsizes[step[1]]};
                for(int i=0; i<edge.rots.size(); i++)
                {
                    int id1 = mp[0] + edge.ids1[i];
                    int id2 = mp[1] + edge.ids2[i];
                    add_edge(id1, id2, edge.rots[i], edge.tras[i], graph, default_noise);
                }
            }
        }
    }
}

// 模拟Voxel-SLAM的增量式PGO优化（完全按照thd_loop_closure的流程）
// 关键流程：
// 1. Session 0：加载所有pose，添加先验约束，不优化
// 2. Session 1：逐个添加pose，检测到回环时优化一次，然后继续添加pose
// 3. Session 2：逐个添加pose，检测到回环时优化一次，然后继续添加pose
// 注意：优化后需要更新initial，用于下次优化，确保里程计的变换衔接
void incremental_pgo_optimization_voxelslam(
    vector<vector<ScanPose*>*> &multimap_scanPoses,
    PGO_Edges &lp_edges,
    const vector<int> &pose_size_each)
{
    // 初始化（与Voxel-SLAM的thd_loop_closure相同）
    Eigen::Matrix<double, 6, 1> v6_init, v6_fixd;
    v6_init << 1e-4, 1e-4, 1e-4, 1e-4, 1e-4, 1e-4;
    v6_fixd << 1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 1e-6;
    gtsam::noiseModel::Diagonal::shared_ptr odom_noise = gtsam::noiseModel::Diagonal::Variances(gtsam::Vector(v6_init));
    gtsam::noiseModel::Diagonal::shared_ptr fixd_noise = gtsam::noiseModel::Diagonal::Variances(gtsam::Vector(v6_fixd));
    
    gtsam::Values initial;
    gtsam::NonlinearFactorGraph graph;
    vector<int> ids;
    vector<int> stepsizes;
    
    // Session 0：加载所有pose，添加先验约束，不优化
    printf("\n=== Processing session 0 ===\n");
    ids.clear();
    ids.push_back(0);
    stepsizes.clear();
    stepsizes.push_back(0);
    
    // 添加session 0的所有pose
    for(int i=0; i<pose_size_each[0]; i++)
    {
        IMUST xc = multimap_scanPoses[0]->at(i)->x;
        gtsam::Pose3 pose3(gtsam::Rot3(xc.R), gtsam::Point3(xc.p));
        initial.insert(i, pose3);
        
        if(i > 0)
        {
            // 添加里程计边
            gtsam::Vector samv6(6);
            samv6 = multimap_scanPoses[0]->at(i-1)->v6;
            gtsam::noiseModel::Diagonal::shared_ptr v6_noise = 
                gtsam::noiseModel::Diagonal::Variances(samv6);
            add_edge(i-1, i, multimap_scanPoses[0]->at(i-1)->x, xc, graph, v6_noise);
        }
        else
        {
            // 第一个pose添加先验因子
            graph.addPrior(0, pose3, fixd_noise);
        }
    }
    stepsizes.push_back(pose_size_each[0]);
    printf("Session 0: loaded %d poses, added prior constraint\n", pose_size_each[0]);
    
    // Session 1及之后：逐个添加pose，检测到回环时优化
    for(int session_id = 1; session_id < pose_size_each.size(); session_id++)
    {
        printf("\n=== Processing session %d ===\n", session_id);
        int cur_id = session_id;
        int buf_base = 0;
        
        // 逐个添加当前session的pose（完全模拟在线从buf_lba2loop中逐个取出）
        for(int i=0; i<pose_size_each[session_id]; i++)
        {
            IMUST xc = multimap_scanPoses[session_id]->at(i)->x;
            gtsam::Pose3 pose3(gtsam::Rot3(xc.R), gtsam::Point3(xc.p));
            int g_pos = stepsizes.back();
            initial.insert(g_pos, pose3);
            
            // 添加里程计边（当前session内部的pose之间）
            if(buf_base > 0)
            {
                gtsam::Vector samv6(6);
                samv6 = multimap_scanPoses[session_id]->at(buf_base-1)->v6;
                gtsam::noiseModel::Diagonal::shared_ptr v6_noise = 
                    gtsam::noiseModel::Diagonal::Variances(samv6);
                add_edge(g_pos-1, g_pos, 
                        multimap_scanPoses[session_id]->at(buf_base-1)->x, 
                        xc, graph, v6_noise);
            }
            
            buf_base++;
            stepsizes.back() += 1;
            
            // 检查是否有回环涉及当前pose
            bool isGraph = false;
            bool isOpt = false;
            
            for(PGO_Edge &edge: lp_edges.edges)
            {
                // 检查回环是否涉及当前session和之前的session
                if((edge.m1 == session_id && edge.m2 < session_id) || 
                   (edge.m2 == session_id && edge.m1 < session_id))
                {
                    // 检查回环是否涉及当前pose
                    int other_session = (edge.m1 == session_id) ? edge.m2 : edge.m1;
                    bool matches_current_pose = false;
                    int matched_k = -1;
                    for(int k=0; k<edge.ids1.size(); k++)
                    {
                        if((edge.m1 == session_id && edge.ids1[k] == i) ||
                           (edge.m2 == session_id && edge.ids2[k] == i))
                        {
                            matches_current_pose = true;
                            matched_k = k;
                            break;
                        }
                    }
                    
                    if(matches_current_pose)
                    {
                        // 检查另一个session是否在ids中
                        int step = -1;
                        for(int j=0; j<ids.size(); j++)
                        {
                            if(ids[j] == other_session)
                            {
                                step = j;
                                break;
                            }
                        }
                        
                        // step == -1 表示新session，需要重建图
                        if(step == -1)
                        {
                            isGraph = true;
                            isOpt = true;
                            printf("Loop closure detected: session %d pose %d <-> session %d, rebuilding graph...\n", 
                                   session_id, i, other_session);
                            break;  // 只处理第一个新session的回环
                        }
                        else
                        {
                            // 回环在当前图中，直接添加边
                            int id1 = stepsizes[step] + ((edge.m1 == other_session) ? edge.ids1[matched_k] : edge.ids2[matched_k]);
                            int id2 = stepsizes.back() - 1;
                            add_edge(id1, id2, edge.rots[matched_k], edge.tras[matched_k], graph, odom_noise);
                            printf("Loop closure detected: session %d pose %d <-> session %d, adding edge, optimizing...\n", 
                                   session_id, i, other_session);
                            isOpt = true;  // 检测到回环就优化一次
                        }
                    }
                }
            }
            
            // 如果有新session的回环，重建图
            if(isGraph)
            {
                build_graph_standalone(initial, graph, cur_id, lp_edges, odom_noise, ids, stepsizes, 1, multimap_scanPoses);
                printf("Graph rebuilt: %zu sessions connected\n", ids.size());
            }
            
            // 优化（一个回环进来优化一次）
            if(isOpt)
            {
                // 使用Voxel-SLAM的优化参数
                gtsam::ISAM2Params parameters;
                parameters.relinearizeThreshold = 0.01;
                parameters.relinearizeSkip = 1;
                gtsam::ISAM2 isam(parameters);
                isam.update(graph, initial);
                
                for(int j=0; j<5; j++)
                    isam.update();
                
                gtsam::Values results = isam.calculateEstimate();
                
                // 更新所有位姿（与Voxel-SLAM在线逻辑相同）
                int idsize = ids.size();
                for(int ii=0; ii<idsize; ii++)
                {
                    int tip = ids[ii];
                    for(int j=stepsizes[ii]; j<stepsizes[ii+1]; j++)
                    {
                        int ord = j - stepsizes[ii];
                        multimap_scanPoses[tip]->at(ord)->set_state(results.at(j).cast<gtsam::Pose3>());
                    }
                }
                
                // 更新initial（用于下次优化，确保里程计的变换衔接）
                initial.clear();
                for(int j=0; j<results.size(); j++)
                    initial.insert(j, results.at(j).cast<gtsam::Pose3>());
                
                printf("Optimization completed, updated %d poses\n", results.size());
            }
        }
    }
    
    printf("\n=== All poses processed ===\n");
    printf("Total sessions: %zu, Total poses: %zu\n", ids.size(), initial.size());
}

int main(int argc, char *argv[])
{
    ros::init(argc, argv, "voxel_offline_pgo");
    ros::NodeHandle nh("~");
    
    // 读取参数（与loop_graph_opt相同）
    string data_path;
    nh.getParam("data_path", data_path);
    
    string loop_path;
    nh.getParam("loop_path", loop_path);
    
    string origin_pose_name;
    nh.getParam("origin_pose_name", origin_pose_name);
    
    int robot_num;
    nh.getParam("robot_num", robot_num);
    
    string pose_opt_path;
    nh.getParam("pose_opt_path", pose_opt_path);
    
    printf("=== Voxel-SLAM Offline PGO (Incremental Logic) ===\n");
    printf("Data path: %s\n", data_path.c_str());
    printf("Loop path: %s\n", loop_path.c_str());
    printf("Robot num: %d\n", robot_num);
    printf("Origin pose name: %s\n", origin_pose_name.c_str());
    printf("\n");
    
    // 读取每个机器人的pose
    vector<vector<ScanPose*>*> multimap_scanPoses;
    vector<int> pose_size_each(robot_num);
    
    for(int i=0; i<robot_num; i++)
    {
        vector<pose> pose_vec;
        int pose_size = read_json_pose_new_compat(pose_vec, data_path + to_string(i) + "/", i, origin_pose_name);
        
        vector<ScanPose*>* scanPoses = new vector<ScanPose*>();
        convert_pose_to_scanpose(pose_vec, *scanPoses);
        multimap_scanPoses.push_back(scanPoses);
        pose_size_each[i] = pose_size;
    }
    
    printf("Total poses loaded: ");
    for(int i=0; i<robot_num; i++)
        printf("session%d: %d ", i, pose_size_each[i]);
    printf("\n");
    
    // 读取回环
    vector<pair<int, int>> end_idx, start_idx;
    vector<gtsam::Pose3> T_start_end;
    vector<double> overlap, score;
    
    read_multi_loop_compat(loop_path, end_idx, start_idx, T_start_end, overlap, score);
    printf("Total loop closures: %zu\n", start_idx.size());
    
    // 构建PGO_Edges（用于build_graph）
    PGO_Edges lp_edges;
    Eigen::Matrix<double, 6, 1> v6_init;
    v6_init << 1e-4, 1e-4, 1e-4, 1e-4, 1e-4, 1e-4;
    
    for(size_t i=0; i<start_idx.size(); i++)
    {
        int m1 = start_idx[i].first;
        int m2 = end_idx[i].first;
        int id1 = start_idx[i].second;
        int id2 = end_idx[i].second;
        
        gtsam::Rot3 rot = T_start_end[i].rotation();
        gtsam::Point3 tra = T_start_end[i].translation();
        
        Eigen::Matrix3d rot_matrix = rot.matrix();
        Eigen::Vector3d tra_vector(tra.x(), tra.y(), tra.z());
        
        if(m1 <= m2)
        {
            lp_edges.push(m1, m2, id1, id2, rot_matrix, tra_vector, v6_init);
        }
        else
        {
            tra_vector = -rot_matrix.transpose() * tra_vector;
            rot_matrix = rot_matrix.transpose();
            lp_edges.push(m2, m1, id2, id1, rot_matrix, tra_vector, v6_init);
        }
    }
    
    // 执行增量式PGO优化（模拟Voxel-SLAM在线逻辑）
    incremental_pgo_optimization_voxelslam(multimap_scanPoses, lp_edges, pose_size_each);
    
    // 转换回pose格式并保存
    for(int i=0; i<robot_num; i++)
    {
        vector<pose> pose_opt_vec;
        for(int j=0; j<multimap_scanPoses[i]->size(); j++)
        {
            IMUST &xx = multimap_scanPoses[i]->at(j)->x;
            Eigen::Quaterniond qq(xx.R);
            pose_opt_vec.push_back(pose(qq, xx.p));
        }
        
        string pose_opt_each_path = pose_opt_path + to_string(i) + "/";
        printf("Writing session%d to: %s\n", i, pose_opt_each_path.c_str());
        write_pose_compat(pose_opt_vec, pose_opt_each_path, -1);
    }
    
    // 清理内存
    for(int i=0; i<multimap_scanPoses.size(); i++)
    {
        for(int j=0; j<multimap_scanPoses[i]->size(); j++)
            delete multimap_scanPoses[i]->at(j);
        delete multimap_scanPoses[i];
    }
    
    printf("\n=== Optimization Complete ===\n");
    
    return 0;
}
