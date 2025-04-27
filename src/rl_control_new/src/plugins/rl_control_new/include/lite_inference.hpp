// lite_inference.hpp
// 20250417 MT
// This file is part of the BitBot project to cook data from/to inference net.

// #pragma once
#ifndef INTERFACE_BITBOT_HPP
#define INTERFACE_BITBOT_HPP
#include "interface_Headers.h"

template <typename T>
struct st_inferenceInput { // 这里12是因为humanoid训练的时候只有12个关节输入
  Eigen::Matrix<T, 12, 1> dof_pos_obs;
  Eigen::Matrix<T, 12, 1> dof_vel_obs;
  Eigen::Matrix<T, 12, 1> last_action = Eigen::Matrix<T, 12, 1>::Zero();
  Eigen::Matrix<T, 3, 1> base_ang_vel;
  Eigen::Matrix<T, 3, 1> projected_gravity;
  Eigen::Matrix<T, 3, 1> commands;
  Eigen::Matrix<T, 2, 1> clock_input_vec;
};

template <typename T, size_t N = 12>
struct st_inferenceOutput { // 这里12是因为humanoid训练的时候只有12个关节输出
  Eigen::Matrix<T, N, 1> dof_pos_net_out;
};

template <typename T, size_t N = 20> struct Outputdata {
  Eigen::Matrix<T, 12, 1> action_output; // 插值后的输出
  Eigen::Matrix<T, N, 1> joint_target_position;
  Eigen::Matrix<T, N, 1> joint_target_torque;
  Eigen::Matrix<T, N, 1> motor_target_position;
  Eigen::Matrix<T, N, 1> motor_target_torque; // 都是弧度制
  Eigen::Matrix<T, N, 1> motor_target_velocity;
  Eigen::Matrix<T, N, 1> motor_target_position_last =
      Eigen::Matrix<T, N, 1>::Zero();
};

template <typename T> struct Config {
  const int policy_frequency = 10;
  float dt = 0.0025;
  size_t run_ctrl_cnt = 0;
  size_t start_delay_cnt = 1000;
  std::array<T, 20> default_position = {{0.0, 0.0, -0.46, 1.01, -0.549, 0.0,
                                         0.0, 0.0, -0.46, 1.01, -0.549, 0.0,
                                         0.0, 0.0, 0.0,   0.0,  0.0,    0.0,
                                         0.0, 0.0}}; // 重写，20个关节，索引按照手册来
  float_inference_net::NetConfigT net_config = {
      .input_config = {.obs_scales_ang_vel = 1.0,
                       .obs_scales_lin_vel = 2.0,
                       .scales_commands = 1.0,
                       .obs_scales_dof_pos = 1.0,
                       .obs_scales_dof_vel = 0.05,
                       .obs_scales_euler = 1.0,
                       .clip_observations = 18.0,
                       .ctrl_model_input_size = 15 * 47,
                       .stack_length = 15,
                       .ctrl_model_unit_input_size = 47},
      .output_config = {.ctrl_clip_action = 18.0,
                        .action_scale = 0.5,
                        .ctrl_action_lower_limit =
                            {
                                -30,
                                -30,
                                -30,
                                -30,
                                -30,
                                -30,
                                -30,
                                -30,
                                -30,
                                -30,
                                -30,
                                -30,
                            },
                        .ctrl_action_upper_limit = {30, 30, 30, 30, 30, 30, 30,
                                                    30, 30, 30, 30, 30},
                        .ctrl_model_output_size = 12},
      .action_default_pos = {
          default_position[0],
          default_position[1],
          default_position[2],
          default_position[3],
          default_position[4], // Left ank pit
          default_position[5], // Left ank roll
          default_position[6],
          default_position[7],
          default_position[8],
          default_position[9],
          default_position[10],
          default_position[11],
      }};
};

template <typename T> struct st_interface {
  st_inferenceInput<T> input;
  st_inferenceOutput<T> net_output;
  Outputdata<T> Outputdata;
  Config<T> config;
};

// 推理过程中其他需要的变量
template <typename T, size_t N = 20> struct sovle_st {
  Eigen::Matrix<T, N, 1> joint_current_position;
  Eigen::Matrix<T, N, 1> joint_current_velocity;
  Eigen::Matrix<T, N, 1> motor_current_position;
  Eigen::Matrix<T, N, 1> motor_current_velocity; // 都是弧度制
  Eigen::Matrix<T, 9, 1> IMU;
  Eigen::Matrix<T, 3, 1> commands;
  T CoM_angle[3];      // CoM rpy orientation
  T CoM_angle_velo[3]; // CoM angular velocity
  T CoM_acc[3];        // CoM acceleration
  T gravity_vec[3] = {0.0, 0.0, -1.0};
  T clock_input = 0;
};
// init the net: 把所有user_function.cpp里和网络部署相关的构建都放在这里

constexpr float deg2rad = M_PI / 180.0;
constexpr float rad2deg = 180.0 / M_PI;
constexpr float rpm2radps = 2.0 * M_PI / 60.0;

template <typename T> class c_interface {
  float_inference_net::Ptr inference_net;

public:
  // void c_interface(st_interface<T> &interface) { this->interface = interface;
  // c_interface() {
  //   this->interface = interface;
  // }
  c_interface() = default;
  // } init the net
  void init(st_interface<float> &inference_data) {
    this->InitPolicy(inference_data);
    this->AnkleSet();
  }

  // prepare data
  void Getdata(sovle_st<T> &raw_data, st_interface<T> &inference_data) {
    // get observation:
    // 算kf，得到关节端observation，imu等其他量在PolicyController里对齐
    this->GetJointObservation(raw_data,
                              inference_data); // dof_pos_obs,dof_vel_obs
    this->GetImuObservation(
        raw_data,
        inference_data); // base_ang_vel,projected_gravity,clock_input_vec,command
  }
  // run the control loop
  void run(st_interface<float> &inference_data) {
    // run the policy controller loop: 算出了关节端action输出
    this->PolicyController(inference_data);
  }
  void set_msg(st_interface<T> &inference_data) {
    // get torque: 算ik，得到电机端目标
    this->SetJointAction(inference_data);
  }

private:
  ParallelAnkle<float>::AnkleParameters left_params;
  ParallelAnkle<float>::AnkleParameters right_params;
  ParallelAnkle<float> left_ankle;
  ParallelAnkle<float> right_ankle;

  void AnkleSet() {
    // 并联踝，需要改参数
    this->left_params.l_bar1 = 0.06;
    this->left_params.l_rod1 = 0.14;
    this->left_params.r_a1 = {0.0, 0.043, 0.141};
    this->left_params.r_b1_0 = {-0.056, 0.043, 0.163};
    this->left_params.r_c1_0 = {-0.056, 0.043, 0.023};
    this->left_params.l_bar2 = 0.06;
    this->left_params.l_rod2 = 0.215;
    this->left_params.r_a2 = {0.0, -0.044, 0.215};
    this->left_params.r_b2_0 = {-0.056, -0.044, 0.237};
    this->left_params.r_c2_0 = {-0.056, -0.044, 0.022};

    this->right_params.l_bar1 = 0.06;
    this->right_params.l_rod1 = 0.215;
    this->right_params.r_a1 = {0.0, -0.044, 0.215};
    this->right_params.r_b1_0 = {-0.056, -0.044, 0.237};
    this->right_params.r_c1_0 = {-0.056, -0.044, 0.022};
    this->right_params.l_bar2 = 0.06;
    this->right_params.l_rod2 = 0.14;
    this->right_params.r_a2 = {0.0, 0.043, 0.141};
    this->right_params.r_b2_0 = {-0.056, 0.043, 0.163};
    this->right_params.r_c2_0 = {-0.056, 0.043, 0.023};

    this->left_ankle = ParallelAnkle<float>(this->left_params, 1e-6);
    this->right_ankle = ParallelAnkle<float>(this->right_params, 1e-6);
  }
  void InitPolicy(st_interface<float> &inference_data) {
    // // Action interpolation buffer
    // inference_data.Outputdata.action_interpolated =
    //     std::vector<std::vector<float>>(
    //         inference_data.config.policy_frequency + 1,
    //         std::vector<float>(12));
    // Create the policy network instance
    // last 4000
    this->inference_net = std::make_unique<float_inference_net>(
        "/home/huahui/Project/hhfc-bitbot/checkpoint/"
        "policy_1.pt",                    // control model  policy_202412271
        inference_data.config.net_config, // net config
        true,                             // use async mode
        inference_data.config.policy_frequency // policy frequency
    );
  }

  void GetImuObservation(sovle_st<float> &raw_data,
                         st_interface<float> &inference_data) {
    // Orientation from IMU
    // test:是否为弧度制
    // CoM_angle[0] = raw_data.CoM_angle[0]; // imu angle
    // CoM_angle[1] = raw_data.CoM_angle[1];
    // CoM_angle[2] = raw_data.CoM_angle[2];

    // This is sort of filter.
    // Acceleration from IMU
    // CoM_acc[0] =
    //     (abs(raw_data.CoM_acc[0]) > 50) ? raw_data.CoM_acc[0] :
    //     raw_data.CoM_acc[0];
    // CoM_acc[1] =
    //     (abs(raw_data.CoM_acc[1]) > 50) ? raw_data.CoM_acc[1] :
    //     raw_data.CoM_acc[1];
    // CoM_acc[2] =
    //     (abs(raw_data.CoM_acc[2]) > 50) ? raw_data.CoM_acc[2] :
    //     raw_data.CoM_acc[2];

    // Angular velocity from IMU
    float angle_velo_x = raw_data.CoM_angle_velo[0]; // imu angular velocity
    float angle_velo_y = raw_data.CoM_angle_velo[1];
    float angle_velo_z = raw_data.CoM_angle_velo[2];

    for (size_t i = 0; i < 3; i++) {
      if (std::isnan(raw_data.CoM_angle[i]))
        raw_data.CoM_angle[i] = 0;
    }
    for (size_t i = 0; i < 3; i++) {
      if (std::isnan(raw_data.CoM_angle_velo[i]))
        raw_data.CoM_angle_velo[i] = 0;
    }

    raw_data.CoM_angle_velo[0] = std::abs(angle_velo_x) > 10
                                     ? raw_data.CoM_angle_velo[0]
                                     : angle_velo_x; // filter the noise
    raw_data.CoM_angle_velo[1] =
        std::abs(angle_velo_y) > 10 ? raw_data.CoM_angle_velo[1] : angle_velo_y;
    raw_data.CoM_angle_velo[2] =
        std::abs(angle_velo_z) > 10 ? raw_data.CoM_angle_velo[2] : angle_velo_z;

    Eigen::Matrix3f RotationMatrix;
    const Eigen::AngleAxisf roll(raw_data.CoM_angle[0],
                                 Eigen::Vector3f::UnitX());
    const Eigen::AngleAxisf pitch(raw_data.CoM_angle[1],
                                  Eigen::Vector3f::UnitY());
    const Eigen::AngleAxisf yaw(raw_data.CoM_angle[2],
                                Eigen::Vector3f::UnitZ());
    RotationMatrix = yaw * pitch * roll;

    // Eigen::Matrix<float, 3, 1> CoMAngleVeloMat << raw_data.CoM_angle_velo[0],
    //     raw_data.CoM_angle_velo[1], raw_data.CoM_angle_velo[2];
    // Eigen::Matrix<float, 3, 1> gravity_vec_mat << raw_data.gravity_vec[0],
    //     raw_data.gravity_vec[1], raw_data.gravity_vec[2];

    Eigen::Matrix<float, 3, 1> CoMAngleVeloMat;
    Eigen::Matrix<float, 3, 1> gravity_vec_mat;
    CoMAngleVeloMat.setZero();
    gravity_vec_mat.setZero();

    for (int i = 0; i < 3; ++i) {
      CoMAngleVeloMat(i, 0) = raw_data.CoM_angle_velo[i];
    }
    for (int i = 0; i < 3; ++i) {
      gravity_vec_mat(i, 0) = raw_data.gravity_vec[i];
    }

    Eigen::Matrix<float, 3, 1> projected_gravity_mat =
        RotationMatrix.transpose() * gravity_vec_mat;

    // inference data prepare
    inference_data.input.base_ang_vel = CoMAngleVeloMat;
    inference_data.input.projected_gravity = {projected_gravity_mat(0, 0),
                                              projected_gravity_mat(1, 0),
                                              projected_gravity_mat(2, 0)};
    inference_data.input.clock_input_vec = {float(sin(raw_data.clock_input)),
                                            float(cos(raw_data.clock_input))};
    inference_data.input.commands = {raw_data.commands[0], raw_data.commands[1],
                                     raw_data.commands[2]};
  }
  void PolicyController(st_interface<float> &inference_data) { //
    // Euler angle
    /*bool ok = inference_net->InferenceOnceErax(*/
    /*    clock_input_vec, commands, dof_pos_obs, dof_vel_obs, last_action,*/
    /*    base_ang_vel, eu_ang);*/
    // Projected gravity
    std::vector<float> clock_input_vec =
        std::vector<float>(inference_data.input.clock_input_vec.data(),
                           inference_data.input.clock_input_vec.data() +
                               inference_data.input.clock_input_vec.size());
    std::vector<float> commands =
        std::vector<float>(inference_data.input.commands.data(),
                           inference_data.input.commands.data() +
                               inference_data.input.commands.size());
    std::vector<float> dof_pos_obs =
        std::vector<float>(inference_data.input.dof_pos_obs.data(),
                           inference_data.input.dof_pos_obs.data() +
                               inference_data.input.dof_pos_obs.size());
    std::vector<float> dof_vel_obs =
        std::vector<float>(inference_data.input.dof_vel_obs.data(),
                           inference_data.input.dof_vel_obs.data() +
                               inference_data.input.dof_vel_obs.size());
    std::vector<float> last_action =
        std::vector<float>(inference_data.input.last_action.data(),
                           inference_data.input.last_action.data() +
                               inference_data.input.last_action.size());
    std::vector<float> base_ang_vel =
        std::vector<float>(inference_data.input.base_ang_vel.data(),
                           inference_data.input.base_ang_vel.data() +
                               inference_data.input.base_ang_vel.size());
    std::vector<float> projected_gravity =
        std::vector<float>(inference_data.input.projected_gravity.data(),
                           inference_data.input.projected_gravity.data() +
                               inference_data.input.projected_gravity.size());

    bool ok = this->inference_net->InferenceOnceErax(
        clock_input_vec, commands, dof_pos_obs, dof_vel_obs, last_action,
        base_ang_vel, projected_gravity);

    if (!ok) {
      std::cout << "inference failed" << std::endl;
    }

    std::vector<float> last_action_output = {0};
    last_action.resize(12);
    std::vector<float> action_output = {0};
    action_output.resize(12);

    // get inference result when inference finished
    if (auto inference_status = this->inference_net->GetStatus();
        inference_status == float_inference_net::StatusT::FINISHED) {
      auto ok = this->inference_net->GetInfereceResult(last_action_output,
                                                       action_output);
      // TODO: Reset order
      {}
      if (!ok) {
        std::cout << "get inference result failed" << std::endl;
      }
      // control_periods = 0; // reset control periods when new target is
      // generated
    }

    for (size_t i = 0; i < 12; i++) {
      inference_data.input.last_action[i] = last_action_output[i];
      inference_data.Outputdata.action_output[i] = action_output[i];
    }

    if (inference_data.config.run_ctrl_cnt >
        inference_data.config.start_delay_cnt) {
      // TODO: Optimize this
      for (size_t i = 0; i < 12; i++) {
        inference_data.Outputdata.joint_target_position[i] =
            inference_data.Outputdata.action_output[i];
      }
      for (size_t i = 12; i < 20; i++) {
        inference_data.Outputdata.joint_target_position[i] = 0.0;
      }
    } else {
      for (int i = 0; i < 20; i++)
        inference_data.Outputdata.joint_target_position[i] = 0.0;
    }

    // control_periods += 1;
    // control_periods = (control_periods > policy_frequency) ? policy_frequency
    //                                                        : control_periods;

    // end_time = clock();
    // if (((float)(end_time - start_time) / CLOCKS_PER_SEC) > 0.001) {
    //   std::cout << "Calling time: "
    //             << (float)(end_time - start_time) / CLOCKS_PER_SEC << "s"
    //             << std::endl;
    // }

    // TorqueController();  // 计算力矩，放外面
    inference_data.config.run_ctrl_cnt++;
  }

  void GetJointObservation(sovle_st<float> &raw_data,
                           st_interface<float> &inference_data) {
    for (size_t i = 0; i < 20; i++) {
      if (i == 4 || i == 5 || i == 10 || i == 11) {
        continue;
      } else {
        raw_data.joint_current_position[i] = raw_data.motor_current_position[i];
        raw_data.joint_current_velocity[i] = raw_data.motor_current_velocity[i];
      }
    }
    auto left_res = this->left_ankle.ForwardKinematics(
        raw_data.motor_current_position[4], raw_data.motor_current_position[5]);
    auto right_res = this->right_ankle.ForwardKinematics(
        raw_data.motor_current_position[11],
        raw_data.motor_current_position[10]);

    auto left_vel_res = this->left_ankle.VelocityMapping(
        raw_data.motor_current_velocity[4], raw_data.motor_current_velocity[5]);
    auto right_vel_res =
        this->right_ankle.VelocityMapping(raw_data.motor_current_velocity[11],
                                          raw_data.motor_current_velocity[10]);
    raw_data.joint_current_position[4] = left_res(0, 0);
    raw_data.joint_current_position[5] = left_res(1, 0);
    raw_data.joint_current_position[10] = right_res(0, 0);
    raw_data.joint_current_position[11] = right_res(1, 0);
    raw_data.joint_current_velocity[4] = left_vel_res(0, 0);
    raw_data.joint_current_velocity[5] = left_vel_res(1, 0);
    raw_data.joint_current_velocity[10] = right_vel_res(0, 0);
    raw_data.joint_current_velocity[11] = right_vel_res(1, 0);
    /// 位置返回是pitch-roll没错，速度还得在确定一下。这块解算再看看源码和论文。
    // TODO: Optimize this
    for (size_t i = 0; i < 12; i++) {
      inference_data.input.dof_pos_obs[i] = raw_data.joint_current_position[i];
      inference_data.input.dof_vel_obs[i] = raw_data.joint_current_velocity[i];
    }
  }
  // joint pk control
  void TorqueController() { // 计算力矩放到外面了
    // Logger_More.startLog();

    // // Compute torque for non-wheel joints.
    // for (int i = 0; i < 2; i++)
    //   for (int j = 0; j < 7; j++) {
    //     float position_error = joint_target_position[i][j] -
    //                            joint_current_position[i][j]; // position
    //                            error
    //     /*joint_current_velocity[i][j] = velo_filters[i][j](*/
    //     /*    joint_current_velocity[i][j]);  // filter the velocity*/

    //     joint_target_torque[i][j] = p_gains[i][j] * position_error; // P
    //     joint_target_torque[i][j] +=
    //         -d_gains[i][j] * joint_current_velocity[i][j]; // D

    //     // Torque compensation is zero...
    //     if (abs(joint_target_position[i][j] - joint_current_position[i][j]) >
    //         compensate_threshold[i][j] * deg2rad) {
    //       /*std::cout << "Compensate!!!" << std::endl;*/
    //       joint_target_torque[i][j] +=
    //           torque_compensate[i][j] *
    //           (joint_target_position[i][j] - joint_current_position[i][j]) /
    //           abs((joint_target_position[i][j] -
    //           joint_current_position[i][j]));
    //     }
    //   }

    // log_result(0);

    // SetJointAction(); // 在外面搞
    // SetJointTorque_User(); // bitbot框架的关节力矩下发，这里不用
  }

  void SetJointAction(st_interface<T> &inference_data) {
    // For normal joints
    for (size_t i = 0; i < 20; i++) {
      if (i == 4 || i == 5 || i == 10 || i == 11) {
        continue;
      } else {
        inference_data.Outputdata.motor_target_position[i] =
            inference_data.Outputdata.joint_target_position[i];
        inference_data.Outputdata.motor_target_torque[i] =
            inference_data.Outputdata.joint_target_torque[i];
      }
    }

    // For ankle joints
    auto left_mot_pos_res = this->left_ankle.InverseKinematics(
        inference_data.Outputdata.joint_target_position[4],
        inference_data.Outputdata.joint_target_position[5]);
    auto right_mot_pos_res = this->right_ankle.InverseKinematics(
        inference_data.Outputdata.joint_target_position[10],
        inference_data.Outputdata.joint_target_position[11]);
    inference_data.Outputdata.motor_target_position[4] = -left_mot_pos_res[0];
    inference_data.Outputdata.motor_target_position[5] = -left_mot_pos_res[1];
    inference_data.Outputdata.motor_target_position[10] = -right_mot_pos_res[1];
    inference_data.Outputdata.motor_target_position[11] = -right_mot_pos_res[0];

    auto left_tor_res = this->left_ankle.TorqueRemapping(
        inference_data.Outputdata.joint_target_torque[4],
        inference_data.Outputdata.joint_target_torque[5]);
    auto right_tor_res = this->right_ankle.TorqueRemapping(
        inference_data.Outputdata.joint_target_torque[10],
        inference_data.Outputdata.joint_target_torque[11]);
    inference_data.Outputdata.motor_target_torque[4] = -left_tor_res[0];
    inference_data.Outputdata.motor_target_torque[5] = -left_tor_res[1];
    inference_data.Outputdata.motor_target_torque[10] = -right_tor_res[1];
    inference_data.Outputdata.motor_target_torque[11] = -right_tor_res[0];

    // 差分算目标速度
    for (size_t i = 0; i < 20; i++) {
      float diff = inference_data.Outputdata.motor_target_position[i] -
                   inference_data.Outputdata.motor_target_position_last[i];
      inference_data.Outputdata.motor_target_velocity[i] =
          diff / inference_data.config
                     .dt; // TODO: 这里的dt是0.0025，和policy controller里的一样
    }
  }
  // give joint action to interface: 还没写
  // void GiveJointAction2Interface() {}
};

#endif