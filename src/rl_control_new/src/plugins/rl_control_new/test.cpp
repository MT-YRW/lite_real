#include <Eigen/Core>
#include <Eigen/Dense>
#include <bodyctrl_msgs/CmdMotorCtrl.h>
#include <iostream>
#include <nodelet/nodelet.h>
#include <pluginlib/class_list_macros>
#include <stdio.h>
#include "util/LockFreeQueue.h"

namespace rl_control_test {
class test : public nodelet::Nodelet {
public:
  test() {}

private:
  virtual void onInit() {
    // Initialize the nodelet
    NODELET_INFO("test nodelet initialized");
    ros::NodeHandle &nh = getPrivateNodeHandle();

    pi = 3.14159265358979;
    rpm2rps = pi / 30.0;
    motor_num = 20;

    Q_a = Eigen::VectorXd::Zero(motor_num);
    Qdot_a = Eigen::VectorXd::Zero(motor_num);
    Tor_a = Eigen::VectorXd::Zero(motor_num);
    Q_d = Eigen::VectorXd::Zero(motor_num);
    Qdot_d = Eigen::VectorXd::Zero(motor_num);
    Tor_d = Eigen::VectorXd::Zero(motor_num);

    q_a = Eigen::VectorXd::Zero(motor_num);
    qdot_a = Eigen::VectorXd::Zero(motor_num);
    tor_a = Eigen::VectorXd::Zero(motor_num);
    q_d = Eigen::VectorXd::Zero(motor_num);
    qdot_d = Eigen::VectorXd::Zero(motor_num);
    tor_d = Eigen::VectorXd::Zero(motor_num);

    // 参数初始化
    ct_scale = Eigen::VectorXd::Ones(motor_num);
    ct_scale << 2.5, 2.1, 2.5, 2.5, 1.4, 1.4, 2.5, 2.1, 2.5, 2.5, 1.4, 1.4, 1.4,
        1.4, 1.4, 1.4, 1.4, 1.4, 1.4, 1.4; // 电机转矩系数
    zero_pos = Eigen::VectorXd::Zero(motor_num);
    std::stringstream ss;
    for (int32_t i = 0; i < motor_num; ++i) {
      zero_pos[i] = GetConfig("zero_pos_" + std::to_string(i), 0.0);
      ss << zero_pos[i] << "  ";
    }

    pubSetMotorCmd = nh.advertise<bodyctrl_msgs::CmdMotorCtrl>(
        "/BodyControl/motor_ctrl", 1000);
    subState = nh.subscribe("/BodyControl/motor_state", 1000,
                            &RLControlNewPlugin::OnMotorStatusMsg, this);
    subImuXsens = nh.subscribe("/BodyControl/imu", 1000,
                               &RLControlNewPlugin::OnXsensImuStatusMsg, this);
    subJoyCmd = nh.subscribe<sensor_msgs::Joy>(
        "/sbus_data", 1000, &RLControlNewPlugin::xbox_map_read, this);
    subCmdVel = nh.subscribe<geometry_msgs::Twist>(
        "/cmd_vel", 1000, &RLControlNewPlugin::OnCmdVelMsg, this);

    sleep(1); // 这个是为了干啥

    std::thread([this]() { test_loop(); }).detach();
  }

  void test_loop() {
    ros::Rate rate(1000); // 为什么天工的代码里没有用roa::rate控制线程频率？

    // set sched-strategy  // 这里设置线程优先级的时候和ros有关系吗？
    struct sched_param sched;
    int max_priority;

    max_priority = sched_get_priority_max(SCHED_RR); // 获取系统调度最高优先级
    sched.sched_priority = max_priority; // 设置当前线程的调度策略和优先级

    if (sched_setscheduler(gettid(), SCHED_RR, &sched) == -1) {
      printf("Set Scheduler Param, ERROR:%s\n", strerror(errno));
    }
    usleep(1000);
    printf("set scheduler success\n");

    while (!queueMotorState.empty()) { // 从消息队列中读取电机状态
      auto msg = queueMotorState.pop();
      // set motor buf
      int id = 0;
      for (auto &one : msg->status) {
        id = motor_id[one.name];
        Q_a(id) = one.pos;
        Qdot_a(id) = one.speed;
        Tor_a(id) = one.current * ct_scale(id);
      }
    }
    for (int i = 0; i < motor_num; i++) {
      q_a(i) = (Q_a(i) - zero_pos(i)) * motor_dir(i) +
               zero_offset(
                   i); // 将原始编码器值转换为统一坐标系下的关节角度（弧度制）
      zero_cnt(i) = (q_a(i) > pi) ? -1.0 : zero_cnt(i);
      zero_cnt(i) = (q_a(i) < -pi) ? 1.0 : zero_cnt(i);
      q_a(i) += zero_cnt(i) * 2.0 * pi;
    }
    std::cout << "current Q_A pos: " << Q_a.transpose() << std::endl;
    std::cout << "current pos: " << q_a.transpose() << std::endl;
    std::cout << "enter 1: " << std::endl;
    double a;
    std::cin >> a;

    // Test loop code
    while (ros::ok()) {
      // Your test logic here
      std::cout << "Running test loop..." << std::endl;
      ros::Duration(1.0).sleep(); // Sleep for 1 second
    }
  }

  void OnMotorStatusMsg(const bodyctrl_msgs::MotorStatusMsg::ConstPtr &msg) {
    auto wrapper = msg;
    queueMotorState.push(wrapper);
  }

  void OnImuStatusMsg(const bodyctrl_msgs::Imu::ConstPtr &msg) {
    auto wrapper = msg;
    queueImuRm.push(wrapper);
  }

  void OnXsensImuStatusMsg(const bodyctrl_msgs::Imu::ConstPtr &msg) {
    auto wrapper = msg;
    queueImuXsens.push(wrapper);
  }

  void xbox_map_read(const sensor_msgs::Joy::ConstPtr &msg) {
    auto wrapper = msg;
    queueJoyCmd.push(wrapper);
  }

  void OnCmdVelMsg(const geometry_msgs::Twist::ConstPtr &msg) {
    auto wrapper = msg;
    queueCmdVel.push(wrapper);
  }

  // 电机当前状态
  Eigen::VectorXd Q_a;
  Eigen::VectorXd Qdot_a;
  Eigen::VectorXd Tor_a;
  Eigen::VectorXd q_a;
  Eigen::VectorXd qdot_a;
  Eigen::VectorXd tor_a;

  // 电机目标状态
  Eigen::VectorXd Q_d;
  Eigen::VectorXd Qdot_d;
  Eigen::VectorXd Tor_d;
  Eigen::VectorXd q_d;
  Eigen::VectorXd qdot_d;
  Eigen::VectorXd tor_d;

  // 电机默认参数
  Eigen::VectorXd ct_scale;
  Eigen::VectorXd zero_pos;
  Eigen::VectorXd zero_offset;
  Eigen::VectorXd zero_cnt;
  Eigen::VectorXd init_pos;
  Eigen::VectorXd motor_dir;

  // 电机控制参数
  Eigen::VectorXd kp;
  Eigen::VectorXd kd;

  // 其他参数
  int motor_num;
  std::map<int, int> motor_id, motor_name;
  float rpm2rps;
  float pi;
  std::string _config_file = "/home/ubuntu/data/param/rl_control_new.txt";
  std::unordered_map<std::string, std::string> _config_map;

  ros::Subscriber subState;
  // fast_ros::Subscriber subImu, subImuXsens;
  ros::Subscriber subImu, subImuXsens;
  ros::Publisher pubSetMotorCmd;
  ros::Subscriber subJoyCmd;
  ros::Subscriber subCmdVel;
  LockFreeQueue<bodyctrl_msgs::MotorStatusMsg::ConstPtr> queueMotorState;
  LockFreeQueue<bodyctrl_msgs::Imu::ConstPtr> queueImuRm;
  LockFreeQueue<bodyctrl_msgs::Imu::ConstPtr> queueImuXsens;
  LockFreeQueue<sensor_msgs::Joy::ConstPtr> queueJoyCmd;
  LockFreeQueue<geometry_msgs::Twist::ConstPtr> queueCmdVel;
};

} // namespace rl_control_test
PLUGINLIB_EXPORT_CLASS(rl_control_test::test, nodelet::Nodelet);