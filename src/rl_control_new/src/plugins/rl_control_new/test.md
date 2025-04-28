# single joint test
1. 看一下/BodyControl/motor_ctrl、/BodyControl/motor_state这两个话题，rosbag录包
2. 新写一个程序，发布单关节运动的指令

# 2025.4.27
1. 路由器配置的时候有问题，工控机现在找不到任何wifi。而且rostopic list找不到话题，怀疑是网络的问题。明天重启把这个问题解决。
2. 看说明文档的时候发现力位混合控制下发的力矩是*前馈力矩*，应该都先给0？？明天上机要看一看天工自己下发的是多少。
3. 看一下下发的kp,kd是多少
