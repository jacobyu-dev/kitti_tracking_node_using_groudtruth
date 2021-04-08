# kitti_tracking_node_using_groudtruth
This ros package do multi-object tracking using kitti pointcloud ground truth data (tracklet.xml)

![Peek 2020-10-28 16-15](https://user-images.githubusercontent.com/59205405/113991659-a3273b00-988d-11eb-822e-c71de0f0b631.gif)

## 사전준비
- Kitti dataset rosbag file 
- jsk_rviz_plugins 설치
- jsk_recognition_msgs 설치
```
sudo apt-get install ros-melodic-jsk-recognition-msgs & sudo apt-get install ros-melodic-jsk-rviz-plugins
```
## 1.  kitti2bag Converter를 이용해 Rosbag파일 생성
Tomáš Krejčí created a simple tool for conversion of raw kitti datasets to ROS bag files:  [kitti2bag](https://github.com/tomas789/kitti2bag)

## 2. 실행방법
```
- 터미널 [1] : roscore
- 터미널 [2] : rosbag play -l xxx.bag
- 터미널 [3] : rrosrun mot_kf_tracking mot_ab3dmot_track_node.py
- 터미널 [4] : rviz
```
## 3. Rviz
- Fixed Frame을 velo_link로 바꾸어준다.

![image](https://user-images.githubusercontent.com/59205405/93186176-49d4ae80-f779-11ea-98ed-59ce06cc656d.png)

- Add 버튼에서 다음 디스플레이 요소들을 추가해준다.

![image](https://user-images.githubusercontent.com/59205405/93186544-b8197100-f779-11ea-9cfd-080de325f0e9.png)

- BoundingBoxArray, PictogramArray, Image, PointCloud2 의 topic을 rqt_graph를 보고 맞추어준다.

## 4. Rqt_graph
![image](https://user-images.githubusercontent.com/59205405/93187056-50175a80-f77a-11ea-9419-755a8d767e87.png)
![image](https://user-images.githubusercontent.com/59205405/113992227-33658000-988e-11eb-8df3-57e776df701c.png)


## 5. 결과
![image](https://user-images.githubusercontent.com/59205405/93187240-9076d880-f77a-11ea-99f6-4f23337b8e2d.png)
