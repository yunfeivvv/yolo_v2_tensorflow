# yolo_v2_tensorflow_practice

**现在初步有效果，但是表现不好，还需要继续调整!**  
**It works, but not good enough!**

todo:  

- loss计算不对：`loss.py`的184行，只是计算了对应位置的iou，实际应该是：计算所有pred bbox和所有gt bbox之间的iou。

