## 阴阳师自动识别点击程序
这是一个基于YOLOv5的阴阳师自动识别点击程序。通过模型训练，程序能够自动识别并点击游戏界面中的特定元素。

### 如何开始
首先，您需要安装所需的Python包。这可以通过运行以下命令完成：

```bash
pip install -r requirements.txt
```

完成设置后，您就可以启动程序了：

```bash
python main.py
```

### 类别说明
模型可以识别以下类别
*  0: jiacheng（加成）
*  1: gold（金币）
*  2: gouyu（勾玉）
*  3: tili（体力）
*  4: tansuo（探索）
*  5: tingzhong（町中）
*  6: zudui（组队）
*  7: yingyangliao（阴阳寮）
*  8: yinyangshu（阴阳术）
*  9: shishenlu（式神录）
*  10: liaotian（聊天）
*  11: tiaozhan_on（挑战金色）
*  12: dashe（大蛇）
*  13: win（胜利）
*  14: hun（魂）
*  15: yaoqing（邀请）
*  16: yes（是的）
*  17: tiaozhan_off（挑战灰色）
*  18: zhunbei（准备）
*  19: yizhunbei（已准备）
*  20: tongyi1（同意1）
*  21: tongyi2（同意2）
*  22: tupo_win（已突破）
*  23: tupo_loss（已突破但打不过）
*  24: tupo_new（未突破）
*  25: tupo_fail（突破失败）
*  26: shuaxin（刷新）
*  27: jingong（进攻）
*  28: fengmo_icon
*  29: shoulie_icon
*  30: daoguan_icon
*  31: yanhui_icon
*  32: xiajian_icon
*  33: fengmo
*  34: shoulie
*  35: daoguan
*  36: yanhui
*  37: xiajian
*  38: qianwang
*  39: tansuo_botton
*  40: attack
*  41: attack_head
*  42: egg_up
*  43: exp_up
*  44: gold_up
*  45: gift
*  46: gain_gift
*  47: back
*  48: instances_27