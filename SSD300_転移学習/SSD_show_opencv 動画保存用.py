#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2  # OpenCVライブラリ
import matplotlib.pyplot as plt 
import numpy as np
import torch

#get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


from utils.ssd_model import DataTransform


# ## 学習済みモデルの読み込み

# In[3]:


from utils.ssd_model import SSD

voc_classes = ['safe', 'caution']

# SSD300の設定
ssd_cfg = {
    'num_classes': 3,  # 背景クラスを含めた合計クラス数
    'input_size': 300,  # 画像の入力サイズ
    'bbox_aspect_num': [4, 6, 6, 6, 4, 4],  # 出力するDBoxのアスペクト比の種類
    'feature_maps': [38, 19, 10, 5, 3, 1],  # 各sourceの画像サイズ
    'steps': [8, 16, 32, 64, 100, 300],  # DBOXの大きさを決める
    'min_sizes': [30, 60, 111, 162, 213, 264],  # DBOXの大きさを決める
    'max_sizes': [60, 111, 162, 213, 264, 315],  # DBOXの大きさを決める
    'aspect_ratios': [[2], [2, 3], [2, 3], [2, 3], [2], [2]],
}

# SSDネットワークモデル
net = SSD(phase="inference", cfg=ssd_cfg)




# SSDの学習済みの重みを設定(ここは場合によって変更)
net_weights = torch.load('./weights/fine_tuning_32batch/ssd300_fine260.pth',
                         map_location={'cuda:0': 'cpu'})

#net_weights = torch.load('./weights/ssd300_mAP_77.43_v2.pth',
#                         map_location={'cuda:0': 'cpu'})

net.load_state_dict(net_weights)

print('ネットワーク設定完了：学習済みの重みをロードしました')


# ## 予測と表示を行うクラスを作成

# In[4]:


class SSDPredictShow():
    """SSDでの予測と画像の表示をまとめて行うクラス"""

    def __init__(self, eval_categories, net):
        self.eval_categories = eval_categories  # クラス名
        self.net = net  # SSDネットワーク

        color_mean = (104, 117, 123)  # (BGR)の色の平均値
        input_size = 300  # 画像のinputサイズを300×300にする
        self.transform = DataTransform(input_size, color_mean)  # 前処理クラス

    def show(self, frame, data_confidence_level):
        """
        物体検出の予測結果を表示をする関数。

        Parameters
        ----------
        image_file_path:  str
            画像のファイルパス
        data_confidence_level: float
            予測で発見とする確信度の閾値

        Returns
        -------
        なし。rgb_imgに物体検出結果が加わった画像が表示される。
        """
        #SSDで予測させ(予測BBox, 予測ラベル, 確信度)を返す
        frame, predict_bbox, pre_dict_label_index, scores = self.ssd_predict(frame, data_confidence_level) 
        #予測結果を表示
        self.vis_bbox(frame, bbox=predict_bbox, label_index=pre_dict_label_index,
                      scores=scores, label_names=self.eval_categories)

    #def ssd_predict(self, image_file_path, data_confidence_level=0.5):
    def ssd_predict(self, frame, data_confidence_level=0.3):
        
        """
        SSDで予測させる関数。

        Parameters
        ----------
        image_file_path:  strt
            画像のファイルパス

        dataconfidence_level: float
            予測で発見とする確信度の閾値

        Returns
        -------
        rgb_img, true_bbox, true_label_index, predict_bbox, pre_dict_label_index, scores
        """

        # rgbの画像データを取得
        #img = frame
        #img = cv2.imread(image_file_path)  # [高さ][幅][色BGR]
        height, width, channels = frame.shape  # 画像のサイズを取得
        #rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)#BGR→RGB

        # 画像の前処理
        phase = "val"
        img_transformed, boxes, labels = self.transform(
            frame, phase, "", "")  # アノテーションが存在しないので""にする。
        img = torch.from_numpy(
            img_transformed[:, :, (2, 1, 0)]).permute(2, 0, 1)

        # SSDで予測
        self.net.eval()  # ネットワークを推論モードへ
        x = img.unsqueeze(0)  # ミニバッチ化：torch.Size([1, 3, 300, 300])

        detections = self.net(x)#予測結果torch.Size([batch_num, クラス数, 200, 5])

        # detectionsの形は、torch.Size([1, 21, 200, 5])  ※200はtop_kの値

        
        
        # confidence_levelが基準以上を取り出す
        predict_bbox = []
        pre_dict_label_index = []
        scores = []
        detections = detections.cpu().detach().numpy()#予測結果をnumpy形式に

        # 条件以上の値を抽出
        find_index = np.where(detections[:, 0:, :, 0] >= data_confidence_level)#確信度が(表示用の)閾値より大きいところはTrue(0:も0から最後まで)
        detections = detections[find_index]
        for i in range(len(find_index[1])):#find_index[1]は第一次元の数(クラス数)
            # 抽出した物体数分ループを回す
            if (find_index[1][i]) > 0:  #find_index[1][i]はクラスの番号
                # 背景クラスでないもの
                sc = detections[i][0]  # 確信度
                bbox = detections[i][1:] * [width, height, width, height]#DBOXの規格化を解除してframeの大きさに合わせる
                # find_indexはミニバッチ数、クラス、topのtuple
                lable_ind = find_index[1][i]-1
                # （注釈）
                # 背景クラスが0なので1を引く

                # 返り値のリストに追加
                predict_bbox.append(bbox)
                pre_dict_label_index.append(lable_ind)
                scores.append(sc)

        return  frame, predict_bbox, pre_dict_label_index, scores

    def vis_bbox(self, img, bbox, label_index, scores, label_names):
        """
        物体検出の予測結果を画像で表示させる関数。

        Parameters
        ----------
        rgb_img:rgbの画像
            対象の画像データ
        bbox: list
            物体のBBoxのリスト
        label_index: list
            物体のラベルへのインデックス
        scores: list
            物体の確信度。
        label_names: list
            ラベル名の配列

        Returns
        -------
        なし。rgb_imgに物体検出結果が加わった画像が表示される。
        """

        # 枠の色の設定
        num_classes = len(label_names)  # クラス数（背景のぞく）
        #colors = plt.cm.hsv(np.linspace(0, 1, num_classes)).tolist()

        # 画像の表示
        #plt.figure(figsize=(10, 10))
        #plt.imshow(rgb_img)
        #currentAxis = plt.gca()

        # BBox分のループ
        for i, bb in enumerate(bbox):

            # 取り出したBBoxに対応するラベル名
            label_name = label_names[label_index[i]]
            #color = colors[label_index[i]]  # クラスごとに別の色の枠を与える

            # 枠につけるラベル　例：person;0.72　
            if scores is not None:
                sc = scores[i]
                display_txt = '%s: %.2f' % (label_name, sc)
            else:
                display_txt = '%s: ans' % (label_name)
            
            color = (0, 0, 0)
            #色の設定
            if label_index[i] == 1:
                color = (0, 0, 255)
            if label_index[i] == 0:
                color = (0, 255, 0)

            # 枠の座標
            xy = (bb[0], bb[1])
            width = bb[2] - bb[0]
            height = bb[3] - bb[1]
            #bb[0]=xmin, bb[1]=ymin, bb[2]=xmax, bb[3]=ymax

            # 長方形を描画する
            #currentAxis.add_patch(plt.Rectangle(
            #   xy, width, height, fill=False, edgecolor=color, linewidth=2))

            
            #opevcvで長方形を描画
            cv2.rectangle(img, (int(bb[0]), int(bb[1])), (int(bb[2]), int(bb[3])), color, 2)

            # 長方形の枠の左上にラベルを描画する
            #currentAxis.text(xy[0], xy[1], display_txt, bbox={
             #                'facecolor': color, 'alpha': 0.5})
                
            # opencvで長方形の枠の左上にラベルを描画する
            font = cv2.FONT_HERSHEY_PLAIN#フォントの設定
            cv2.putText(img, display_txt, (int(bb[0]),int(bb[1])), font, 1, color,1, lineType=cv2.LINE_AA)#(文字, 座標, フォント, (太さ), 色, 線を綺麗に)


# In[5]:


#video_path ='C:\\Users\\yuki\\Desktop\\うつ伏せ発見プログラム\\data\\nwcam\\original\\val_data\\10588047_200122140112_L.mp4'


# In[6]:


#動画の読み込み
video_path = input('動画のパスを入力してください>')
cap = cv2.VideoCapture(video_path)#動画の読み込み




# In[8]:


#動画の読み込み
#video_path = input('動画のパスを入力してください>')
cap = cv2.VideoCapture(video_path)#動画の読み込み


if cap.isOpened()== False:#動画が読み込めたか
    sys.exit() #プログラムを終了
#FPS調整
#import time
#wait_time = 0.1

#動画保存の設定
ret, frame = cap.read()
h, w = frame.shape[:2]#0と1（２の一つ前まで)
fourcc = cv2.VideoWriter_fourcc(*"XVID")
import os
basename = os.path.basename(video_path)

dst = cv2.VideoWriter("output/checked_"+basename, fourcc, 1.0, (w,h))#(出力先　書き込み設定　FPS 解像度)

while True:

    ret, frame = cap.read() #動画は１フレームの読み込みと表示の繰り返し
    if ret == False:
        break
    #予測(SSDの出力)
    ssd = SSDPredictShow(eval_categories=voc_classes, net=net)
    # output : torch.Size([batch_num, 21, 200, 5])  (今回1枚だけなので、batch_num=1)
#  =（batch_num、クラス、confのtop200、規格化されたBBoxの情報）
#   規格化されたBBoxの情報（確信度、xmin, ymin, xmax, ymax）

    #予測結果を含んだframeを返す
    ssd.show(frame, data_confidence_level=0.3)#確信度の下限をいじると変わる
    
    cv2.imshow("video", frame)

    dst.write(frame)  # 指定した設定で書き込み

    if cv2.waitKey(30) == 27 :#30mms(FPS30より)待つ間に27(Escキー)が押されると動画からbreak
        break
cv2.destroyAllWindows()
cap.release


# In[ ]:





# In[ ]:





# In[ ]:




