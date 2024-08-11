import sys

# 添加特定路径到 Python 解释器的搜索路径中
sys.path.append('/workspace/change_task')
import os.path

import cv2
import torch.optim
import argparse
import json

from skimage import measure

from model.model_encoder_att import Encoder, AttentiveEncoder
from model.model_decoder import DecoderTransformer
from utils_tool.utils import *
from imageio.v2 import imread
from PIL import Image
import numpy as np
from torchvision import transforms


# compute_change_map(path_A, path_B)函数: 生成一个掩膜mask用来表示两个图像之间的变化区域
'''
Args:
    path_A: 图像A的路径
    path_B: 图像B的路径
Returns:
    change_map: 变化区域的掩膜
'''
# def compute_change_mask(path_A, path_B):
#     import cv2
#     import numpy as np
#     img_A = cv2.imread(path_A)
#     img_B = cv2.imread(path_B)
#     change_map = (img_B-img_A).astype(np.uint8)
#     # 阈值化
#     change_map = cv2.cvtColor(change_map, cv2.COLOR_BGR2GRAY)
#     change_map = cv2.threshold(change_map, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
#     cv2.imwrite('E:\change_map.png', change_map)
#     return 'I have save the changed mask in E:\change_map.png'

# compute_change_caption(path_A, path_B)函数：生成一个文本用于描述两个图像之间变化
'''
Args:
    path_A: 图像A的路径
    path_B: 图像B的路径
Returns:
    caption: 变化描述文本
'''
class Change_Perception(object):
    def define_args(self):


        script_path = os.path.abspath(__file__)
        script_dir = os.path.dirname(script_path)
        print(script_dir)
        parser = argparse.ArgumentParser(description='Remote_Sensing_Image_Change_Interpretation')

        parser.add_argument('--data_folder', default='/home/wangzexin/model/Change_Agent/Levir-MCI-dataset/images',
                            help='folder with data files')
        parser.add_argument('--list_path', default='./',
                            help='path of the data lists')
        parser.add_argument('--vocab_file', default='vocab', help='path of the data lists')
        parser.add_argument('--max_length', type=int, default=41, help='path of the data lists')

        # inference
        parser.add_argument('--gpu_id', type=int, default=0, help='gpu id in the training.')
        parser.add_argument('--checkpoint', default='./my_train.pth',help='path to checkpoint')
        parser.add_argument('--result_path', default="./predict_result/",
                            help='path to save the result of masks and captions')

        # backbone parameters
        parser.add_argument('--network', default='segformer-mit_b1',
                            help='define the backbone encoder to extract features')
        parser.add_argument('--encoder_dim', type=int, default=512,
                            help='the dimension of extracted features using backbone ')
        parser.add_argument('--feat_size', type=int, default=16,
                            help='define the output size of encoder to extract features')
        parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
        # Model parameters
        parser.add_argument('--n_heads', type=int, default=8, help='Multi-head attention in Transformer.')
        parser.add_argument('--n_layers', type=int, default=3, help='Number of layers in AttentionEncoder.')
        parser.add_argument('--decoder_n_layers', type=int, default=1)
        parser.add_argument('--feature_dim', type=int, default=512, help='embedding dimension')

        args = parser.parse_args()

        return args

    def __init__(self,):
        """
        Training and validation.
        """
        # args = self.define_args()
        vocab_file_path = os.path.join(os.path.dirname(__file__), 'vocab.json')
        snapshot_full_path =  os.path.join(os.path.dirname(__file__), "my_train.pth")
        self.mean = [0.39073 * 255, 0.38623 * 255, 0.32989 * 255]
        self.std = [0.15329 * 255, 0.14628 * 255, 0.13648 * 255]
        self.position_dict = {
            "0-0":"左上角",
            "1-0":"上方",
            "2-0":"上方",
            "3-0":"右上角",
            "0-1":"中部靠左",
            "1-1":"中部靠左",
            "2-1":"中部靠右",
            "3-1":"中部靠右",
            "0-2":"中部靠左",
            "1-2":"中部靠左",
            "2-2":"中部靠右",
            "3-2":"中部靠右",
            "0-3":"左下角",
            "1-3":"下方",
            "2-3":"下方",
            "3-3":"右下方",
        }

        with open(vocab_file_path, 'r') as f:
            self.word_vocab = json.load(f)
        # Load checkpoint
        # snapshot_full_path = args.checkpoint

        checkpoint = torch.load(snapshot_full_path)

        # self.encoder = Encoder(args.network)
        self.encoder = Encoder("segformer-mit_b1")
        self.encoder_trans = AttentiveEncoder(train_stage=None, n_layers=3,
                                         feature_size=[16, 16, 512],
                                         heads=8, dropout=0.1)
        self.decoder = DecoderTransformer(encoder_dim=512, feature_dim=512,
                                     vocab_size=len(self.word_vocab), max_lengths=41,
                                     word_vocab=self.word_vocab, n_head=8,
                                     n_layers=1, dropout=0.1)

        self.encoder.load_state_dict(checkpoint['encoder_dict'])
        self.encoder_trans.load_state_dict(checkpoint['encoder_trans_dict'], strict=False)
        self.decoder.load_state_dict(checkpoint['decoder_dict'])
        # Move to GPU, if available
        self.encoder.eval()
        self.encoder = self.encoder.cuda()
        self.encoder_trans.eval()
        self.encoder_trans = self.encoder_trans.cuda()
        self.decoder.eval()
        self.decoder = self.decoder.cuda()

    def split_image(self,path):
        img = Image.open(path)
        # 获得图片的宽度和高度
        width, height = img.size
        # 计算可以分割的小块数量
        num_blocks_width = width // 256
        num_blocks_height = height // 256
        # 存储每个小块的list
        sub_images = []
        positions = []
        # 遍历图像并分割
        for i in range(num_blocks_height):
            for j in range(num_blocks_width):
                # 计算左上角坐标
                left = j * 256
                top = i * 256
                # 裁剪图像
                sub_image = img.crop((left, top, left + 256, top + 256))
                # 存储小块的位置信息
                position_str = str(j) + "-" + str(i)
                position_describe_str = self.position_dict[position_str]
                positions.append(position_describe_str)

                # 存储小块的像素信息，进行预处理
                sub_images.append(sub_image)
                # sub_image.save("/home/wangzexin/model/change_agent/Multi_change/test_save/read"+ position_str +".png")
                

        return positions,sub_images


    def preprocess(self, img_a_original, img_b_original):

        # imgA = Image.open(path_A)
        # imgB = Image.open(path_B)
        # resize_transform = transforms.Resize((256, 256))
        # imgA = resize_transform(imgA)
        # imgB = resize_transform(imgB)
        # imgA.save("/home/wangzexin/model/change_agent/Multi_change/test_save/read0.png")
        # imgB.save("/home/wangzexin/model/change_agent/Multi_change/test_save/read1.png")

        imgA = np.asarray(img_a_original, np.float32)
        imgB = np.asarray(img_b_original, np.float32)

        imgA = imgA.transpose(2, 0, 1)
        imgB = imgB.transpose(2, 0, 1)
        for i in range(len(self.mean)):
            imgA[i, :, :] -= self.mean[i]
            imgA[i, :, :] /= self.std[i]
            imgB[i, :, :] -= self.mean[i]
            imgB[i, :, :] /= self.std[i]

        # print(type(imgA))
        if imgA.shape[1] != 256 or imgA.shape[2] != 256:
            imgA = cv2.resize(imgA, (256, 256))
            imgB = cv2.resize(imgB, (256, 256))

        imgA = torch.FloatTensor(imgA)
        imgB = torch.FloatTensor(imgB)
        imgA = imgA.unsqueeze(0)  # (1, 3, 256, 256)
        imgB = imgB.unsqueeze(0)

        return imgA, imgB

    def generate_change_caption(self, img_a_original, img_b_original):
        # print('model_infer_change_captioning: start')
        
        imgA, imgB = self.preprocess(img_a_original, img_b_original)
        # Move to GPU, if available
        imgA = imgA.cuda()
        imgB = imgB.cuda()
        feat1, feat2 = self.encoder(imgA, imgB)
        feat1, feat2, seg_pre = self.encoder_trans(feat1, feat2)
        seq = self.decoder.sample(feat1, feat2, k=1)
        pred_seq = [w for w in seq if w not in {self.word_vocab['<START>'], self.word_vocab['<END>'], self.word_vocab['<NULL>']}]
        pred_caption = ""
        for i in pred_seq:
            pred_caption += (list(self.word_vocab.keys())[i]) + " "

        caption ='图中有一栋建筑被移除'
        caption = pred_caption

        return caption

    def change_detection(self, path_A, path_B, savepath_mask):
        # print('model_infer_change_detection: start')
        imgA, imgB = self.preprocess(path_A, path_B)
        # Move to GPU, if available
        imgA = imgA.cuda()
        imgB = imgB.cuda()
        feat1, feat2 = self.encoder(imgA, imgB)
        feat1, feat2, seg_pre = self.encoder_trans(feat1, feat2)
        # for segmentation
        pred_seg = seg_pre.data.cpu().numpy()
        pred_seg = np.argmax(pred_seg, axis=1)
        # 保存图片
        pred = pred_seg[0].astype(np.uint8)
        pred_rgb = np.zeros((pred.shape[0], pred.shape[1], 3), dtype=np.uint8)
        pred_rgb[pred == 1] = [0, 255, 255]
        pred_rgb[pred == 2] = [0, 0, 255]

        cv2.imwrite(savepath_mask, pred_rgb)
        print('model_infer: mask saved in', savepath_mask)

        print('model_infer_change_detection: end')
        return pred # (256,256,3)
        # return 'change detection successfully. '

    def compute_object_num(self, changed_mask, object):
        print("compute num start")
        # compute the number of connected components
        mask = changed_mask
        mask_cp = 0 * mask.copy()
        if object == 'road':
            mask_cp[mask == 1] = 255
        elif object == 'building':
            mask_cp[mask == 2] = 255
        lbl = measure.label(mask_cp, connectivity=2)
        props = measure.regionprops(lbl)
        # get bboxes by a for loop
        bboxes = []
        for prop in props:
            # print('Found bbox', prop.bbox, 'area:', prop.area)
            if prop.area > 5:
                bboxes.append([prop.bbox[1], prop.bbox[0], prop.bbox[3], prop.bbox[2]])
        num = len(bboxes)
        # visual
        # mask_array_copy = mask.copy()*255
        # for bbox in bboxes:
        #     print('Found bbox', bbox)
        #     cv2.rectangle(mask_array_copy, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255), 2)
        # cv2.namedWindow('findCorners', 0)
        # cv2.resizeWindow('findCorners', 600, 600)
        # cv2.imshow('findCorners', mask_array_copy)
        # cv2.waitKey(0)
        print('Found', num, object)
        print('compute num end')
        # return
        num_str = 'Found ' + str(num) + ' changed ' + object
        return num_str

    def get_change_position_caption_list(self,imgA_path, imgB_path):
        
        negative_words = ["same as before","no difference","seem identical","no change","nothing has changed"]
        change_flag = True
        res = {}
        positions_a,sub_images_a = self.split_image(imgA_path)
        positions_b,sub_images_b = self.split_image(imgB_path)

        for index in range(len(sub_images_a)):
            change_flag = True
            img_a_original = sub_images_a[index]
            img_b_original = sub_images_b[index]

            change_caption = self.generate_change_caption(img_a_original, img_b_original)

            # 查看是否有改变
            for negative_word in negative_words:
                if negative_word in change_caption:
                    change_flag = False
            
            # 假若存在改变，保存位置和对应caption
            if change_flag:
                # res_positions.append(positions_a[index])
                # res_captions.append(change_caption)
                res[positions_a[index]] = change_caption

        return res


    # design more tool functions:


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Remote_Sensing_Image_Change_Interpretation')
    parser.add_argument('--imgA_path', default=r'/home/wangzexin/model/MiniCPM-V/input_path/Change_caption/image2/0.png')
    parser.add_argument('--imgB_path', default=r'/home/wangzexin/model/MiniCPM-V/input_path/Change_caption/image1/0.png')
    # parser.add_argument('--mask_save_path', default=r'/home/wangzexin/model/change_agent/Multi_change/test_save/test1.png')

    args = parser.parse_args()

    imgA_path = args.imgA_path
    imgB_path = args.imgB_path

    change_perception = Change_Perception()
    pos_caption_map = change_perception.get_change_position_caption_list(imgA_path, imgB_path)
    print(pos_caption_map)

    # for position in res_positions:
    #     print(position)
    
    # for caption in res_captions:
    #     print(caption)

    


        # Change_Perception.change_detection(imgA_path, imgB_path, args.mask_save_path)
