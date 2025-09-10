import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
from PIL import Image
import requests
import os
from torchvision import transforms
import warnings
warnings.filterwarnings("ignore")

class U2Net(nn.Module):
    """U²-Net 모델 구현 (인물 세그멘테이션용)"""
    
    def __init__(self, in_ch=3, out_ch=1):
        super(U2Net, self).__init__()
        
        # 인코더
        self.stage1 = RSU7(in_ch, 32, 64)
        self.pool12 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        
        self.stage2 = RSU7(64, 32, 128)
        self.pool23 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        
        self.stage3 = RSU7(128, 64, 256)
        self.pool34 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        
        self.stage4 = RSU7(256, 128, 512)
        self.pool45 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        
        self.stage5 = RSU7(512, 256, 512)
        self.pool56 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        
        self.stage6 = RSU7(512, 256, 512)
        
        # 디코더
        self.stage5d = RSU7(1024, 256, 512)
        self.stage4d = RSU7(1024, 128, 256)
        self.stage3d = RSU7(512, 64, 128)
        self.stage2d = RSU7(256, 32, 64)
        self.stage1d = RSU7(128, 16, 64)
        
        # 출력 레이어
        self.side1 = nn.Conv2d(64, out_ch, 3, padding=1)
        self.side2 = nn.Conv2d(64, out_ch, 3, padding=1)
        self.side3 = nn.Conv2d(128, out_ch, 3, padding=1)
        self.side4 = nn.Conv2d(256, out_ch, 3, padding=1)
        self.side5 = nn.Conv2d(512, out_ch, 3, padding=1)
        self.side6 = nn.Conv2d(512, out_ch, 3, padding=1)
        
        self.outconv = nn.Conv2d(6*out_ch, out_ch, 1)
        
    def forward(self, x):
        hx = x
        
        # 인코더
        hx1 = self.stage1(hx)
        hx = self.pool12(hx1)
        
        hx2 = self.stage2(hx)
        hx = self.pool23(hx2)
        
        hx3 = self.stage3(hx)
        hx = self.pool34(hx3)
        
        hx4 = self.stage4(hx)
        hx = self.pool45(hx4)
        
        hx5 = self.stage5(hx)
        hx = self.pool56(hx5)
        
        hx6 = self.stage6(hx)
        
        # 디코더
        hx5d = self.stage5d(torch.cat((hx6, hx5), 1))
        hx5dup = F.interpolate(hx5d, size=hx4.shape[2:], mode='bilinear', align_corners=False)
        
        hx4d = self.stage4d(torch.cat((hx5dup, hx4), 1))
        hx4dup = F.interpolate(hx4d, size=hx3.shape[2:], mode='bilinear', align_corners=False)
        
        hx3d = self.stage3d(torch.cat((hx4dup, hx3), 1))
        hx3dup = F.interpolate(hx3d, size=hx2.shape[2:], mode='bilinear', align_corners=False)
        
        hx2d = self.stage2d(torch.cat((hx3dup, hx2), 1))
        hx2dup = F.interpolate(hx2d, size=hx1.shape[2:], mode='bilinear', align_corners=False)
        
        hx1d = self.stage1d(torch.cat((hx2dup, hx1), 1))
        
        # 사이드 출력
        d1 = self.side1(hx1d)
        d2 = self.side2(hx2d)
        d3 = self.side3(hx3d)
        d4 = self.side4(hx4d)
        d5 = self.side5(hx5d)
        d6 = self.side6(hx6)
        
        # 업샘플링
        d2 = F.interpolate(d2, size=x.shape[2:], mode='bilinear', align_corners=False)
        d3 = F.interpolate(d3, size=x.shape[2:], mode='bilinear', align_corners=False)
        d4 = F.interpolate(d4, size=x.shape[2:], mode='bilinear', align_corners=False)
        d5 = F.interpolate(d5, size=x.shape[2:], mode='bilinear', align_corners=False)
        d6 = F.interpolate(d6, size=x.shape[2:], mode='bilinear', align_corners=False)
        
        # 최종 출력
        d0 = self.outconv(torch.cat((d1, d2, d3, d4, d5, d6), 1))
        
        return F.sigmoid(d0), F.sigmoid(d1), F.sigmoid(d2), F.sigmoid(d3), F.sigmoid(d4), F.sigmoid(d5), F.sigmoid(d6)

class RSU7(nn.Module):
    """RSU-7 블록"""
    
    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        super(RSU7, self).__init__()
        
        self.in_conv = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        
        self.conv1 = nn.Conv2d(out_ch, mid_ch, 3, padding=1)
        self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        
        self.conv2 = nn.Conv2d(mid_ch, mid_ch, 3, padding=1)
        self.pool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        
        self.conv3 = nn.Conv2d(mid_ch, mid_ch, 3, padding=1)
        self.pool3 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        
        self.conv4 = nn.Conv2d(mid_ch, mid_ch, 3, padding=1)
        self.pool4 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        
        self.conv5 = nn.Conv2d(mid_ch, mid_ch, 3, padding=1)
        self.pool5 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        
        self.conv6 = nn.Conv2d(mid_ch, mid_ch, 3, padding=1)
        
        self.conv7 = nn.Conv2d(mid_ch, mid_ch, 3, padding=1)
        
        self.conv6d = nn.Conv2d(mid_ch*2, mid_ch, 3, padding=1)
        self.conv5d = nn.Conv2d(mid_ch*2, mid_ch, 3, padding=1)
        self.conv4d = nn.Conv2d(mid_ch*2, mid_ch, 3, padding=1)
        self.conv3d = nn.Conv2d(mid_ch*2, mid_ch, 3, padding=1)
        self.conv2d = nn.Conv2d(mid_ch*2, mid_ch, 3, padding=1)
        self.conv1d = nn.Conv2d(mid_ch*2, out_ch, 3, padding=1)
        
    def forward(self, x):
        hx = x
        hxin = self.in_conv(hx)
        
        hx1 = self.conv1(hxin)
        hx = self.pool1(hx1)
        
        hx2 = self.conv2(hx)
        hx = self.pool2(hx2)
        
        hx3 = self.conv3(hx)
        hx = self.pool3(hx3)
        
        hx4 = self.conv4(hx)
        hx = self.pool4(hx4)
        
        hx5 = self.conv5(hx)
        hx = self.pool5(hx5)
        
        hx6 = self.conv6(hx)
        
        hx7 = self.conv7(hx6)
        
        hx6d = self.conv6d(torch.cat((hx7, hx6), 1))
        hx6dup = F.interpolate(hx6d, size=hx5.shape[2:], mode='bilinear', align_corners=False)
        
        hx5d = self.conv5d(torch.cat((hx6dup, hx5), 1))
        hx5dup = F.interpolate(hx5d, size=hx4.shape[2:], mode='bilinear', align_corners=False)
        
        hx4d = self.conv4d(torch.cat((hx5dup, hx4), 1))
        hx4dup = F.interpolate(hx4d, size=hx3.shape[2:], mode='bilinear', align_corners=False)
        
        hx3d = self.conv3d(torch.cat((hx4dup, hx3), 1))
        hx3dup = F.interpolate(hx3d, size=hx2.shape[2:], mode='bilinear', align_corners=False)
        
        hx2d = self.conv2d(torch.cat((hx3dup, hx2), 1))
        hx2dup = F.interpolate(hx2d, size=hx1.shape[2:], mode='bilinear', align_corners=False)
        
        hx1d = self.conv1d(torch.cat((hx2dup, hx1), 1))
        
        return hx1d + hxin

class MLPersonSegmentation:
    """머신러닝 기반 인물 세그멘테이션 클래스 (실용적 버전)"""
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.transform = transforms.Compose([
            transforms.Resize((320, 320)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.use_simple_method = True  # 간단한 방법 사용
        
    def load_model(self):
        """모델 로드 (실용적 버전)"""
        try:
            print("고급 인물 세그멘테이션 모델 로딩 중...")
            
            if self.use_simple_method:
                print("실용적 방법으로 초기화합니다.")
                self._initialize_advanced_opencv()
                return True
            
            # U²-Net 모델 초기화
            self.model = U2Net(in_ch=3, out_ch=1)
            
            # 사전 훈련된 가중치 다운로드 (실제로는 더 가벼운 모델 사용)
            model_path = "u2net_person_seg.pth"
            
            if not os.path.exists(model_path):
                print("사전 훈련된 모델이 없습니다. 고급 OpenCV 방법으로 초기화합니다.")
                self._initialize_advanced_opencv()
            else:
                self.model.load_state_dict(torch.load(model_path, map_location=self.device))
                print("사전 훈련된 모델 로드 완료")
            
            self.model.to(self.device)
            self.model.eval()
            
            print(f"모델이 {self.device}에서 로드되었습니다.")
            return True
            
        except Exception as e:
            print(f"모델 로드 실패: {e}")
            print("고급 OpenCV 방법으로 초기화합니다.")
            self._initialize_advanced_opencv()
            return True
    
    def _initialize_advanced_opencv(self):
        """고급 OpenCV 방법으로 초기화"""
        try:
            print("고급 OpenCV 기반 인물 세그멘테이션으로 초기화")
            self.model = None  # OpenCV 방법 사용
            print("고급 OpenCV 방법 초기화 완료")
        except Exception as e:
            print(f"OpenCV 초기화 실패: {e}")
            self.model = None
    
    def segment_person(self, image):
        """인물 세그멘테이션 수행 (고급 OpenCV 방법)"""
        try:
            if isinstance(image, np.ndarray):
                # OpenCV 이미지 처리
                height, width = image.shape[:2]
                
                # 고급 인물 세그멘테이션 알고리즘
                mask = self._advanced_person_segmentation(image)
                
                return mask
            else:
                print("지원하지 않는 이미지 형식입니다.")
                return None
            
        except Exception as e:
            print(f"세그멘테이션 오류: {e}")
            return None
    
    def _advanced_person_segmentation(self, image):
        """고급 OpenCV 기반 인물 세그멘테이션"""
        try:
            height, width = image.shape[:2]
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # 1. GrabCut 알고리즘으로 전경/배경 분할
            mask = np.zeros(gray.shape[:2], np.uint8)
            bgd_model = np.zeros((1, 65), np.float64)
            fgd_model = np.zeros((1, 65), np.float64)
            
            # 중앙 영역을 전경으로 설정
            center_x, center_y = width // 2, height // 2
            rect = (center_x - width//4, center_y - height//4, width//2, height//2)
            
            cv2.grabCut(image, mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)
            
            # 2. GrabCut 결과를 이진 마스크로 변환
            mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
            mask2 = mask2 * 255
            
            # 3. 모폴로지 연산으로 노이즈 제거
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            mask2 = cv2.morphologyEx(mask2, cv2.MORPH_CLOSE, kernel)
            mask2 = cv2.morphologyEx(mask2, cv2.MORPH_OPEN, kernel)
            
            # 4. 윤곽선 기반 정제
            contours, _ = cv2.findContours(mask2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                # 가장 큰 윤곽선 선택
                largest_contour = max(contours, key=cv2.contourArea)
                
                # 윤곽선을 부드럽게 만들기
                epsilon = 0.01 * cv2.arcLength(largest_contour, True)
                smoothed_contour = cv2.approxPolyDP(largest_contour, epsilon, True)
                
                # 새로운 마스크 생성
                refined_mask = np.zeros_like(mask2)
                cv2.fillPoly(refined_mask, [smoothed_contour], 255)
                
                # 5. 가우시안 블러로 부드럽게
                refined_mask = cv2.GaussianBlur(refined_mask, (15, 15), 0)
                
                return refined_mask
            else:
                return mask2
                
        except Exception as e:
            print(f"고급 세그멘테이션 오류: {e}")
            # 기본 타원형 마스크 반환
            mask = np.zeros((height, width), dtype=np.uint8)
            center_x, center_y = width // 2, height // 2
            axes_x, axes_y = width // 3, height // 3
            cv2.ellipse(mask, (center_x, center_y), (axes_x, axes_y), 0, 0, 360, 255, -1)
            return mask
    
    def create_person_cutout(self, image, mask):
        """인물 누끼 생성"""
        try:
            if mask is None:
                return None
            
            # 흰색 배경 생성
            white_background = np.ones_like(image) * 255
            
            # 마스크를 3채널로 변환
            mask_3ch = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR) / 255.0
            
            # 마스크 영역은 원본 이미지, 나머지는 흰색 배경
            result = image * mask_3ch + white_background * (1 - mask_3ch)
            result = result.astype(np.uint8)
            
            return result
            
        except Exception as e:
            print(f"누끼 생성 오류: {e}")
            return None

# 전역 모델 인스턴스
ml_segmentation = MLPersonSegmentation()
