#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OpenCV DNN 얼굴 감지 모델 다운로드 스크립트
"""

import os
import urllib.request
import cv2

def download_file(url, filename):
    """파일 다운로드"""
    try:
        print(f"다운로드 중: {filename}")
        urllib.request.urlretrieve(url, filename)
        print(f"다운로드 완료: {filename}")
        return True
    except Exception as e:
        print(f"다운로드 실패: {filename} - {e}")
        return False

def download_dnn_models():
    """DNN 모델 파일들 다운로드"""
    
    # 모델 파일들
    models = {
        "opencv_face_detector_uint8.pb": "https://github.com/opencv/opencv_3rdparty/raw/dnn_samples_face_detector_20170830/opencv_face_detector_uint8.pb",
        "opencv_face_detector.pbtxt": "https://github.com/opencv/opencv/raw/master/samples/dnn/face_detector/opencv_face_detector.pbtxt",
        "res10_300x300_ssd_iter_140000.caffemodel": "https://github.com/opencv/opencv_3rdparty/raw/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel",
        "deploy.prototxt": "https://github.com/opencv/opencv/raw/master/samples/dnn/face_detector/deploy.prototxt"
    }
    
    # 대체 URL들 (GitHub Raw 파일이 안 될 경우)
    alternative_models = {
        "opencv_face_detector_uint8.pb": "https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20170830/opencv_face_detector_uint8.pb",
        "opencv_face_detector.pbtxt": "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/opencv_face_detector.pbtxt",
        "res10_300x300_ssd_iter_140000.caffemodel": "https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel",
        "deploy.prototxt": "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt"
    }
    
    # 로컬 모델 파일들 (OpenCV 설치 경로에서 찾기)
    local_paths = [
        r"C:\opencv\build\etc\haarcascades",
        r"C:\opencv\sources\data\haarcascades",
        r"C:\Program Files\opencv\build\etc\haarcascades",
        r"C:\Program Files (x86)\opencv\build\etc\haarcascades"
    ]
    
    print("OpenCV DNN 얼굴 감지 모델 다운로드 시작...")
    
    # 현재 디렉토리에 models 폴더 생성
    models_dir = "models"
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
    
    success_count = 0
    
    # 모델 파일들 다운로드 시도
    for filename, url in models.items():
        filepath = os.path.join(models_dir, filename)
        
        if os.path.exists(filepath):
            print(f"이미 존재: {filepath}")
            success_count += 1
            continue
        
        # 첫 번째 URL로 시도
        if download_file(url, filepath):
            success_count += 1
        else:
            # 대체 URL로 시도
            alt_url = alternative_models.get(filename)
            if alt_url and download_file(alt_url, filepath):
                success_count += 1
    
    print(f"\n다운로드 완료: {success_count}/{len(models)} 파일")
    
    # OpenCV 버전 확인
    print(f"OpenCV 버전: {cv2.__version__}")
    
    # 다운로드된 모델 테스트
    test_dnn_models()

def test_dnn_models():
    """다운로드된 DNN 모델 테스트"""
    print("\nDNN 모델 테스트 시작...")
    
    models_dir = "models"
    
    # OpenCV DNN 모델 테스트
    pb_file = os.path.join(models_dir, "opencv_face_detector_uint8.pb")
    pbtxt_file = os.path.join(models_dir, "opencv_face_detector.pbtxt")
    
    if os.path.exists(pb_file) and os.path.exists(pbtxt_file):
        try:
            net = cv2.dnn.readNetFromTensorflow(pb_file, pbtxt_file)
            print("✓ OpenCV DNN 모델 로드 성공")
        except Exception as e:
            print(f"✗ OpenCV DNN 모델 로드 실패: {e}")
    else:
        print("✗ OpenCV DNN 모델 파일이 없습니다")
    
    # Caffe 모델 테스트
    caffemodel_file = os.path.join(models_dir, "res10_300x300_ssd_iter_140000.caffemodel")
    prototxt_file = os.path.join(models_dir, "deploy.prototxt")
    
    if os.path.exists(caffemodel_file) and os.path.exists(prototxt_file):
        try:
            net = cv2.dnn.readNetFromCaffe(prototxt_file, caffemodel_file)
            print("✓ Caffe DNN 모델 로드 성공")
        except Exception as e:
            print(f"✗ Caffe DNN 모델 로드 실패: {e}")
    else:
        print("✗ Caffe DNN 모델 파일이 없습니다")

if __name__ == "__main__":
    download_dnn_models()
