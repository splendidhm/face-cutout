import urllib.request
import os

def download_dnn_models():
    """OpenCV DNN 얼굴 감지 모델 다운로드"""
    models = {
        'opencv_face_detector_uint8.pb': 'https://github.com/opencv/opencv_3rdparty/raw/dnn_samples_face_detector_20170830/opencv_face_detector_uint8.pb',
        'opencv_face_detector.pbtxt': 'https://github.com/opencv/opencv_3rdparty/raw/dnn_samples_face_detector_20170830/opencv_face_detector.pbtxt'
    }
    
    # 대안 URL들
    alternative_models = {
        'opencv_face_detector_uint8.pb': 'https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20170830/opencv_face_detector_uint8.pb',
        'opencv_face_detector.pbtxt': 'https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20170830/opencv_face_detector.pbtxt'
    }
    
    print("OpenCV DNN 얼굴 감지 모델 다운로드를 시작합니다...")
    
    for filename, url in models.items():
        if not os.path.exists(filename):
            success = False
            # 기본 URL 시도
            try:
                print(f"{filename} 다운로드 중...")
                urllib.request.urlretrieve(url, filename)
                print(f"{filename} 다운로드 완료!")
                success = True
            except Exception as e:
                print(f"기본 URL 실패: {e}")
            
            # 대안 URL 시도
            if not success and filename in alternative_models:
                try:
                    print(f"대안 URL로 {filename} 다운로드 중...")
                    urllib.request.urlretrieve(alternative_models[filename], filename)
                    print(f"{filename} 다운로드 완료!")
                    success = True
                except Exception as e:
                    print(f"대안 URL도 실패: {e}")
            
            if not success:
                print(f"{filename} 다운로드에 실패했습니다.")
        else:
            print(f"{filename} 이미 존재합니다.")
    
    print("\n모델 다운로드가 완료되었습니다!")

if __name__ == "__main__":
    download_dnn_models()
