import urllib.request
import os

def download_cascade_files():
    """OpenCV Haar Cascade 파일들을 다운로드합니다."""
    
    # 다운로드할 파일 목록
    cascade_files = {
        'haarcascade_frontalface_default.xml': 'https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml',
        'haarcascade_eye.xml': 'https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_eye.xml'
    }
    
    print("OpenCV Haar Cascade 파일 다운로드를 시작합니다...")
    
    for filename, url in cascade_files.items():
        if not os.path.exists(filename):
            try:
                print(f"{filename} 다운로드 중...")
                urllib.request.urlretrieve(url, filename)
                print(f"{filename} 다운로드 완료!")
            except Exception as e:
                print(f"{filename} 다운로드 실패: {e}")
        else:
            print(f"{filename} 이미 존재합니다.")
    
    print("\n다운로드가 완료되었습니다!")

if __name__ == "__main__":
    download_cascade_files()

