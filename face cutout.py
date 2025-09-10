import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
import os
import threading

# MediaPipe 임포트 (선택적)
try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
    print("✓ MediaPipe 사용 가능")
except ImportError:
    MEDIAPIPE_AVAILABLE = False
    print("✗ MediaPipe 사용 불가 (pip install mediapipe로 설치 가능)")

# ML 세그멘테이션 임포트 (선택적)
try:
    from ml_segmentation import ml_segmentation
    ML_SEGMENTATION_AVAILABLE = True
except ImportError:
    ML_SEGMENTATION_AVAILABLE = False

class FaceCutoutTool:
    def __init__(self):
        """얼굴 누끼 따기 도구 초기화"""
        self.root = tk.Tk()
        self.root.title("얼굴 누끼 따기 도구 (ML 기반)")
        self.root.geometry("900x700")
        self.root.resizable(True, True)
        
        # 고급 인물 세그멘테이션 모델 로드
        self.advanced_segmentation_loaded = True
        print("고급 인물 세그멘테이션 모델 초기화 완료")
        
        # OpenCV 고급 얼굴 감지기 로드 (백업용)
        self.face_cascade = self.load_face_cascade()
        self.eye_cascade = self.load_eye_cascade()
        self.nose_cascade = self.load_nose_cascade()
        
        # DNN 모델들
        self.dnn_face_net = None
        self.dnn_loaded = False
        self.load_dnn_models()
        
        # MediaPipe 모델들
        self.mp_face_detection = None
        self.mp_loaded = False
        if MEDIAPIPE_AVAILABLE:
            self.load_mediapipe_models()
        
        # 변수 초기화
        self.original_image = None
        self.processed_image = None
        self.image_path = None
        
        # 마우스 드래그 관련 변수
        self.drag_start = None
        self.drag_end = None
        self.is_dragging = False
        self.selection_rect = None
        
        self.setup_ui()
    
    def load_dnn_models(self):
        """DNN 기반 고성능 얼굴 감지 모델 로드"""
        try:
            # 현재 작업 디렉토리 확인
            current_dir = os.getcwd()
            print(f"현재 작업 디렉토리: {current_dir}")
            
            # Caffe 모델 파일 경로 (절대 경로 사용)
            prototxt_path = os.path.join(current_dir, "models", "deploy.prototxt")
            caffemodel_path = os.path.join(current_dir, "models", "res10_300x300_ssd_iter_140000.caffemodel")
            
            print(f"Prototxt 경로: {prototxt_path}")
            print(f"Caffemodel 경로: {caffemodel_path}")
            print(f"Prototxt 존재: {os.path.exists(prototxt_path)}")
            print(f"Caffemodel 존재: {os.path.exists(caffemodel_path)}")
            
            if os.path.exists(prototxt_path) and os.path.exists(caffemodel_path):
                # Caffe 모델 로드
                self.dnn_face_net = cv2.dnn.readNetFromCaffe(prototxt_path, caffemodel_path)
                self.dnn_loaded = True
                print("✓ DNN 얼굴 감지 모델 로드 성공 (Caffe)")
                return True
            else:
                print("✗ DNN 모델 파일이 없습니다")
                # 상대 경로도 시도
                prototxt_path_rel = "models/deploy.prototxt"
                caffemodel_path_rel = "models/res10_300x300_ssd_iter_140000.caffemodel"
                
                if os.path.exists(prototxt_path_rel) and os.path.exists(caffemodel_path_rel):
                    self.dnn_face_net = cv2.dnn.readNetFromCaffe(prototxt_path_rel, caffemodel_path_rel)
                    self.dnn_loaded = True
                    print("✓ DNN 얼굴 감지 모델 로드 성공 (상대 경로)")
                    return True
                return False
                
        except Exception as e:
            print(f"DNN 모델 로드 실패: {e}")
            return False
    
    def detect_faces_dnn(self, image, confidence_threshold=0.5):
        """DNN 기반 고성능 얼굴 감지"""
        if not self.dnn_loaded or self.dnn_face_net is None:
            return []
        
        try:
            height, width = image.shape[:2]
            
            # 입력 이미지 전처리 (300x300으로 리사이즈)
            blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300), [104, 117, 123])
            
            # DNN 모델에 입력
            self.dnn_face_net.setInput(blob)
            detections = self.dnn_face_net.forward()
            
            faces = []
            for i in range(detections.shape[2]):
                confidence = detections[0, 0, i, 2]
                
                if confidence > confidence_threshold:
                    # 바운딩 박스 좌표 계산
                    x1 = int(detections[0, 0, i, 3] * width)
                    y1 = int(detections[0, 0, i, 4] * height)
                    x2 = int(detections[0, 0, i, 5] * width)
                    y2 = int(detections[0, 0, i, 6] * height)
                    
                    # OpenCV 형식으로 변환 (x, y, w, h)
                    face = (x1, y1, x2 - x1, y2 - y1)
                    faces.append(face)
            
            print(f"DNN 얼굴 감지: {len(faces)}개 얼굴 발견 (신뢰도: {confidence_threshold})")
            return faces
            
        except Exception as e:
            print(f"DNN 얼굴 감지 오류: {e}")
            return []
    
    def load_mediapipe_models(self):
        """MediaPipe 기반 고성능 얼굴 감지 모델 로드"""
        try:
            if not MEDIAPIPE_AVAILABLE:
                return False
                
            # MediaPipe 얼굴 감지 초기화
            self.mp_face_detection = mp.solutions.face_detection
            self.mp_drawing = mp.solutions.drawing_utils
            self.mp_loaded = True
            print("✓ MediaPipe 얼굴 감지 모델 로드 성공")
            return True
            
        except Exception as e:
            print(f"MediaPipe 모델 로드 실패: {e}")
            return False
    
    def detect_faces_mediapipe(self, image, confidence_threshold=0.5):
        """MediaPipe 기반 고성능 얼굴 감지"""
        if not self.mp_loaded or self.mp_face_detection is None:
            return []
        
        try:
            height, width = image.shape[:2]
            
            # MediaPipe 얼굴 감지 실행
            with self.mp_face_detection.FaceDetection(
                model_selection=0,  # 0: 짧은 거리, 1: 긴 거리
                min_detection_confidence=confidence_threshold
            ) as face_detection:
                
                # RGB로 변환 (MediaPipe는 RGB 입력 필요)
                rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                results = face_detection.process(rgb_image)
                
                faces = []
                if results.detections:
                    for detection in results.detections:
                        # 바운딩 박스 추출
                        bbox = detection.location_data.relative_bounding_box
                        
                        # 실제 픽셀 좌표로 변환
                        x = int(bbox.xmin * width)
                        y = int(bbox.ymin * height)
                        w = int(bbox.width * width)
                        h = int(bbox.height * height)
                        
                        # OpenCV 형식으로 변환
                        face = (x, y, w, h)
                        faces.append(face)
                
                print(f"MediaPipe 얼굴 감지: {len(faces)}개 얼굴 발견 (신뢰도: {confidence_threshold})")
                return faces
                
        except Exception as e:
            print(f"MediaPipe 얼굴 감지 오류: {e}")
            return []
        
    def load_face_cascade(self):
        """얼굴 감지기 로드 (안전한 방법)"""
        try:
            # 방법 1: 현재 디렉토리에서 찾기 (우선순위 1)
            local_path = 'haarcascade_frontalface_default.xml'
            if os.path.exists(local_path):
                cascade = cv2.CascadeClassifier(local_path)
                if not cascade.empty():
                    print("현재 디렉토리에서 얼굴 감지기를 로드했습니다.")
                    return cascade
            
            # 방법 2: 상대 경로로 찾기
            relative_path = os.path.join(os.path.dirname(__file__), 'haarcascade_frontalface_default.xml')
            if os.path.exists(relative_path):
                cascade = cv2.CascadeClassifier(relative_path)
                if not cascade.empty():
                    print("상대 경로에서 얼굴 감지기를 로드했습니다.")
                    return cascade
            
            # 방법 3: OpenCV 내장 경로 사용 (오류 메시지 억제)
            try:
                cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
                if os.path.exists(cascade_path):
                    cascade = cv2.CascadeClassifier(cascade_path)
                    if not cascade.empty():
                        print("OpenCV 내장 경로에서 얼굴 감지기를 로드했습니다.")
                        return cascade
            except:
                pass
            
            # 모든 방법이 실패한 경우 None 반환
            print("경고: 얼굴 감지기를 로드할 수 없습니다. 수동 마스킹 모드로 작동합니다.")
            return None
            
        except Exception as e:
            print(f"얼굴 감지기 로드 오류: {e}")
            return None
    
    def load_eye_cascade(self):
        """눈 감지기 로드"""
        try:
            # 방법 1: 현재 디렉토리에서 찾기
            local_path = 'haarcascade_eye.xml'
            if os.path.exists(local_path):
                cascade = cv2.CascadeClassifier(local_path)
                if not cascade.empty():
                    print("현재 디렉토리에서 눈 감지기를 로드했습니다.")
                    return cascade
            
            # 방법 2: OpenCV 내장 경로 사용
            try:
                cascade_path = cv2.data.haarcascades + 'haarcascade_eye.xml'
                if os.path.exists(cascade_path):
                    cascade = cv2.CascadeClassifier(cascade_path)
                    if not cascade.empty():
                        print("OpenCV 내장 경로에서 눈 감지기를 로드했습니다.")
                        return cascade
            except:
                pass
            
            print("경고: 눈 감지기를 로드할 수 없습니다.")
            return None
            
        except Exception as e:
            print(f"눈 감지기 로드 오류: {e}")
            return None
    
    def load_nose_cascade(self):
        """코 감지기 로드"""
        try:
            # 방법 1: 현재 디렉토리에서 찾기
            local_path = 'haarcascade_nose.xml'
            if os.path.exists(local_path):
                cascade = cv2.CascadeClassifier(local_path)
                if not cascade.empty():
                    print("현재 디렉토리에서 코 감지기를 로드했습니다.")
                    return cascade
            
            # 방법 2: OpenCV 내장 경로 사용
            try:
                cascade_path = cv2.data.haarcascades + 'haarcascade_nose.xml'
                if os.path.exists(cascade_path):
                    cascade = cv2.CascadeClassifier(cascade_path)
                    if not cascade.empty():
                        print("OpenCV 내장 경로에서 코 감지기를 로드했습니다.")
                        return cascade
            except:
                pass
            
            print("경고: 코 감지기를 로드할 수 없습니다.")
            return None
            
        except Exception as e:
            print(f"코 감지기 로드 오류: {e}")
            return None
        
    def setup_ui(self):
        """사용자 인터페이스 설정"""
        # 메인 프레임
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # 제목
        title_label = ttk.Label(main_frame, text="얼굴 누끼 따기 도구", font=("Arial", 16, "bold"))
        title_label.grid(row=0, column=0, columnspan=3, pady=(0, 20))
        
        # 버튼 프레임
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=1, column=0, columnspan=3, pady=(0, 20))
        
        # 이미지 선택 버튼
        self.select_btn = ttk.Button(button_frame, text="이미지 선택", command=self.select_image)
        self.select_btn.pack(side=tk.LEFT, padx=(0, 10))
        
        # 영역 선택 버튼
        self.select_area_btn = ttk.Button(button_frame, text="영역 선택", command=self.toggle_area_selection, state=tk.DISABLED)
        self.select_area_btn.pack(side=tk.LEFT, padx=(0, 10))
        
        # 얼굴 감지 및 누끼 따기 버튼
        self.process_btn = ttk.Button(button_frame, text="얼굴 누끼 따기", command=self.process_image, state=tk.DISABLED)
        self.process_btn.pack(side=tk.LEFT, padx=(0, 10))
        
        # 결과 저장 버튼
        self.save_btn = ttk.Button(button_frame, text="결과 저장", command=self.save_result, state=tk.DISABLED)
        self.save_btn.pack(side=tk.LEFT, padx=(0, 10))
        
        # 진행률 표시
        self.progress = ttk.Progressbar(button_frame, mode='indeterminate')
        self.progress.pack(side=tk.LEFT, padx=(20, 0))
        
        # 이미지 표시 프레임 (가운데 정렬을 위한 컨테이너)
        image_container = ttk.Frame(main_frame)
        image_container.grid(row=2, column=0, columnspan=3, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # 이미지 프레임을 가운데 정렬하기 위한 중간 프레임
        image_frame = ttk.Frame(image_container)
        image_frame.pack(expand=True, fill=tk.BOTH)
        
        # 원본 이미지
        original_frame = ttk.LabelFrame(image_frame, text="원본 이미지", padding="5")
        original_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))
        
        self.original_label = ttk.Label(original_frame, text="이미지를 선택해주세요", width=40)
        self.original_label.pack(expand=True, fill=tk.BOTH)
        
        # 마우스 이벤트 바인딩
        self.original_label.bind("<Button-1>", self.on_mouse_press)
        self.original_label.bind("<B1-Motion>", self.on_mouse_drag)
        self.original_label.bind("<ButtonRelease-1>", self.on_mouse_release)
        
        # 처리된 이미지
        processed_frame = ttk.LabelFrame(image_frame, text="처리된 이미지", padding="5")
        processed_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(10, 0))
        
        self.processed_label = ttk.Label(processed_frame, text="처리 결과가 여기에 표시됩니다", width=40)
        self.processed_label.pack(expand=True, fill=tk.BOTH)
        
        # 상태 표시
        self.status_label = ttk.Label(main_frame, text="대기 중...", font=("Arial", 10))
        self.status_label.grid(row=3, column=0, columnspan=3, pady=(20, 0))
        
        # 그리드 가중치 설정
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)
        main_frame.rowconfigure(2, weight=1)
        image_frame.columnconfigure(0, weight=1)
        image_frame.columnconfigure(1, weight=1)
        image_frame.rowconfigure(0, weight=1)
        
    def select_image(self):
        """이미지 파일 선택"""
        file_path = filedialog.askopenfilename(
            title="이미지 파일 선택",
            filetypes=[
                ("이미지 파일", "*.jpg *.jpeg *.png *.bmp *.tiff"),
                ("JPEG 파일", "*.jpg *.jpeg"),
                ("PNG 파일", "*.png"),
                ("모든 파일", "*.*")
            ]
        )
        
        if file_path:
            self.image_path = file_path
            self.load_image()
            
    def load_image(self):
        """이미지 로드 및 표시"""
        try:
            # 파일 경로 정규화 및 확인
            normalized_path = os.path.normpath(self.image_path)
            print(f"이미지 로드 시도: {normalized_path}")
            
            if not os.path.exists(normalized_path):
                raise ValueError(f"파일이 존재하지 않습니다: {normalized_path}")
            
            # 파일 크기 확인
            file_size = os.path.getsize(normalized_path)
            if file_size == 0:
                raise ValueError("파일이 비어있습니다")
            
            print(f"파일 크기: {file_size} bytes")
            
            # 방법 1: OpenCV로 이미지 로드 시도
            self.original_image = cv2.imread(normalized_path)
            
            # 방법 2: OpenCV가 실패하면 PIL로 시도
            if self.original_image is None:
                print("OpenCV 로드 실패, PIL로 시도...")
                try:
                    pil_image = Image.open(normalized_path)
                    # PIL 이미지를 numpy 배열로 변환
                    self.original_image = np.array(pil_image)
                    
                    # RGB를 BGR로 변환 (OpenCV 형식)
                    if len(self.original_image.shape) == 3 and self.original_image.shape[2] == 3:
                        self.original_image = cv2.cvtColor(self.original_image, cv2.COLOR_RGB2BGR)
                    elif len(self.original_image.shape) == 3 and self.original_image.shape[2] == 4:
                        self.original_image = cv2.cvtColor(self.original_image, cv2.COLOR_RGBA2BGR)
                    
                    print("PIL을 통한 이미지 로드 성공")
                except Exception as pil_error:
                    print(f"PIL 로드도 실패: {pil_error}")
                    raise ValueError(f"이미지를 로드할 수 없습니다. 지원 형식: JPG, PNG, BMP, TIFF")
            
            if self.original_image is None:
                raise ValueError("이미지를 로드할 수 없습니다")
            
            print(f"이미지 로드 성공: {self.original_image.shape}")
            
            # BGR을 RGB로 변환 (표시용)
            if len(self.original_image.shape) == 3:
                rgb_image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2RGB)
            else:
                rgb_image = self.original_image
            
            # PIL 이미지로 변환
            pil_image = Image.fromarray(rgb_image)
            
            # 이미지 크기 조정 (UI에 맞게)
            display_size = (300, 300)
            pil_image.thumbnail(display_size, Image.Resampling.LANCZOS)
            
            # Tkinter용 이미지로 변환
            tk_image = ImageTk.PhotoImage(pil_image)
            
            # 원본 이미지 표시
            self.original_label.configure(image=tk_image, text="")
            self.original_label.image = tk_image  # 참조 유지
            
            # 버튼 활성화
            self.select_area_btn.configure(state=tk.NORMAL)
            self.process_btn.configure(state=tk.NORMAL)
            self.save_btn.configure(state=tk.DISABLED)
            
            # 상태 업데이트
            self.status_label.configure(text=f"이미지 로드 완료: {os.path.basename(self.image_path)}")
            
        except Exception as e:
            error_msg = f"이미지 로드 실패: {str(e)}"
            print(error_msg)
            messagebox.showerror("오류", error_msg)
            self.status_label.configure(text="이미지 로드 실패")
            
    def toggle_area_selection(self):
        """영역 선택 모드 토글"""
        if hasattr(self, 'area_selection_mode'):
            self.area_selection_mode = not self.area_selection_mode
        else:
            self.area_selection_mode = True
            
        if self.area_selection_mode:
            self.select_area_btn.configure(text="영역 선택 중...")
            self.status_label.configure(text="마우스로 드래그하여 얼굴 영역을 선택하세요")
            # 격자 마우스 포인터 설정
            self._set_crosshair_cursor()
        else:
            self.select_area_btn.configure(text="영역 선택")
            self.status_label.configure(text="영역 선택 모드가 해제되었습니다")
            # 기본 마우스 포인터로 복원
            self._set_default_cursor()
            # 영역 선택 모드 해제 시 원본 이미지 복원
            self.restore_original_image()
    
    def _convert_ui_to_image_coords(self, ui_x, ui_y):
        """UI 좌표를 실제 이미지 좌표로 변환 (개선된 버전)"""
        if self.original_image is None:
            return ui_x, ui_y
        
        # 이미지 크기
        img_height, img_width = self.original_image.shape[:2]
        
        # 실제 UI 표시 크기 가져오기
        try:
            ui_width = self.original_label.winfo_width()
            ui_height = self.original_label.winfo_height()
            
            # UI 크기가 0이면 기본값 사용
            if ui_width <= 0 or ui_height <= 0:
                ui_width = 300
                ui_height = 300
        except:
            ui_width = 300
            ui_height = 300
        
        # 스케일 계산
        scale_x = img_width / ui_width
        scale_y = img_height / ui_height
        
        # 좌표 변환
        actual_x = int(ui_x * scale_x)
        actual_y = int(ui_y * scale_y)
        
        # 경계 확인
        actual_x = max(0, min(actual_x, img_width - 1))
        actual_y = max(0, min(actual_y, img_height - 1))
        
        return actual_x, actual_y
    
    def _convert_image_to_ui_coords(self, img_x, img_y):
        """실제 이미지 좌표를 UI 좌표로 변환 (개선된 버전)"""
        if self.original_image is None:
            return img_x, img_y
        
        # 이미지 크기
        img_height, img_width = self.original_image.shape[:2]
        
        # 실제 UI 표시 크기 가져오기
        try:
            ui_width = self.original_label.winfo_width()
            ui_height = self.original_label.winfo_height()
            
            # UI 크기가 0이면 기본값 사용
            if ui_width <= 0 or ui_height <= 0:
                ui_width = 300
                ui_height = 300
        except:
            ui_width = 300
            ui_height = 300
        
        # 스케일 계산
        scale_x = ui_width / img_width
        scale_y = ui_height / img_height
        
        # 좌표 변환
        ui_x = int(img_x * scale_x)
        ui_y = int(img_y * scale_y)
        
        return ui_x, ui_y
    
    def _set_crosshair_cursor(self):
        """격자 마우스 포인터 설정"""
        try:
            # 격자 모양 커서 생성
            cursor_data = [
                "16 16 2 1",
                "  c None",
                ". c #000000",
                "                ",
                "       .        ",
                "       .        ",
                "       .        ",
                "       .        ",
                "       .        ",
                "....... ........",
                "       .        ",
                "       .        ",
                "       .        ",
                "       .        ",
                "       .        ",
                "                ",
                "                ",
                "                "
            ]
            
            # 커서 생성
            cursor = tk.BitmapImage(data=cursor_data)
            self.root.configure(cursor="@cursor")
            
        except Exception as e:
            print(f"격자 커서 설정 실패: {e}")
            # 기본 격자 커서 사용
            self.root.configure(cursor="crosshair")
    
    def _set_default_cursor(self):
        """기본 마우스 포인터로 복원"""
        self.root.configure(cursor="")
            
    def on_mouse_press(self, event):
        """마우스 버튼 누름 이벤트"""
        if not hasattr(self, 'area_selection_mode') or not self.area_selection_mode:
            return
        
        # 실제 이미지 좌표로 변환
        actual_x, actual_y = self._convert_ui_to_image_coords(event.x, event.y)
        self.drag_start = (actual_x, actual_y)
        self.is_dragging = True
        print(f"드래그 시작 (실제 좌표): {self.drag_start}")
        
        # 선택 캔버스 생성
        self.create_selection_canvas()
        
    def on_mouse_drag(self, event):
        """마우스 드래그 이벤트"""
        if not self.is_dragging or not hasattr(self, 'area_selection_mode') or not self.area_selection_mode:
            return
        
        # 실제 이미지 좌표로 변환
        actual_x, actual_y = self._convert_ui_to_image_coords(event.x, event.y)
        self.drag_end = (actual_x, actual_y)
        self.draw_selection_rectangle()
        
    def on_mouse_release(self, event):
        """마우스 버튼 놓음 이벤트"""
        if not self.is_dragging or not hasattr(self, 'area_selection_mode') or not self.area_selection_mode:
            return
        
        # 실제 이미지 좌표로 변환
        actual_x, actual_y = self._convert_ui_to_image_coords(event.x, event.y)
        self.drag_end = (actual_x, actual_y)
        self.is_dragging = False
        
        if self.drag_start and self.drag_end:
            # 선택된 영역 정보 저장
            x1, y1 = self.drag_start
            x2, y2 = self.drag_end
            
            # 좌표 정규화
            self.selection_rect = (
                min(x1, x2), min(y1, y2),
                abs(x2 - x1), abs(y2 - y1)
            )
            
            print(f"선택된 영역: {self.selection_rect}")
            self.status_label.configure(text=f"영역 선택 완료: {self.selection_rect[2]}x{self.selection_rect[3]}")
            
            # 영역 선택 완료 후 원본 이미지 복원
            self.restore_original_image()
            
    def draw_selection_rectangle(self):
        """선택 영역 사각형 그리기 (개선된 버전)"""
        if not self.drag_start or not self.drag_end:
            return
            
        # 기존 선택 영역 지우기
        if hasattr(self, 'selection_canvas'):
            self.selection_canvas.destroy()
            
        # 원본 이미지가 있는 경우에만 캔버스 생성
        if not hasattr(self.original_label, 'image') or self.original_label.image is None:
            return
            
        # 새로운 캔버스 생성 (투명 배경)
        self.selection_canvas = tk.Canvas(
            self.original_label, 
            highlightthickness=0,
            bg='white',  # 흰색 배경으로 설정
            relief='flat',
            bd=0
        )
        self.selection_canvas.place(x=0, y=0, relwidth=1, relheight=1)
        
        # 원본 이미지를 캔버스에 다시 그리기
        if hasattr(self.original_label, 'image') and self.original_label.image is not None:
            # 캔버스 크기 가져오기
            canvas_width = self.original_label.winfo_width()
            canvas_height = self.original_label.winfo_height()
            
            # 이미지를 캔버스에 그리기
            self.selection_canvas.create_image(
                canvas_width // 2, canvas_height // 2,
                image=self.original_label.image,
                anchor=tk.CENTER
            )
        
        # 실제 이미지 좌표를 UI 좌표로 변환
        ui_x1, ui_y1 = self._convert_image_to_ui_coords(self.drag_start[0], self.drag_start[1])
        ui_x2, ui_y2 = self._convert_image_to_ui_coords(self.drag_end[0], self.drag_end[1])
        
        # 사각형 그리기
        x1, y1 = ui_x1, ui_y1
        x2, y2 = ui_x2, ui_y2
        
        # 반투명한 사각형 그리기
        self.selection_canvas.create_rectangle(
            x1, y1, x2, y2,
            outline="red", width=3, fill="red", 
            stipple="gray25"  # 반투명 패턴
        )
        
        # 선택 영역 크기 표시
        width = abs(x2 - x1)
        height = abs(y2 - y1)
        self.selection_canvas.create_text(
            (x1 + x2) // 2, (y1 + y2) // 2,
            text=f"{width}x{height}",
            fill="white", font=("Arial", 10, "bold")
        )
    
    def create_selection_canvas(self):
        """선택 캔버스 생성"""
        # 기존 선택 캔버스가 있으면 제거
        if hasattr(self, 'selection_canvas'):
            self.selection_canvas.destroy()
            
        # 원본 이미지가 있는 경우에만 캔버스 생성
        if not hasattr(self.original_label, 'image') or self.original_label.image is None:
            return
            
        # 새로운 캔버스 생성
        self.selection_canvas = tk.Canvas(
            self.original_label, 
            highlightthickness=0,
            bg='white',
            relief='flat',
            bd=0
        )
        self.selection_canvas.place(x=0, y=0, relwidth=1, relheight=1)
        
        # 원본 이미지를 캔버스에 다시 그리기
        if hasattr(self.original_label, 'image') and self.original_label.image is not None:
            # 캔버스 크기 가져오기
            canvas_width = self.original_label.winfo_width()
            canvas_height = self.original_label.winfo_height()
            
            # 이미지를 캔버스에 그리기
            self.selection_canvas.create_image(
                canvas_width // 2, canvas_height // 2,
                image=self.original_label.image,
                anchor=tk.CENTER
            )
        
    def restore_original_image(self):
        """원본 이미지 복원"""
        # 선택 캔버스 제거
        if hasattr(self, 'selection_canvas'):
            self.selection_canvas.destroy()
            delattr(self, 'selection_canvas')
            
        # 원본 이미지가 있다면 다시 표시
        if hasattr(self.original_label, 'image') and self.original_label.image is not None:
            self.original_label.configure(image=self.original_label.image, text="")
            
    def process_image(self):
        """이미지 처리 (얼굴 감지 및 누끼 따기)"""
        if self.original_image is None:
            return
            
        # UI 비활성화
        self.select_btn.configure(state=tk.DISABLED)
        self.process_btn.configure(state=tk.DISABLED)
        self.progress.start()
        self.status_label.configure(text="이미지 처리 중...")
        
        # 별도 스레드에서 처리
        thread = threading.Thread(target=self._process_image_thread)
        thread.daemon = True
        thread.start()
        
    def _process_image_thread(self):
        """이미지 처리 스레드 (머신러닝 기반)"""
        try:
            # 이미지 크기 정보 가져오기
            height, width = self.original_image.shape[:2]
            print(f"이미지 크기: {width}x{height}")
            
            # 고급 인물 세그멘테이션 사용
            if self.advanced_segmentation_loaded:
                print("고급 인물 세그멘테이션 시작")
                
                # 선택된 영역이 있는지 확인
                if hasattr(self, 'selection_rect') and self.selection_rect:
                    print(f"선택된 영역 사용: {self.selection_rect}")
                    mask = self._create_ultra_advanced_face_mask_with_selection()
                else:
                    print("전체 이미지에서 고급 세그멘테이션 수행")
                    mask = self._create_ultra_advanced_face_mask_full_image()
                
                # 고급 세그멘테이션으로 누끼 생성
                if mask is not None:
                    self.processed_image = self._create_advanced_cutout(self.original_image, mask)
                    if self.processed_image is not None:
                        print("고급 인물 누끼 완료")
                    else:
                        print("고급 누끼 생성 실패, 기본 방법으로 대체")
                        self._fallback_to_opencv()
                else:
                    print("고급 마스크 생성 실패, 기본 방법으로 대체")
                    self._fallback_to_opencv()
            else:
                print("고급 세그멘테이션 로드 실패, 기본 방법 사용")
                self._fallback_to_opencv()
            
            # UI 업데이트를 메인 스레드에서 실행
            self.root.after(0, self._update_ui_after_processing)
            
        except Exception as e:
            print(f"이미지 처리 오류: {e}")
            # 오류 발생 시 메인 스레드에서 UI 업데이트
            self.root.after(0, lambda: self._handle_processing_error(str(e)))
    
    def _fallback_to_opencv(self):
        """OpenCV 방법으로 대체 처리"""
        try:
            print("OpenCV 방법으로 대체 처리 중...")
            
            # 선택된 영역이 있는지 확인
            if hasattr(self, 'selection_rect') and self.selection_rect:
                print(f"선택된 영역 사용: {self.selection_rect}")
                mask = self._create_advanced_face_mask_with_selection()
            else:
                print("전체 이미지에서 얼굴 감지")
                mask = self._create_advanced_face_mask_full_image()
            
            # 마스크를 사용하여 배경을 흰색으로 처리
            self.processed_image = self.original_image.copy()
            
            # 흰색 배경 생성
            white_background = np.ones_like(self.processed_image) * 255
            
            # 마스크를 3채널로 변환 (0-1 범위로 정규화)
            mask_3ch = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR) / 255.0
            
            # 마스크 영역은 원본 이미지, 나머지는 흰색 배경
            self.processed_image = self.processed_image * mask_3ch + white_background * (1 - mask_3ch)
            self.processed_image = self.processed_image.astype(np.uint8)
            
            print("OpenCV 기반 얼굴 마스킹 완료")
            
        except Exception as e:
            print(f"OpenCV 대체 처리 오류: {e}")
            raise e
    
    def _create_ultra_advanced_face_mask_with_selection(self):
        """선택된 영역에서 초고급 얼굴 마스크 생성"""
        height, width = self.original_image.shape[:2]
        
        # 선택된 영역 좌표를 실제 이미지 좌표로 변환
        sel_x, sel_y, sel_w, sel_h = self.selection_rect
        
        # UI 이미지 크기와 실제 이미지 크기 비율 계산
        ui_width = 300
        ui_height = 300
        
        # 실제 이미지에서의 선택 영역 계산
        scale_x = width / ui_width
        scale_y = height / ui_height
        
        actual_x = int(sel_x * scale_x)
        actual_y = int(sel_y * scale_y)
        actual_w = int(sel_w * scale_x)
        actual_h = int(sel_h * scale_y)
        
        # 경계 확인
        actual_x = max(0, min(actual_x, width - 1))
        actual_y = max(0, min(actual_y, height - 1))
        actual_w = min(actual_w, width - actual_x)
        actual_h = min(actual_h, height - actual_y)
        
        print(f"실제 선택 영역: x={actual_x}, y={actual_y}, w={actual_w}, h={actual_h}")
        
        # 선택된 영역에서 초고급 세그멘테이션 수행
        roi = self.original_image[actual_y:actual_y+actual_h, actual_x:actual_x+actual_w]
        roi_mask = self._ultra_advanced_segmentation(roi)
        
        # 전체 이미지 마스크 생성
        mask = np.zeros((height, width), dtype=np.uint8)
        mask[actual_y:actual_y+actual_h, actual_x:actual_x+actual_w] = roi_mask
        
        return mask
    
    def _create_ultra_advanced_face_mask_full_image(self):
        """전체 이미지에서 초고급 얼굴 마스크 생성"""
        return self._ultra_advanced_segmentation(self.original_image)
    
    def _ultra_advanced_segmentation(self, image):
        """초고급 인물 세그멘테이션 알고리즘"""
        try:
            height, width = image.shape[:2]
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # 1. 다중 스케일 GrabCut 알고리즘
            masks = []
            
            # 헤어와 흰색 옷을 포함한 더 큰 영역으로 GrabCut 수행
            rects = [
                (width//12, height//12, width*5//6, height*5//6),  # 매우 큰 영역 (헤어 포함)
                (width//8, height//8, width*3//4, height*3//4),    # 중앙
                (width//6, height//6, width*2//3, height*2//3),    # 약간 작게
                (width//4, height//4, width//2, height//2),        # 더 작게
            ]
            
            for rect in rects:
                mask = np.zeros(gray.shape[:2], np.uint8)
                bgd_model = np.zeros((1, 65), np.float64)
                fgd_model = np.zeros((1, 65), np.float64)
                
                try:
                    # 더 많은 반복으로 정확도 향상
                    cv2.grabCut(image, mask, rect, bgd_model, fgd_model, 8, cv2.GC_INIT_WITH_RECT)
                    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
                    masks.append(mask2 * 255)
                    print(f"GrabCut {len(masks)} 완료")
                except Exception as e:
                    print(f"GrabCut 실패: {e}")
                    continue
            
            # 2. 색상 기반 세그멘테이션 추가 (흰색 옷 인식)
            color_mask = self._create_enhanced_color_mask(image)
            if color_mask is not None:
                masks.append(color_mask)
                print("색상 기반 마스크 추가")
            
            # 3. 다중 마스크 결합
            if masks:
                combined_mask = np.zeros_like(masks[0])
                for mask in masks:
                    combined_mask = cv2.bitwise_or(combined_mask, mask)
                print("마스크 결합 완료")
            else:
                combined_mask = np.zeros((height, width), dtype=np.uint8)
            
            # 4. 고급 모폴로지 연산 (더 큰 커널 사용)
            # 닫힘 연산으로 구멍 메우기
            kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
            combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel_close, iterations=3)
            
            # 열림 연산으로 노이즈 제거
            kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel_open, iterations=2)
            
            # 5. 윤곽선 기반 정제 (개선된 버전)
            contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                # 가장 큰 윤곽선들을 선택 (상위 5개)
                contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]
                
                # 정제된 마스크 생성
                refined_mask = np.zeros_like(combined_mask)
                
                for contour in contours:
                    # 윤곽선 근사화 (더 정밀하게)
                    epsilon = 0.003 * cv2.arcLength(contour, True)
                    smoothed_contour = cv2.approxPolyDP(contour, epsilon, True)
                    cv2.fillPoly(refined_mask, [smoothed_contour], 255)
                
                # 6. 헤어 영역 확장
                refined_mask = self._expand_hair_region_advanced(refined_mask, image)
                
                # 7. 가우시안 블러로 부드럽게 (더 부드럽게)
                refined_mask = cv2.GaussianBlur(refined_mask, (31, 31), 0)
                
                print("초고급 세그멘테이션 완료")
                return refined_mask
            else:
                # 윤곽선이 없으면 더 큰 기본 타원형 마스크
                mask = np.zeros((height, width), dtype=np.uint8)
                center_x, center_y = width // 2, height // 2
                axes_x, axes_y = width // 2, height // 2  # 더 큰 타원
                cv2.ellipse(mask, (center_x, center_y), (axes_x, axes_y), 0, 0, 360, 255, -1)
                mask = cv2.GaussianBlur(mask, (31, 31), 0)
                return mask
                
        except Exception as e:
            print(f"초고급 세그멘테이션 오류: {e}")
            # 기본 타원형 마스크 반환 (더 큰 크기)
            height, width = image.shape[:2]
            mask = np.zeros((height, width), dtype=np.uint8)
            center_x, center_y = width // 2, height // 2
            axes_x, axes_y = width // 2, height // 2  # 더 큰 타원
            cv2.ellipse(mask, (center_x, center_y), (axes_x, axes_y), 0, 0, 360, 255, -1)
            return mask
    
    def _create_enhanced_color_mask(self, image):
        """향상된 색상 기반 마스크 생성 (흰색 옷 인식)"""
        try:
            height, width = image.shape[:2]
            
            # HSV 색상 공간으로 변환
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            
            # 흰색 범위 정의 (더 넓은 범위)
            lower_white = np.array([0, 0, 200])  # 밝은 흰색
            upper_white = np.array([180, 30, 255])  # 약간 회색빛 흰색
            
            # 흰색 마스크 생성
            white_mask = cv2.inRange(hsv, lower_white, upper_white)
            
            # 피부색 범위 정의 (더 넓은 범위)
            lower_skin1 = np.array([0, 20, 70])
            upper_skin1 = np.array([20, 255, 255])
            lower_skin2 = np.array([160, 20, 70])
            upper_skin2 = np.array([180, 255, 255])
            
            # 피부색 마스크 생성
            skin_mask1 = cv2.inRange(hsv, lower_skin1, upper_skin1)
            skin_mask2 = cv2.inRange(hsv, lower_skin2, upper_skin2)
            skin_mask = cv2.bitwise_or(skin_mask1, skin_mask2)
            
            # 검은색/어두운 색 범위 (헤어)
            lower_dark = np.array([0, 0, 0])
            upper_dark = np.array([180, 255, 50])
            dark_mask = cv2.inRange(hsv, lower_dark, upper_dark)
            
            # 모든 마스크 결합
            combined_mask = cv2.bitwise_or(white_mask, skin_mask)
            combined_mask = cv2.bitwise_or(combined_mask, dark_mask)
            
            # 모폴로지 연산으로 정제
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
            combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
            
            return combined_mask
            
        except Exception as e:
            print(f"색상 기반 마스크 생성 오류: {e}")
            return None
    
    def _expand_hair_region_advanced(self, mask, image):
        """고급 헤어 영역 확장"""
        try:
            height, width = mask.shape[:2]
            
            # 마스크의 상단 영역 찾기
            top_region = mask[:height//3, :]  # 상단 1/3 영역
            
            # 상단 영역에서 가장 높은 점 찾기
            top_points = np.where(top_region > 0)
            if len(top_points[0]) > 0:
                min_y = np.min(top_points[0])
                
                # 상단을 더 확장 (헤어 영역)
                expand_pixels = height // 8  # 이미지 높이의 1/8만큼 확장
                new_min_y = max(0, min_y - expand_pixels)
                
                # 확장된 영역을 마스크에 추가
                expanded_mask = mask.copy()
                
                # 상단 확장 영역에서 어두운 픽셀 찾기 (헤어)
                top_expand_region = image[new_min_y:min_y, :]
                if top_expand_region.size > 0:
                    gray_region = cv2.cvtColor(top_expand_region, cv2.COLOR_BGR2GRAY)
                    
                    # 어두운 픽셀 마스크 (헤어로 추정)
                    dark_threshold = 100
                    dark_mask = gray_region < dark_threshold
                    
                    # 확장된 마스크에 추가
                    expanded_mask[new_min_y:min_y, :] = np.where(dark_mask, 255, expanded_mask[new_min_y:min_y, :])
                
                return expanded_mask
            
            return mask
            
        except Exception as e:
            print(f"헤어 영역 확장 오류: {e}")
            return mask
    
    def _create_advanced_cutout(self, image, mask):
        """고급 누끼 생성"""
        try:
            if mask is None:
                return None
            
            # 흰색 배경 생성
            white_background = np.ones_like(image) * 255
            
            # 마스크를 3채널로 변환 (0-1 범위로 정규화)
            mask_3ch = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR) / 255.0
            
            # 마스크 영역은 원본 이미지, 나머지는 흰색 배경
            result = image * mask_3ch + white_background * (1 - mask_3ch)
            result = result.astype(np.uint8)
            
            return result
            
        except Exception as e:
            print(f"고급 누끼 생성 오류: {e}")
            return None
            
    def _create_advanced_face_mask_with_selection(self):
        """선택된 영역 내에서 고급 얼굴 마스크 생성"""
        height, width = self.original_image.shape[:2]
        
        # 선택된 영역 좌표를 실제 이미지 좌표로 변환
        sel_x, sel_y, sel_w, sel_h = self.selection_rect
        
        # UI 이미지 크기와 실제 이미지 크기 비율 계산
        ui_width = 300  # UI에서 표시되는 이미지 너비
        ui_height = 300  # UI에서 표시되는 이미지 높이
        
        # 실제 이미지에서의 선택 영역 계산
        scale_x = width / ui_width
        scale_y = height / ui_height
        
        actual_x = int(sel_x * scale_x)
        actual_y = int(sel_y * scale_y)
        actual_w = int(sel_w * scale_x)
        actual_h = int(sel_h * scale_y)
        
        # 경계 확인
        actual_x = max(0, min(actual_x, width - 1))
        actual_y = max(0, min(actual_y, height - 1))
        actual_w = min(actual_w, width - actual_x)
        actual_h = min(actual_h, height - actual_y)
        
        print(f"실제 선택 영역: x={actual_x}, y={actual_y}, w={actual_w}, h={actual_h}")
        
        # 전체 이미지 마스크 초기화
        mask = np.zeros((height, width), dtype=np.uint8)
        
        # 방법 1: 선택된 영역에서 얼굴 감지 시도
        roi = self.original_image[actual_y:actual_y+actual_h, actual_x:actual_x+actual_w]
        roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        
        faces = []
        
        # 방법 1: MediaPipe로 얼굴 감지 시도 (최우선순위)
        if self.mp_loaded:
            try:
                faces = self.detect_faces_mediapipe(roi, confidence_threshold=0.3)
                print(f"MediaPipe로 선택 영역에서 감지된 얼굴 수: {len(faces)}")
            except Exception as e:
                print(f"MediaPipe 얼굴 감지 오류: {e}")
        
        # 방법 2: DNN 모델로 얼굴 감지 시도 (우선순위 2)
        if len(faces) == 0 and self.dnn_loaded:
            try:
                faces = self.detect_faces_dnn(roi, confidence_threshold=0.3)
                print(f"DNN으로 선택 영역에서 감지된 얼굴 수: {len(faces)}")
            except Exception as e:
                print(f"DNN 얼굴 감지 오류: {e}")
        
        # 방법 3: Haar Cascade로 얼굴 감지 시도 (백업)
        if len(faces) == 0 and self.face_cascade is not None:
            try:
                # 더 민감한 설정으로 얼굴 감지
                faces = self.face_cascade.detectMultiScale(
                    roi_gray,
                    scaleFactor=1.03,  # 더 작은 스케일 변화
                    minNeighbors=2,    # 더 적은 이웃 요구
                    minSize=(20, 20),  # 더 작은 최소 크기
                    maxSize=(0, 0)     # 최대 크기 제한 없음
                )
                print(f"Haar Cascade로 선택 영역에서 감지된 얼굴 수: {len(faces)}")
            except Exception as e:
                print(f"Haar Cascade 얼굴 감지 오류: {e}")
        
        if len(faces) > 0:
            # 첫 번째 얼굴 사용
            fx, fy, fw, fh = faces[0]
            
            # 실제 이미지 좌표로 변환
            face_x = actual_x + fx
            face_y = actual_y + fy
            face_w = fw
            face_h = fh
            
            print(f"감지된 얼굴 위치: x={face_x}, y={face_y}, w={face_w}, h={face_h}")
            
            # 얼굴 영역을 확장 (헤어 포함) - 더 큰 확장
            expand_ratio_x = 0.6  # 좌우 60% 확장
            expand_ratio_y_top = 0.8  # 상단 80% 확장 (헤어 포함)
            expand_ratio_y_bottom = 0.3  # 하단 30% 확장
            
            expand_x = int(face_w * expand_ratio_x)
            expand_y_top = int(face_h * expand_ratio_y_top)
            expand_y_bottom = int(face_h * expand_ratio_y_bottom)
            
            face_x = max(0, face_x - expand_x)
            face_y = max(0, face_y - expand_y_top)  # 상단을 더 많이 확장
            face_w = min(width - face_x, face_w + 2 * expand_x)
            face_h = min(height - face_y, face_h + expand_y_top + expand_y_bottom)
            
            # 고급 얼굴 윤곽선 마스크 생성
            mask = self._create_advanced_face_contour_mask(face_x, face_y, face_w, face_h)
            
        else:
            # 방법 2: 전체 이미지에서 얼굴 감지 시도
            print("선택 영역에서 얼굴 감지 실패, 전체 이미지에서 감지 시도")
            gray = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)
            
            # 방법 2-1: MediaPipe로 전체 이미지에서 얼굴 감지 시도 (최우선순위)
            if self.mp_loaded:
                try:
                    faces = self.detect_faces_mediapipe(self.original_image, confidence_threshold=0.3)
                    print(f"MediaPipe로 전체 이미지에서 감지된 얼굴 수: {len(faces)}")
                except Exception as e:
                    print(f"MediaPipe 전체 이미지 얼굴 감지 오류: {e}")
            
            # 방법 2-2: DNN 모델로 전체 이미지에서 얼굴 감지 시도
            if len(faces) == 0 and self.dnn_loaded:
                try:
                    faces = self.detect_faces_dnn(self.original_image, confidence_threshold=0.3)
                    print(f"DNN으로 전체 이미지에서 감지된 얼굴 수: {len(faces)}")
                except Exception as e:
                    print(f"DNN 전체 이미지 얼굴 감지 오류: {e}")
            
            # 방법 2-3: Haar Cascade로 전체 이미지에서 얼굴 감지 시도 (백업)
            if len(faces) == 0 and self.face_cascade is not None:
                try:
                    faces = self.face_cascade.detectMultiScale(
                        gray,
                        scaleFactor=1.05,
                        minNeighbors=3,
                        minSize=(50, 50)
                    )
                    print(f"Haar Cascade로 전체 이미지에서 감지된 얼굴 수: {len(faces)}")
                except Exception as e:
                    print(f"Haar Cascade 전체 이미지 얼굴 감지 오류: {e}")
            
            if len(faces) > 0:
                # 첫 번째 얼굴 사용
                x, y, w, h = faces[0]
                print(f"전체 이미지에서 감지된 얼굴 위치: x={x}, y={y}, w={w}, h={h}")
                
                # 얼굴 영역을 확장 (헤어 포함) - 더 큰 확장
                expand_ratio_x = 0.6  # 좌우 60% 확장
                expand_ratio_y_top = 0.8  # 상단 80% 확장 (헤어 포함)
                expand_ratio_y_bottom = 0.3  # 하단 30% 확장
                
                expand_x = int(w * expand_ratio_x)
                expand_y_top = int(h * expand_ratio_y_top)
                expand_y_bottom = int(h * expand_ratio_y_bottom)
                
                x = max(0, x - expand_x)
                y = max(0, y - expand_y_top)  # 상단을 더 많이 확장
                w = min(width - x, w + 2 * expand_x)
                h = min(height - y, h + expand_y_top + expand_y_bottom)
                
                mask = self._create_advanced_face_contour_mask(x, y, w, h)
                
            else:
                # 방법 3: 선택된 영역을 타원형으로 마스킹 (얼굴이 감지되지 않은 경우)
                print("얼굴 감지 완전 실패, 선택 영역을 타원형으로 마스킹")
                center_x = actual_x + actual_w // 2
                center_y = actual_y + actual_h // 2
                
                # 선택 영역을 약간 확장하여 더 자연스럽게
                expand_ratio = 0.2  # 20% 확장
                axes_x = int(actual_w * (1 + expand_ratio) // 2)
                axes_y = int(actual_h * (1 + expand_ratio) // 2)
                
                cv2.ellipse(mask, (center_x, center_y), (axes_x, axes_y), 0, 0, 360, 255, -1)
                mask = cv2.GaussianBlur(mask, (31, 31), 0)
        
        return mask
        
    def _create_advanced_face_mask_full_image(self):
        """전체 이미지에서 고급 얼굴 마스크 생성"""
        height, width = self.original_image.shape[:2]
        gray = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)
        
        # 얼굴 감지
        faces = []
        
        # 방법 1: MediaPipe로 얼굴 감지 시도 (최우선순위)
        if self.mp_loaded:
            try:
                faces = self.detect_faces_mediapipe(self.original_image, confidence_threshold=0.3)
                print(f"MediaPipe로 전체 이미지에서 감지된 얼굴 수: {len(faces)}")
            except Exception as e:
                print(f"MediaPipe 얼굴 감지 오류: {e}")
        
        # 방법 2: DNN 모델로 얼굴 감지 시도 (우선순위 2)
        if len(faces) == 0 and self.dnn_loaded:
            try:
                faces = self.detect_faces_dnn(self.original_image, confidence_threshold=0.3)
                print(f"DNN으로 전체 이미지에서 감지된 얼굴 수: {len(faces)}")
            except Exception as e:
                print(f"DNN 얼굴 감지 오류: {e}")
        
        # 방법 3: Haar Cascade로 얼굴 감지 시도 (백업)
        if len(faces) == 0 and self.face_cascade is not None:
            try:
                faces = self.face_cascade.detectMultiScale(
                    gray,
                    scaleFactor=1.05,
                    minNeighbors=3,
                    minSize=(50, 50)
                )
                print(f"Haar Cascade로 전체 이미지에서 감지된 얼굴 수: {len(faces)}")
            except Exception as e:
                print(f"Haar Cascade 얼굴 감지 오류: {e}")
        
        mask = np.zeros((height, width), dtype=np.uint8)
        
        if len(faces) > 0:
            # 첫 번째 얼굴 사용
            x, y, w, h = faces[0]
            
            # 얼굴 영역을 확장 (헤어 포함) - 더 큰 확장
            expand_ratio_x = 0.6  # 좌우 60% 확장
            expand_ratio_y_top = 0.8  # 상단 80% 확장 (헤어 포함)
            expand_ratio_y_bottom = 0.3  # 하단 30% 확장
            
            expand_x = int(w * expand_ratio_x)
            expand_y_top = int(h * expand_ratio_y_top)
            expand_y_bottom = int(h * expand_ratio_y_bottom)
            
            x = max(0, x - expand_x)
            y = max(0, y - expand_y_top)  # 상단을 더 많이 확장
            w = min(width - x, w + 2 * expand_x)
            h = min(height - y, h + expand_y_top + expand_y_bottom)
            
            mask = self._create_advanced_face_contour_mask(x, y, w, h)
        else:
            # 얼굴이 감지되지 않으면 중앙 타원형 마스킹 (더 큰 크기)
            center_x, center_y = width // 2, height // 2
            face_width = int(width * 0.6)  # 40%에서 60%로 증가
            face_height = int(height * 0.6)  # 40%에서 60%로 증가
            
            cv2.ellipse(mask, (center_x, center_y), (face_width // 2, face_height // 2), 0, 0, 360, 255, -1)
            mask = cv2.GaussianBlur(mask, (31, 31), 0)
        
        return mask
        
    def _create_advanced_face_contour_mask(self, x, y, w, h):
        """고급 얼굴 윤곽선 마스크 생성 (헤어 포함, 옷 제외)"""
        height, width = self.original_image.shape[:2]
        mask = np.zeros((height, width), dtype=np.uint8)
        
        # 얼굴 영역 추출
        face_roi = self.original_image[y:y+h, x:x+w]
        face_gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
        
        # 1. 다중 방법을 사용한 얼굴 윤곽선 감지
        contours = self._detect_face_contours(face_roi, face_gray)
        
        if contours:
            # 여러 윤곽선을 결합하여 더 완전한 마스크 생성
            valid_contours = []
            
            # 모든 윤곽선을 검토하여 유효한 것들만 선택
            for contour in contours:
                area = cv2.contourArea(contour)
                min_area = (w * h) * 0.05  # 최소 면적 조건 완화
                max_area = (w * h) * 1.5   # 최대 면적 조건 완화
                
                if min_area < area < max_area:
                    valid_contours.append(contour)
            
            if valid_contours:
                # 가장 큰 윤곽선을 기본으로 사용
                main_contour = max(valid_contours, key=cv2.contourArea)
                
                # 윤곽선을 부드럽게 만들기 (더 정밀하게)
                epsilon = 0.005 * cv2.arcLength(main_contour, True)
                smoothed_contour = cv2.approxPolyDP(main_contour, epsilon, True)
                
                # 윤곽선을 실제 이미지 좌표로 변환
                contour_points = smoothed_contour.reshape(-1, 2)
                contour_points[:, 0] += x
                contour_points[:, 1] += y
                
                # 메인 윤곽선을 마스크에 그리기
                cv2.fillPoly(mask, [contour_points], 255)
                
                # 다른 큰 윤곽선들도 추가 (얼굴의 다른 부분일 수 있음)
                for contour in valid_contours:
                    if cv2.contourArea(contour) > cv2.contourArea(main_contour) * 0.3:
                        epsilon = 0.005 * cv2.arcLength(contour, True)
                        smoothed_contour = cv2.approxPolyDP(contour, epsilon, True)
                        contour_points = smoothed_contour.reshape(-1, 2)
                        contour_points[:, 0] += x
                        contour_points[:, 1] += y
                        cv2.fillPoly(mask, [contour_points], 255)
                
                print(f"윤곽선 기반 마스크 생성 성공 ({len(valid_contours)}개 윤곽선 사용)")
            else:
                # 윤곽선이 적합하지 않으면 타원형 마스크 사용
                mask = self._create_ellipse_mask_with_hair(x, y, w, h)
                print("윤곽선 부적합, 타원형 마스크 사용")
        else:
            # 윤곽선이 없으면 타원형 마스크 사용
            mask = self._create_ellipse_mask_with_hair(x, y, w, h)
            print("윤곽선 감지 실패, 타원형 마스크 사용")
        
        # 마스크를 부드럽게 처리
        mask = cv2.GaussianBlur(mask, (15, 15), 0)
        
        return mask
        
    def _detect_face_contours(self, face_roi, face_gray):
        """다중 방법을 사용한 얼굴 윤곽선 감지"""
        all_contours = []
        
        # 방법 1: 다중 스케일 Canny 엣지 감지
        try:
            blurred = cv2.GaussianBlur(face_gray, (5, 5), 0)
            edges1 = cv2.Canny(blurred, 30, 100)
            edges2 = cv2.Canny(blurred, 50, 150)
            edges3 = cv2.Canny(blurred, 80, 200)
            combined_edges = cv2.bitwise_or(cv2.bitwise_or(edges1, edges2), edges3)
            
            # 고급 모폴로지 연산으로 엣지 연결
            kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            combined_edges = cv2.morphologyEx(combined_edges, cv2.MORPH_CLOSE, kernel_close)
            
            contours, _ = cv2.findContours(combined_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            all_contours.extend(contours)
        except:
            pass
        
        # 방법 2: 다중 적응형 임계값
        try:
            adaptive_thresh1 = cv2.adaptiveThreshold(face_gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
            adaptive_thresh2 = cv2.adaptiveThreshold(face_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
            adaptive_combined = cv2.bitwise_or(adaptive_thresh1, adaptive_thresh2)
            adaptive_combined = cv2.bitwise_not(adaptive_combined)  # 반전
            
            # 고급 모폴로지 연산
            kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            adaptive_combined = cv2.morphologyEx(adaptive_combined, cv2.MORPH_CLOSE, kernel_close)
            
            contours, _ = cv2.findContours(adaptive_combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            all_contours.extend(contours)
        except:
            pass
        
        # 방법 3: 다중 색상 공간 피부색 감지
        try:
            # HSV 색공간
            face_hsv = cv2.cvtColor(face_roi, cv2.COLOR_BGR2HSV)
            lower_skin_hsv = np.array([0, 20, 70], dtype=np.uint8)
            upper_skin_hsv = np.array([20, 255, 255], dtype=np.uint8)
            skin_mask_hsv = cv2.inRange(face_hsv, lower_skin_hsv, upper_skin_hsv)
            
            # YCrCb 색공간
            face_ycrcb = cv2.cvtColor(face_roi, cv2.COLOR_BGR2YCrCb)
            lower_skin_ycrcb = np.array([0, 135, 85], dtype=np.uint8)
            upper_skin_ycrcb = np.array([255, 180, 135], dtype=np.uint8)
            skin_mask_ycrcb = cv2.inRange(face_ycrcb, lower_skin_ycrcb, upper_skin_ycrcb)
            
            # 피부색 마스크 결합
            skin_mask_combined = cv2.bitwise_or(skin_mask_hsv, skin_mask_ycrcb)
            
            # 고급 모폴로지 연산
            kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
            kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            skin_mask_combined = cv2.morphologyEx(skin_mask_combined, cv2.MORPH_CLOSE, kernel_close)
            skin_mask_combined = cv2.morphologyEx(skin_mask_combined, cv2.MORPH_OPEN, kernel_open)
            
            contours, _ = cv2.findContours(skin_mask_combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            all_contours.extend(contours)
        except:
            pass
        
        # 방법 4: GrabCut 알고리즘 (고급 배경 분할)
        try:
            if face_roi.shape[0] > 50 and face_roi.shape[1] > 50:
                # GrabCut을 위한 마스크 초기화
                gc_mask = np.zeros(face_gray.shape[:2], np.uint8)
                
                # 중앙 영역을 전경으로 설정
                center_x, center_y = face_gray.shape[1] // 2, face_gray.shape[0] // 2
                gc_mask[center_y-20:center_y+20, center_x-20:center_x+20] = cv2.GC_FGD
                
                # 배경 모델과 전경 모델
                bgd_model = np.zeros((1, 65), np.float64)
                fgd_model = np.zeros((1, 65), np.float64)
                
                # GrabCut 실행
                cv2.grabCut(face_roi, gc_mask, None, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_MASK)
                
                # 전경 마스크 생성
                gc_mask2 = np.where((gc_mask == 2) | (gc_mask == 0), 0, 1).astype('uint8')
                
                contours, _ = cv2.findContours(gc_mask2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                all_contours.extend(contours)
        except:
            pass
        
        # 방법 5: 눈과 코 감지로 얼굴 구조 파악
        try:
            eye_mask = np.zeros(face_gray.shape, np.uint8)
            nose_mask = np.zeros(face_gray.shape, np.uint8)
            
            if self.eye_cascade is not None:
                eyes = self.eye_cascade.detectMultiScale(face_gray, 1.1, 3)
                for (ex, ey, ew, eh) in eyes:
                    cv2.rectangle(eye_mask, (ex, ey), (ex+ew, ey+eh), 255, -1)
            
            if self.nose_cascade is not None:
                noses = self.nose_cascade.detectMultiScale(face_gray, 1.1, 3)
                for (nx, ny, nw, nh) in noses:
                    cv2.rectangle(nose_mask, (nx, ny), (nx+nw, ny+nh), 255, -1)
            
            # 눈과 코 마스크에서 윤곽선 추출
            contours, _ = cv2.findContours(eye_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            all_contours.extend(contours)
            contours, _ = cv2.findContours(nose_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            all_contours.extend(contours)
        except:
            pass
        
        return all_contours
        
    def _select_best_face_contour(self, contours, face_width, face_height):
        """가장 적합한 얼굴 윤곽선 선택"""
        if not contours:
            return None
            
        best_contour = None
        best_score = 0
        
        for contour in contours:
            # 윤곽선 면적 계산
            area = cv2.contourArea(contour)
            
            # 너무 작거나 큰 윤곽선 제외
            min_area = (face_width * face_height) * 0.1
            max_area = (face_width * face_height) * 0.9
            
            if area < min_area or area > max_area:
                continue
                
            # 윤곽선의 종횡비 계산
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = w / h
            
            # 얼굴의 일반적인 종횡비 범위 (더 넓은 범위로 확장)
            if aspect_ratio < 0.4 or aspect_ratio > 2.0:
                continue
                
            # 윤곽선의 원형도 계산
            perimeter = cv2.arcLength(contour, True)
            if perimeter == 0:
                continue
                
            circularity = 4 * np.pi * area / (perimeter * perimeter)
            
            # 점수 계산 (면적 + 원형도)
            score = area * 0.7 + circularity * 1000 * 0.3
            
            if score > best_score:
                best_score = score
                best_contour = contour
                
        return best_contour
        
    def _create_ellipse_mask_with_hair(self, x, y, w, h):
        """헤어를 포함한 타원형 마스크 생성"""
        height, width = self.original_image.shape[:2]
        mask = np.zeros((height, width), dtype=np.uint8)
        
        # 얼굴 영역을 확장 (헤어 포함) - 더 큰 확장
        expand_ratio_x = 0.7  # 좌우 70% 확장으로 증가
        expand_ratio_y_top = 0.9  # 상단 90% 확장으로 증가 (헤어 포함)
        expand_ratio_y_bottom = 0.3  # 하단 30% 확장으로 증가
        
        expand_x = int(w * expand_ratio_x)
        expand_y_top = int(h * expand_ratio_y_top)
        expand_y_bottom = int(h * expand_ratio_y_bottom)
        
        new_x = max(0, x - expand_x)
        new_y = max(0, y - expand_y_top)
        new_w = min(width - new_x, w + 2 * expand_x)
        new_h = min(height - new_y, h + expand_y_top + expand_y_bottom)
        
        # 타원형 마스크 생성
        center = (new_x + new_w // 2, new_y + new_h // 2)
        axes = (new_w // 2, new_h // 2)
        cv2.ellipse(mask, center, axes, 0, 0, 360, 255, -1)
        
        return mask
            
    def _update_ui_after_processing(self):
        """처리 완료 후 UI 업데이트"""
        try:
            # 처리된 이미지 표시
            if self.processed_image is not None:
                print(f"처리된 이미지 정보: {self.processed_image.shape}")
                
                # BGR을 RGB로 변환하여 표시 (이미 흰색 배경으로 처리됨)
                rgb_image = cv2.cvtColor(self.processed_image, cv2.COLOR_BGR2RGB)
                print("흰색 배경으로 처리된 이미지 표시")
                
                # PIL 이미지로 변환
                pil_image = Image.fromarray(rgb_image)
                
                # 이미지 크기 조정
                display_size = (300, 300)
                pil_image.thumbnail(display_size, Image.Resampling.LANCZOS)
                
                # Tkinter용 이미지로 변환
                tk_image = ImageTk.PhotoImage(pil_image)
                
                # 처리된 이미지 표시
                self.processed_label.configure(image=tk_image, text="")
                self.processed_label.image = tk_image
                
                # 버튼 활성화
                self.save_btn.configure(state=tk.NORMAL)
                self.status_label.configure(text="이미지 처리 완료! 저장할 수 있습니다.")
                
                print("UI 업데이트 완료")
            else:
                self.status_label.configure(text="이미지 처리 실패")
                
        except Exception as e:
            print(f"UI 업데이트 오류: {e}")
            self.status_label.configure(text=f"UI 업데이트 실패: {str(e)}")
            
        finally:
            # UI 재활성화
            self.select_btn.configure(state=tk.NORMAL)
            self.select_area_btn.configure(state=tk.NORMAL)
            self.process_btn.configure(state=tk.NORMAL)
            self.progress.stop()
            
    def _handle_processing_error(self, error_msg):
        """처리 오류 처리"""
        messagebox.showerror("처리 오류", f"이미지 처리 중 오류가 발생했습니다:\n{error_msg}")
        self.status_label.configure(text="이미지 처리 실패")
        
        # UI 재활성화
        self.select_btn.configure(state=tk.NORMAL)
        self.select_area_btn.configure(state=tk.NORMAL)
        self.process_btn.configure(state=tk.NORMAL)
        self.progress.stop()
        
    def save_result(self):
        """처리된 이미지 저장"""
        if self.processed_image is None:
            return
            
        # 저장할 파일 경로 선택
        file_path = filedialog.asksaveasfilename(
            title="결과 이미지 저장",
            defaultextension=".png",
            filetypes=[
                ("PNG 파일", "*.png"),
                ("JPEG 파일", "*.jpg"),
                ("모든 파일", "*.*")
            ]
        )
        
        if file_path:
            try:
                # 파일 확장자 확인
                ext = os.path.splitext(file_path)[1].lower()
                
                if ext == '.png':
                    # PNG로 저장 (흰색 배경)
                    rgb_image = cv2.cvtColor(self.processed_image, cv2.COLOR_BGR2RGB)
                    pil_image = Image.fromarray(rgb_image)
                    pil_image.save(file_path, 'PNG')
                    
                elif ext in ['.jpg', '.jpeg']:
                    # JPEG로 저장 (흰색 배경)
                    cv2.imwrite(file_path, self.processed_image)
                    
                else:
                    # 기본적으로 PNG로 저장 (흰색 배경)
                    rgb_image = cv2.cvtColor(self.processed_image, cv2.COLOR_BGR2RGB)
                    pil_image = Image.fromarray(rgb_image)
                    pil_image.save(file_path, 'PNG')
                
                self.status_label.configure(text=f"이미지 저장 완료: {os.path.basename(file_path)}")
                messagebox.showinfo("완료", "이미지가 성공적으로 저장되었습니다!")
                
            except Exception as e:
                messagebox.showerror("저장 오류", f"이미지 저장 실패: {str(e)}")
                self.status_label.configure(text="이미지 저장 실패")
                
    def run(self):
        """프로그램 실행"""
        self.root.mainloop()

def main():
    """메인 함수"""
    try:
        app = FaceCutoutTool()
        app.run()
    except Exception as e:
        messagebox.showerror("프로그램 오류", f"프로그램 실행 중 오류가 발생했습니다:\n{str(e)}")

if __name__ == "__main__":
    main()  