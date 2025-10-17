#!/usr/bin/env python3
"""
AI Lens Helper - GUI Application
YOLO+CLIP 기반 전시품 인식 시스템 GUI
"""

import os
import sys
import threading
from pathlib import Path
from tkinter import *
from tkinter import ttk, filedialog, messagebox, scrolledtext
from PIL import Image, ImageTk
import io

# Fix OpenMP library conflict on macOS
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['OMP_NUM_THREADS'] = '1'

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from ai_lens_helper.models.yolo_clip import YOLOCLIPPipeline
from ai_lens_helper.infer.clip_index import CLIPIndexBuilder
from ai_lens_helper import Lens


class LensGUI:
    """AI Lens Helper GUI Application."""

    def __init__(self, root):
        self.root = root
        self.root.title("AI Lens Helper - YOLO+CLIP 전시품 인식")
        self.root.geometry("1200x800")

        # State
        self.current_image = None
        self.current_image_path = None
        self.lens = None

        self._setup_ui()

    def _setup_ui(self):
        """Setup UI components."""
        # Create notebook (tabs)
        notebook = ttk.Notebook(self.root)
        notebook.pack(fill=BOTH, expand=True, padx=10, pady=10)

        # Tab 1: 인덱스 빌드 (학습)
        self.train_frame = Frame(notebook)
        notebook.add(self.train_frame, text="📚 인덱스 빌드 (학습)")
        self._setup_train_tab()

        # Tab 2: Inference (테스트)
        self.infer_frame = Frame(notebook)
        notebook.add(self.infer_frame, text="🔍 Inference (테스트)")
        self._setup_infer_tab()

        # Tab 3: OpenAI Vision (테스트)
        self.openai_frame = Frame(notebook)
        notebook.add(self.openai_frame, text="🤖 OpenAI Vision")
        self._setup_openai_tab()

    def _setup_train_tab(self):
        """Setup training tab."""
        # Input section
        input_frame = LabelFrame(self.train_frame, text="설정", padx=10, pady=10)
        input_frame.pack(fill=X, padx=10, pady=10)

        # Data root
        Label(input_frame, text="데이터 루트:").grid(row=0, column=0, sticky=W, pady=5)
        self.train_data_root = StringVar(value="./data")
        Entry(input_frame, textvariable=self.train_data_root, width=50).grid(row=0, column=1, pady=5, padx=5)
        Button(input_frame, text="찾아보기", command=self._browse_data_root).grid(row=0, column=2, pady=5)

        # Place
        Label(input_frame, text="장소 (Place):").grid(row=1, column=0, sticky=W, pady=5)
        self.train_place = StringVar(value="경복궁")
        Entry(input_frame, textvariable=self.train_place, width=50).grid(row=1, column=1, pady=5, padx=5)

        # Save path
        Label(input_frame, text="저장 경로:").grid(row=2, column=0, sticky=W, pady=5)
        self.train_save_path = StringVar(value="./indexes/index")
        Entry(input_frame, textvariable=self.train_save_path, width=50).grid(row=2, column=1, pady=5, padx=5)
        Button(input_frame, text="찾아보기", command=self._browse_save_path).grid(row=2, column=2, pady=5)

        # Device
        Label(input_frame, text="Device:").grid(row=3, column=0, sticky=W, pady=5)
        self.train_device = StringVar(value="cpu")
        device_combo = ttk.Combobox(input_frame, textvariable=self.train_device, values=["cpu", "cuda"], width=47)
        device_combo.grid(row=3, column=1, pady=5, padx=5)
        device_combo.state(["readonly"])

        # Model selection
        Label(input_frame, text="YOLO 모델:").grid(row=4, column=0, sticky=W, pady=5)
        self.train_yolo_model = StringVar(value="yolov8n.pt")
        yolo_combo = ttk.Combobox(
            input_frame,
            textvariable=self.train_yolo_model,
            values=["yolov8n.pt", "yolov8s.pt", "yolov8m.pt", "yolov8l.pt"],
            width=47
        )
        yolo_combo.grid(row=4, column=1, pady=5, padx=5)
        yolo_combo.state(["readonly"])

        # Build button
        self.build_button = Button(
            input_frame,
            text="🚀 인덱스 빌드 시작",
            command=self._start_build,
            bg="#4CAF50",
            fg="white",
            font=("Arial", 12, "bold"),
            height=2
        )
        self.build_button.grid(row=5, column=0, columnspan=3, pady=20, sticky=EW)

        # Progress
        self.build_progress = ttk.Progressbar(input_frame, mode='indeterminate')
        self.build_progress.grid(row=6, column=0, columnspan=3, pady=5, sticky=EW)

        # Log section
        log_frame = LabelFrame(self.train_frame, text="로그", padx=10, pady=10)
        log_frame.pack(fill=BOTH, expand=True, padx=10, pady=10)

        self.train_log = scrolledtext.ScrolledText(log_frame, height=20, state=DISABLED)
        self.train_log.pack(fill=BOTH, expand=True)

    def _setup_infer_tab(self):
        """Setup inference tab."""
        # Left panel: Controls
        left_panel = Frame(self.infer_frame)
        left_panel.pack(side=LEFT, fill=Y, padx=10, pady=10)

        # Model loading
        model_frame = LabelFrame(left_panel, text="모델 로드", padx=10, pady=10)
        model_frame.pack(fill=X, pady=10)

        Label(model_frame, text="인덱스 파일:").grid(row=0, column=0, sticky=W, pady=5)
        self.infer_model_path = StringVar(value="./indexes/경복궁_clip.json")
        Entry(model_frame, textvariable=self.infer_model_path, width=30).grid(row=0, column=1, pady=5)
        Button(model_frame, text="찾아보기", command=self._browse_model).grid(row=0, column=2, pady=5, padx=5)

        Button(
            model_frame,
            text="모델 로드",
            command=self._load_model,
            bg="#2196F3",
            fg="white",
            font=("Arial", 10, "bold")
        ).grid(row=1, column=0, columnspan=3, pady=10, sticky=EW)

        self.model_status = Label(model_frame, text="모델 미로드", fg="red")
        self.model_status.grid(row=2, column=0, columnspan=3)

        # Image selection
        image_frame = LabelFrame(left_panel, text="이미지 선택", padx=10, pady=10)
        image_frame.pack(fill=X, pady=10)

        Button(
            image_frame,
            text="📷 이미지 선택",
            command=self._browse_image,
            font=("Arial", 12),
            height=2
        ).pack(fill=X, pady=5)

        self.image_path_label = Label(image_frame, text="이미지 없음", wraplength=250)
        self.image_path_label.pack(pady=5)

        # Inference settings
        settings_frame = LabelFrame(left_panel, text="설정", padx=10, pady=10)
        settings_frame.pack(fill=X, pady=10)

        Label(settings_frame, text="Place:").grid(row=0, column=0, sticky=W, pady=5)
        self.infer_place = StringVar(value="경복궁")
        Entry(settings_frame, textvariable=self.infer_place, width=20).grid(row=0, column=1, pady=5)

        Label(settings_frame, text="Top-K:").grid(row=1, column=0, sticky=W, pady=5)
        self.infer_topk = IntVar(value=5)
        Spinbox(settings_frame, from_=1, to=10, textvariable=self.infer_topk, width=18).grid(row=1, column=1, pady=5)

        Label(settings_frame, text="Threshold:").grid(row=2, column=0, sticky=W, pady=5)
        self.infer_threshold = DoubleVar(value=0.7)
        Spinbox(
            settings_frame,
            from_=0.0,
            to=1.0,
            increment=0.05,
            textvariable=self.infer_threshold,
            width=18,
            format="%.2f"
        ).grid(row=2, column=1, pady=5)

        # Infer button
        self.infer_button = Button(
            left_panel,
            text="🔍 Inference 실행",
            command=self._run_inference,
            bg="#FF9800",
            fg="white",
            font=("Arial", 12, "bold"),
            height=2,
            state=DISABLED
        )
        self.infer_button.pack(fill=X, pady=20)

        # Right panel: Results
        right_panel = Frame(self.infer_frame)
        right_panel.pack(side=RIGHT, fill=BOTH, expand=True, padx=10, pady=10)

        # Image display
        image_display_frame = LabelFrame(right_panel, text="입력 이미지", padx=10, pady=10)
        image_display_frame.pack(fill=BOTH, expand=True, pady=10)

        self.image_canvas = Canvas(image_display_frame, width=500, height=400, bg="gray")
        self.image_canvas.pack()

        # Results display
        results_frame = LabelFrame(right_panel, text="Inference 결과", padx=10, pady=10)
        results_frame.pack(fill=BOTH, expand=True, pady=10)

        self.results_text = scrolledtext.ScrolledText(results_frame, height=10, state=DISABLED)
        self.results_text.pack(fill=BOTH, expand=True)

    # Training tab callbacks
    def _browse_data_root(self):
        path = filedialog.askdirectory(title="데이터 루트 선택")
        if path:
            self.train_data_root.set(path)

    def _browse_save_path(self):
        path = filedialog.asksaveasfilename(
            title="저장 경로 선택",
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        if path:
            # Remove extension for saving
            path = str(Path(path).with_suffix(""))
            self.train_save_path.set(path)

    def _log_train(self, message):
        """Add message to training log."""
        self.train_log.config(state=NORMAL)
        self.train_log.insert(END, message + "\n")
        self.train_log.see(END)
        self.train_log.config(state=DISABLED)
        self.root.update()

    def _start_build(self):
        """Start index building in background thread."""
        def build_thread():
            try:
                self.build_button.config(state=DISABLED)
                self.build_progress.start()

                self._log_train("=" * 60)
                self._log_train("인덱스 빌드 시작")
                self._log_train("=" * 60)

                data_root = Path(self.train_data_root.get())
                place = self.train_place.get()
                save_path = Path(self.train_save_path.get())
                device = self.train_device.get()
                yolo_model = self.train_yolo_model.get()

                self._log_train(f"데이터 루트: {data_root}")
                self._log_train(f"장소: {place}")
                self._log_train(f"저장 경로: {save_path}")
                self._log_train(f"Device: {device}")
                self._log_train(f"YOLO 모델: {yolo_model}")
                self._log_train("")

                # Initialize pipeline
                self._log_train("YOLO+CLIP 파이프라인 초기화 중...")
                pipeline = YOLOCLIPPipeline(
                    yolo_model=yolo_model,
                    clip_model="ViT-B-16",
                    clip_pretrained="openai",
                    device=device
                )
                self._log_train("✓ 파이프라인 초기화 완료")
                self._log_train("")

                # Build index
                self._log_train(f"이미지 처리 시작: {data_root}/{place}")
                builder = CLIPIndexBuilder(pipeline=pipeline)

                # Custom build with logging
                place_dir = data_root / place
                if not place_dir.exists():
                    raise ValueError(f"장소 디렉토리를 찾을 수 없습니다: {place_dir}")

                index_items = {}
                image_extensions = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

                for item_dir in sorted(place_dir.iterdir()):
                    if not item_dir.is_dir():
                        continue

                    item_name = item_dir.name
                    image_files = [
                        f for f in item_dir.iterdir()
                        if f.is_file() and f.suffix.lower() in image_extensions
                    ]

                    if not image_files:
                        self._log_train(f"⚠️  {item_name}: 이미지 없음 (건너뜀)")
                        continue

                    self._log_train(f"처리 중: {item_name} ({len(image_files)}개 이미지)...")

                    embeddings_list = []
                    for img_path in image_files:
                        try:
                            embedding = pipeline.process_image(img_path)
                            embeddings_list.append(embedding)
                        except Exception as e:
                            self._log_train(f"  ⚠️  {img_path.name} 실패: {e}")
                            continue

                    if not embeddings_list:
                        self._log_train(f"⚠️  {item_name}: 유효한 임베딩 없음 (건너뜀)")
                        continue

                    import numpy as np
                    embeddings = np.stack(embeddings_list, axis=0)
                    mean_embedding = np.mean(embeddings, axis=0)
                    mean_embedding = mean_embedding / np.linalg.norm(mean_embedding)

                    from ai_lens_helper.infer.clip_index import IndexItem
                    index_items[item_name] = IndexItem(
                        item_name=item_name,
                        place=place,
                        embeddings=embeddings,
                        mean_embedding=mean_embedding,
                        image_count=len(embeddings_list)
                    )

                    self._log_train(f"✓ {item_name}: {len(embeddings_list)}개 완료")

                if not index_items:
                    raise ValueError("인덱싱할 아이템이 없습니다!")

                self._log_train("")
                self._log_train("인덱스 저장 중...")
                builder.save_index(
                    index_items=index_items,
                    output_path=save_path,
                    place=place,
                    metadata={"reject_threshold": 0.7}
                )

                total_images = sum(item.image_count for item in index_items.values())

                self._log_train("")
                self._log_train("=" * 60)
                self._log_train("✓ 인덱스 빌드 완료!")
                self._log_train("=" * 60)
                self._log_train(f"장소: {place}")
                self._log_train(f"전시품 수: {len(index_items)}개")
                self._log_train(f"총 이미지 수: {total_images}개")
                self._log_train(f"저장 위치: {save_path}.json / {save_path}.faiss")
                self._log_train("=" * 60)

                messagebox.showinfo("완료", f"인덱스 빌드가 완료되었습니다!\n\n전시품: {len(index_items)}개\n이미지: {total_images}개")

            except Exception as e:
                self._log_train(f"\n❌ 오류 발생: {e}")
                import traceback
                self._log_train(traceback.format_exc())
                messagebox.showerror("오류", f"인덱스 빌드 중 오류가 발생했습니다:\n\n{e}")
            finally:
                self.build_progress.stop()
                self.build_button.config(state=NORMAL)

        threading.Thread(target=build_thread, daemon=True).start()

    # Inference tab callbacks
    def _browse_model(self):
        path = filedialog.askopenfilename(
            title="인덱스 파일 선택",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        if path:
            self.infer_model_path.set(path)

    def _browse_image(self):
        path = filedialog.askopenfilename(
            title="이미지 선택",
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.bmp *.webp"),
                ("All files", "*.*")
            ]
        )
        if path:
            self.current_image_path = Path(path)
            self.image_path_label.config(text=f".../{self.current_image_path.name}")
            self._display_image(self.current_image_path)

            # Enable inference if model is loaded
            if self.lens is not None:
                self.infer_button.config(state=NORMAL)

    def _display_image(self, image_path):
        """Display image on canvas."""
        try:
            img = Image.open(image_path)
            # Resize to fit canvas
            img.thumbnail((500, 400), Image.Resampling.LANCZOS)
            photo = ImageTk.PhotoImage(img)

            self.image_canvas.delete("all")
            self.image_canvas.create_image(250, 200, image=photo)
            self.image_canvas.image = photo  # Keep reference
        except Exception as e:
            messagebox.showerror("오류", f"이미지 로드 실패:\n{e}")

    def _load_model(self):
        """Load inference model."""
        try:
            model_path = Path(self.infer_model_path.get())
            if not model_path.exists():
                messagebox.showerror("오류", "인덱스 파일을 찾을 수 없습니다.")
                return

            self.model_status.config(text="로딩 중...", fg="orange")
            self.root.update()

            self.lens = Lens(model_path=model_path, device="cpu")

            self.model_status.config(text=f"✓ 모델 로드 완료 ({model_path.name})", fg="green")

            # Enable inference if image is selected
            if self.current_image_path is not None:
                self.infer_button.config(state=NORMAL)

            messagebox.showinfo("완료", "모델이 성공적으로 로드되었습니다!")

        except Exception as e:
            self.model_status.config(text="✗ 로드 실패", fg="red")
            messagebox.showerror("오류", f"모델 로드 실패:\n{e}")

    def _run_inference(self):
        """Run inference on selected image."""
        if self.lens is None:
            messagebox.showerror("오류", "먼저 모델을 로드해주세요.")
            return

        if self.current_image_path is None:
            messagebox.showerror("오류", "먼저 이미지를 선택해주세요.")
            return

        def infer_thread():
            try:
                self.infer_button.config(state=DISABLED)
                self._update_results("Inference 실행 중...\n")

                place = self.infer_place.get()
                topk = self.infer_topk.get()
                threshold = self.infer_threshold.get()

                result = self.lens.infer(
                    place=place,
                    image_path=self.current_image_path,
                    reject_threshold=threshold,
                    topk=topk
                )

                # Format results
                output = []
                output.append("=" * 60)
                output.append("INFERENCE 결과")
                output.append("=" * 60)
                output.append(f"이미지: {self.current_image_path.name}")
                output.append(f"장소: {place}")
                output.append("")
                output.append(f"📋 결정: {result.decision.upper()}")
                output.append(f"🎯 선택된 전시품: {result.selected_item}")
                output.append(f"📊 신뢰도: {result.confidence:.4f}")
                output.append("")
                output.append(f"💬 메시지: {result.message}")
                output.append("")
                output.append(f"🏆 Top-{topk} 예측:")
                for i, pred in enumerate(result.predictions, 1):
                    icon = "✓" if i == 1 else " "
                    output.append(f"  {icon} {i}. {pred.item}: {pred.score:.4f}")

                if result.hints:
                    output.append("")
                    output.append("💡 개선 힌트:")
                    for hint in result.hints:
                        output.append(f"  • {hint}")

                output.append("")
                output.append("=" * 60)

                self._update_results("\n".join(output))

                # Show message box for accept/recollect
                if result.decision == "accept":
                    messagebox.showinfo(
                        "✓ Accept",
                        f"전시품: {result.selected_item}\n신뢰도: {result.confidence:.2%}"
                    )
                else:
                    messagebox.showwarning(
                        "⚠ Recollect",
                        f"신뢰도가 낮습니다 ({result.confidence:.2%})\n\n다시 촬영해주세요."
                    )

            except Exception as e:
                self._update_results(f"\n❌ 오류 발생:\n{e}\n")
                import traceback
                self._update_results(traceback.format_exc())
                messagebox.showerror("오류", f"Inference 실패:\n{e}")
            finally:
                self.infer_button.config(state=NORMAL)

        threading.Thread(target=infer_thread, daemon=True).start()

    def _update_results(self, text):
        """Update results text widget."""
        self.results_text.config(state=NORMAL)
        self.results_text.delete(1.0, END)
        self.results_text.insert(END, text)
        self.results_text.config(state=DISABLED)
        self.root.update()

    def _setup_openai_tab(self):
        """Setup OpenAI Vision tab."""
        # Left panel: Controls
        left_panel = Frame(self.openai_frame)
        left_panel.pack(side=LEFT, fill=Y, padx=10, pady=10)

        # API Key
        api_frame = LabelFrame(left_panel, text="OpenAI API 설정", padx=10, pady=10)
        api_frame.pack(fill=X, pady=10)

        Label(api_frame, text="API Key:").grid(row=0, column=0, sticky=W, pady=5)
        self.openai_api_key = StringVar(value="")
        Entry(api_frame, textvariable=self.openai_api_key, width=30, show="*").grid(row=0, column=1, pady=5)

        Label(api_frame, text="Model:").grid(row=1, column=0, sticky=W, pady=5)
        self.openai_model = StringVar(value="gpt-5")
        model_combo = ttk.Combobox(
            api_frame,
            textvariable=self.openai_model,
            values=["gpt-5", "gpt-5-mini", "gpt-5-nano", "gpt-4.1", "gpt-4.1-nano", "gpt-4.1-mini", "gpt-4o", "gpt-4o-mini", "gpt-4-vision-preview"],
            width=27
        )
        model_combo.grid(row=1, column=1, pady=5)

        Label(api_frame, text="Place:").grid(row=2, column=0, sticky=W, pady=5)
        self.openai_place = StringVar(value="경복궁")
        Entry(api_frame, textvariable=self.openai_place, width=30).grid(row=2, column=1, pady=5)

        # Image selection
        image_frame2 = LabelFrame(left_panel, text="이미지 선택", padx=10, pady=10)
        image_frame2.pack(fill=X, pady=10)

        Button(
            image_frame2,
            text="📷 이미지 선택",
            command=self._browse_openai_image,
            font=("Arial", 12),
            height=2
        ).pack(fill=X, pady=5)

        self.openai_image_label = Label(image_frame2, text="이미지 없음", wraplength=250)
        self.openai_image_label.pack(pady=5)

        # Items list
        items_frame = LabelFrame(left_panel, text="항목 리스트 (한 줄에 하나씩)", padx=10, pady=10)
        items_frame.pack(fill=BOTH, expand=True, pady=10)

        self.items_list = scrolledtext.ScrolledText(items_frame, height=10, width=30)
        self.items_list.pack(fill=BOTH, expand=True)

        # Default items
        default_items = '''근정전
경회루
향원정
강녕전
교태전'''
        self.items_list.insert(END, default_items)

        # Infer button
        self.openai_button = Button(
            left_panel,
            text="🤖 OpenAI Vision 실행",
            command=self._run_openai_vision,
            bg="#9C27B0",
            fg="white",
            font=("Arial", 12, "bold"),
            height=2
        )
        self.openai_button.pack(fill=X, pady=20)

        # Right panel: Results
        right_panel = Frame(self.openai_frame)
        right_panel.pack(side=RIGHT, fill=BOTH, expand=True, padx=10, pady=10)

        # Image display
        image_display_frame2 = LabelFrame(right_panel, text="입력 이미지", padx=10, pady=10)
        image_display_frame2.pack(fill=BOTH, expand=True, pady=10)

        self.openai_canvas = Canvas(image_display_frame2, width=500, height=400, bg="gray")
        self.openai_canvas.pack()

        # Results display
        results_frame2 = LabelFrame(right_panel, text="OpenAI Vision 결과", padx=10, pady=10)
        results_frame2.pack(fill=BOTH, expand=True, pady=10)

        self.openai_results = scrolledtext.ScrolledText(results_frame2, height=10, state=DISABLED)
        self.openai_results.pack(fill=BOTH, expand=True)

    def _browse_openai_image(self):
        """Browse image for OpenAI Vision."""
        path = filedialog.askopenfilename(
            title="이미지 선택",
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.bmp *.webp"),
                ("All files", "*.*")
            ]
        )
        if path:
            self.openai_image_path = Path(path)
            self.openai_image_label.config(text=f".../{self.openai_image_path.name}")
            self._display_openai_image(self.openai_image_path)

    def _display_openai_image(self, image_path):
        """Display image on OpenAI canvas."""
        try:
            img = Image.open(image_path)
            img.thumbnail((500, 400), Image.Resampling.LANCZOS)
            photo = ImageTk.PhotoImage(img)

            self.openai_canvas.delete("all")
            self.openai_canvas.create_image(250, 200, image=photo)
            self.openai_canvas.image = photo
        except Exception as e:
            messagebox.showerror("오류", f"이미지 로드 실패:\n{e}")

    def _run_openai_vision(self):
        """Run OpenAI Vision inference."""
        if not hasattr(self, 'openai_image_path'):
            messagebox.showerror("오류", "먼저 이미지를 선택해주세요.")
            return

        api_key = self.openai_api_key.get().strip()
        if not api_key:
            messagebox.showerror("오류", "OpenAI API Key를 입력해주세요.")
            return

        def openai_thread():
            try:
                self.openai_button.config(state=DISABLED)
                self._update_openai_results("OpenAI Vision 실행 중...\n")

                # Get place and items
                import json
                place = self.openai_place.get().strip()
                items_text = self.items_list.get(1.0, END).strip()
                items = [line.strip() for line in items_text.split('\n') if line.strip()]

                if not place:
                    raise ValueError("장소(Place)를 입력해주세요.")
                if not items:
                    raise ValueError("최소 1개 이상의 항목을 입력해주세요.")

                # Run inference
                from ai_lens_helper.infer.openai_vision import OpenAIVisionInference

                model = self.openai_model.get()
                vision = OpenAIVisionInference(api_key=api_key, model=model)

                result = vision.infer(
                    image_path=self.openai_image_path,
                    place=place,
                    items=items
                )

                # Format results
                output = []
                output.append("=" * 60)
                output.append("OpenAI Vision API 결과")
                output.append("=" * 60)
                output.append(f"이미지: {self.openai_image_path.name}")
                output.append(f"모델: {model}")
                output.append(f"장소: {place}")
                output.append(f"항목 수: {len(items)}개")

                # Display timing and cost info if available
                if "_metadata" in result:
                    meta = result["_metadata"]
                    output.append("")
                    if "api_time" in meta:
                        output.append(f"⏱️  API 시간: {meta['api_time']}초")
                    if "total_time" in meta:
                        output.append(f"⏱️  총 시간: {meta['total_time']}초")
                    if "input_tokens" in meta:
                        output.append(f"📥 Input Tokens: {meta['input_tokens']:,}")
                    if "output_tokens" in meta:
                        output.append(f"📤 Output Tokens: {meta['output_tokens']:,}")
                    if "total_tokens" in meta:
                        output.append(f"📊 Total Tokens: {meta['total_tokens']:,}")
                    if "cost_usd" in meta:
                        cost = meta['cost_usd']
                        output.append(f"💰 비용: ${cost:.6f} (≈ ₩{cost * 1400:.2f})")

                output.append("")

                if "error" in result:
                    output.append(f"❌ 오류: {result['error']}")
                elif "raw_response" in result:
                    output.append("⚠️  파싱 실패 - 원본 응답:")
                    output.append("")
                    output.append(f"'{result['raw_response']}'")
                    output.append("")
                    output.append(f"응답 길이: {len(result['raw_response'])} 글자")
                else:
                    output.append(f"🏛️  장소: {result.get('장소', 'N/A')}")
                    output.append(f"🏢  건물: {result.get('건물', 'N/A')}")
                    if "index" in result:
                        output.append(f"📍 인덱스: {result['index']}")

                    # Show success message
                    if result.get('장소') != "알 수 없음":
                        messagebox.showinfo(
                            "✓ 인식 성공",
                            f"장소: {result.get('장소')}\n건물: {result.get('건물')}"
                        )
                    else:
                        messagebox.showwarning(
                            "⚠ 인식 실패",
                            "해당하는 전시품을 찾을 수 없습니다."
                        )

                output.append("")
                output.append("=" * 60)

                self._update_openai_results("\n".join(output))

            except json.JSONDecodeError as e:
                self._update_openai_results(f"\n❌ JSON 파싱 오류:\n{e}\n")
                messagebox.showerror("오류", f"JSON 데이터가 올바르지 않습니다:\n{e}")
            except Exception as e:
                self._update_openai_results(f"\n❌ 오류 발생:\n{e}\n")
                import traceback
                self._update_openai_results(traceback.format_exc())
                messagebox.showerror("오류", f"OpenAI Vision 실행 실패:\n{e}")
            finally:
                self.openai_button.config(state=NORMAL)

        threading.Thread(target=openai_thread, daemon=True).start()

    def _update_openai_results(self, text):
        """Update OpenAI results text widget."""
        self.openai_results.config(state=NORMAL)
        self.openai_results.delete(1.0, END)
        self.openai_results.insert(END, text)
        self.openai_results.config(state=DISABLED)
        self.root.update()


def main():
    """Main entry point."""
    root = Tk()
    app = LensGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
