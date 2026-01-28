# Phân Loại Đánh Giá Phim IMDB với Transformer

Dự án này triển khai một mô hình dựa trên kiến trúc Transformer để phân loại cảm xúc (Tích cực/Tiêu cực) trên bộ dữ liệu IMDB Movie Review sử dụng PyTorch.

## Cấu Trúc Dự Án

- `dataset.py`: Xử lý tải dữ liệu, tiền xử lý và chuẩn bị `IMDBDataModule`.
- `model.py`: Định nghĩa kiến trúc mô hình Transformer bao gồm `PositionalEncoding`, `TransformerEncoderBlock`, và `TransformerClassifier`.
- `train.py`: Script chính để huấn luyện mô hình, đánh giá và lưu các checkpoint.
- `requirements.txt`: Danh sách các thư viện Python cần thiết.

## Yêu Cầu

Cài đặt các thư viện cần thiết bằng pip:

```bash
pip install -r requirements.txt
```

## Dữ Liệu

Dự án yêu cầu bộ dữ liệu IMDB ở định dạng CSV (`IMDB Dataset.csv`). Hãy đảm bảo bạn đã có bộ dữ liệu này. Đường dẫn mặc định trong `train.py` có thể cần được điều chỉnh hoặc truyền vào như một tham số nếu tệp không nằm ở vị trí mặc định.

## Cách Sử Dụng

### Huấn Luyện Mô Hình

Bạn có thể bắt đầu huấn luyện bằng cách chạy `train.py`. Bạn có thể điều chỉnh các siêu tham số thông qua các tham số dòng lệnh.

```bash
python train.py --csv-path "IMDB Dataset.csv" --epochs 20 --batch-size 64
```

**Các tham số:**

- `--csv-path`: Đường dẫn đến tệp CSV dữ liệu IMDB.
- `--epochs`: Số lượng epoch huấn luyện (mặc định: 20).
- `--batch-size`: Kích thước batch cho huấn luyện (mặc định: 64).
- `--num-words`: Kích thước tập từ vựng (mặc định: 500).
- `--maxlen`: Độ dài tối đa của chuỗi (mặc định: 500).
- `--logging`: Thư mục chứa log TensorBoard (mặc định: "tensorboard").
- `--trained-models`: Thư mục để lưu các mô hình đã huấn luyện (mặc định: "trained-models").
- `--checkpoint`: Đường dẫn đến checkpoint để tiếp tục huấn luyện (tùy chọn).

## Theo Dõi Log

Tiến độ huấn luyện và các chỉ số được ghi lại bằng TensorBoard. Bạn có thể xem log bằng cách chạy:

```bash
tensorboard --logdir tensorboard
```

## Kiến Trúc Mô Hình

Mô hình bao gồm:
1.  **Lớp Embedding**: Chuyển đổi các chỉ số token thành các vector dense.
2.  **Positional Encoding**: Thêm thông tin vị trí vào các embedding.
3.  **Transformer Encoder Blocks**: Multi-head attention và mạng feed-forward với normalization và dropout.
4.  **Global Average Pooling**: Tổng hợp chuỗi các vector.
5.  **Classifier Head**: Các lớp fully connected để dự đoán cảm xúc.
