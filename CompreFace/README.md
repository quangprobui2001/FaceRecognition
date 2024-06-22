# FACE RECOGNITION

## A. Installation
1. Docker desktop: 
- Link download: https://www.docker.com/products/docker-desktop/
2. CompreFace (phiên bản 1.1.0)
- Link download: https://github.com/exadel-inc/CompreFace/releases
- Phần Source code (zip) mục Assets
3. Python
- Link download: https://www.python.org/downloads/
4. Các thư viện cần thiết (Command Window)
- Thư viện OpenCV, Pandas và Compreface-sdk
```bash
pip install opencv-python
pip install pandas
pip install compreface-sdk
```
6. Running CompreFace:
- Running docker
- Mở command, chuyển đường dẫn tới thư mục CompreFace (Ví dụ anh unzip file ở desktop (thay đường dẫn của anh vào đó):
```bash
cd C:\Users\ACER\Desktop\CompreFace-1.1.0
```
- Running CompreFace
```bash
docker-compose up -d
```
7. Chạy chương trình nhận dạng khuôn mặt:
- Mở Command window, chuyển đường dẫn tới source code (sử dụng câu lệnh: cd directory của source code), thực hiện câu lệnh: php test.php
```bash
cd directory của source code
php test.php
```

## B. USAGE
1. Code
```bash
Log_infor3.py
```
2. Nguyên lý hoạt động
- Webcam sẽ khởi chạy, chuyển từng frame thành dạng ảnh JPG và đẩy từng ảnh này lên CompreFace.
- CompreFace sẽ thực hiện nhận dạng khuôn mặt dựa trên các ảnh đó, sau đó sẽ gửi kết quả nhận dạng lại và hiển thị lên window webcam.
- Thông tin nhận dạng được sẽ được log lại vào file
```bash
recognition_log.csv
```
