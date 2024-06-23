Chương trình xây dựng nhằm mục đích phục vụ bài toán Face Recognition với các kho lưu trữ sau:
#  1.CompreFace
Sử dụng API key CompreFace cho phép nhận diện khuôn mặt. Cách cài đặt và hướng dẫn vui lòng xem tại đây: https://github.com/quangprobui2001/FaceRecognition/tree/main/CompreFace#face-recognition
# 2.Fine-tune Model theo repo facenet-tensorflow
Link repo hướng dẫn: https://github.com/davidsandberg/facenet
Mô hình được tác giả lấy cảm hứng từ bài báo: Center loss based on the paper "A Discriminative Feature Learning Approach for Deep Face Recognition"

Mô hình sẽ trả ra vector 512 chiều và sử dụng SVM để phân loại khuôn mặt. Link lấy mô hình trọng số để pre-train: https://drive.google.com/drive/folders/1CyWreovc2naOYxeSQ5Ckme2PWvsjYAhh?usp=sharing
Tạo thư mục Dataset trong using_mtcnn_centerLoss, trong đó tạo tiếp thư mục FaceData và dưới FaceData là tạo tiếp 2 thư mục raw và processed.
Tạo thư mục Models trong using_mtcnn_centerLoss để lưu model.
Chú ý các args trong các file để thực hiện.
# 3 Sử dụng BlazeFace và phương pháp Shallow Face Learning & FaceX-Zoo
