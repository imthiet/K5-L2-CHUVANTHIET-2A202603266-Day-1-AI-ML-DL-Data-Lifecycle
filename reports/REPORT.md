# Báo cáo bài thực hành Ngày 1 – Đọc prediction YOLO11 để hiểu định dạng nhãn

**Ngày chạy:** 06/11/2026

**Runtime Colab:** CPU/GPU

**Python / PyTorch / Ultralytics:**

**Checkpoint:** `yolo11n-cls.pt`, `yolo11n.pt`, `yolo11n-seg.pt`

**Ngưỡng detection/segmentation dùng khi tạo JSON:** `0.35` (chỉ đổi nếu GV cho phép; nếu đổi hãy ghi rõ)

**Thay đổi so với notebook nguồn:** Không / mô tả rõ thay đổi

> **Cách dùng mẫu này:** copy/điền trên máy, lưu thành `REPORT.md`, rồi **upload** vào panel Files
> của Colab (notebook không tự tạo file này). Ô cuối Colab sẽ đóng ZIP cùng `day1_lab_outputs/`.
> Giải nén và đưa vào `report/` trên GitHub, rồi nộp **link repo** trên LMS.
> Không ghi họ tên, MSSV, email, số điện thoại hoặc dữ liệu cá nhân khác vào báo cáo/output;
> họ tên/MSSV chỉ nằm trong tên repository.

### Từ cần nhớ (đọc nhanh trước khi viết)

| Từ | Nghĩa ngắn |
| --- | --- |
| Prediction | Kết quả model vừa đoán ra; chưa phải nhãn đúng |
| Ground truth | Nhãn do người gán theo quy tắc |
| Taxonomy | Danh sách tên lớp có sẵn kèm model (vd. ImageNet-1K, COCO-80) |
| Checkpoint | File model `.pt` đã huấn luyện sẵn |
| Guideline | Quy tắc gán nhãn do người viết |
| Annotator | Người gán nhãn |
| Reviewer / QC | Người kiểm tra nhãn |
| Escalation | Ca khó → hỏi Lab Coach, không tự bịa |
| Model score | Số “tin” của model; không phải điểm chất lượng nhãn người |

## 1. Phân loại ảnh

**Mở file:** `day1_lab_outputs/classification_predictions.json` và `visuals/classification_top5.png`.  
Chỉ dùng ảnh `sample_id = traffic`, lấy dòng `rank = 1`.

1. Chép record hạng 1: `class_id`=468 ; `class_name`=cab ; `rank`=1 ; `score`=0.510915 ; `taxonomy_name`=ImageNet-1K
2. Record này nói về **cả bức ảnh**, vì sample_id là "traffic"
3. Danh sách tên lớp (class list) do đâu mà có: model tự nghĩ ra, hay đi kèm taxonomy/checkpoint?
Class list đi kèm taxônmy/checkpoint
4. Vì sao cần ghi cả `class_id`, `class_name` và `taxonomy_name`, không chỉ ghi mỗi tên lớp?
Vì id, name, taxonomy là thứ để xác định bức ảnh, vật thể, ghi mỗi tên lớp là không đủ để xác định.
5. Nếu trong ảnh có nhiều vật / nhiều chủ đề, guideline cần nói rõ điều gì để người gán nhãn chọn **một** lớp?
Guideline cần quy định quy tắc ưu tiên (priority rule) để người gán nhãn chỉ chọn một lớp duy nhất
6. Vì sao không được lấy `score` của model làm ground truth?
Vì score chỉ thể hiện mức độ tự tin của model đối với dự đoán, không đảm bảo dự đoán đó đúng
## 2. Phát hiện vật thể

**Mở file:** `day1_lab_outputs/detection_predictions.json` và `visuals/detection_predictions.png`.  
Chỉ dùng ảnh `sample_id = kitchen`. Chọn **một** record.  
Số lượng prediction theo ngưỡng: xem output ô so sánh `0.20` / `0.35` / `0.60` trong notebook.

1. Chép một record: `class_name`=person ; `score`=0.912625 ; `bbox_xyxy`=[385.33,
      69.24,
      498.92,
      348.92] ; `bbox_width`=113.58 ; `bbox_height`=279.68
2. Giải thích box bằng lời thường: góc trên-trái ở đâu, góc dưới-phải ở đâu, rộng/cao bao nhiêu pixel?
Góc trên trái nằm ở phía trên bên trái của Box, tương tự với góc dưới phải. rộng cao là con số thể hiện ở bbox_width và bbox_height.Ví dụ với clas_name = person là :  bbox_width`=113.58 ; `bbox_height`=279.68
3. Chọn hai ngưỡng (vd. 0.20 và 0.60). Mỗi ngưỡng có bao nhiêu prediction?
Ngưỡng 0.20 là: 17 vật thể
ngưỡng 0.60 là 6 vật thể
4. Khi hạ ngưỡng (giữ nhiều hộp hơn), người **reviewer** phải làm thêm việc gì?
Kiểm tra thêm các hộp dự đoán có cònident thấp, loại bỏ các hộp sai, loại bỏ or chỉnh sửa nahnx lớp,kiểm tra bounding box và check lại.
5. Viết **một quy tắc box** để người gán nhãn làm theo (áp dụng được cho nhiều ảnh, không chỉ ảnh này).
Vẽ một bounding box nhỏ nhất có thể nhưng phải bao trọn toàn bộ vật thể cần gán nhãn, không cắt mất bất kỳ bộ phận nào của vật thể và không bao gồm quá nhiều phần nền xung quanh.
6. Trên ảnh `kitchen`, chỉ ra **một** vật bị che khuất hoặc cắt mép. Bạn sẽ viết thành quy tắc trong guideline, hay đưa lên hỏi Lab Coach (escalation)? Vì sao?
Vì chưa chắc chắn và không được đánh nhãn bừa bãi.
## 3. Phân đoạn theo từng đối tượng

**Mở file:** `day1_lab_outputs/segmentation_predictions.json` và `visuals/segmentation_prediction.png`.  
Chỉ dùng ảnh `sample_id = kitchen`. Chọn **một** record.

1. Chép một record: `instance_id`=kitchen-001 ; `class_name`=person ; `score`=0.899318 ; số điểm polygon=`polygon_point_count`=348 ; 3–5 điểm đầu của `polygon_xy`=[
        446.0,
        70.0
      ],
      [
        445.0,
        71.0
      ],
      [
        444.0,
        71.0
      ],
2. So với box, polygon cho thêm thông tin gì về hình dạng vật thể?
Polygon cung cấp thông tin về đường biên và hình dạng thực tế của vật thể (contour), chi tiết hơn bounding box.
- Box: Chỉ xác định vùng hình chữ nhật bao quanh vật thể.
- Polygon: Xác định đường viền, hình dạng và diện tích chính xác hơn, kể cả những vật thể không có hình dạng vuông vức.
3. `instance_id` dùng để làm gì? Nó **không** phải class ID và **không** phải tracking ID — bạn hiểu câu này thế nào?
instance_id dùng để định danh duy nhất một đối tượng cụ thể (instance) trong một ảnh hoặc mẫu dữ liệu, giúp phân biệt các đối tượng cùng thuộc một class.
4. Viết **một quy tắc vẽ biên mask** để người gán nhãn làm theo.
Vẽ mask sát theo đường viền thực tế của vật thể, bao phủ toàn bộ phần nhìn thấy được, không bao gồm nền xung quanh và không tự ý lấp các khoảng trống hoặc vùng bị che khuất.
5. Trên ảnh `kitchen`, chỉ ra **một** chỗ biên mờ / hai vật chạm nhau / bị che. Bạn dùng guideline hay escalation? Vì sao?
Person và bowl, dùng guildline.
## 4. Vòng đời và kiểm tra chất lượng

Lab hôm nay chỉ **xem prediction**. Chuỗi dưới đây là vòng đời production để bạn nối các khái niệm:

`ảnh thô → guideline → ground truth → huấn luyện → prediction → QC/rework`

Điền bảng (viết ngắn, có dẫn chứng từ JSON/PNG khi nói về lỗi):

| Tác vụ | Ground truth trông như thế nào? (đơn vị nhãn) | Lỗi hoặc chỗ khó quyết định bạn thấy trên ảnh/JSON | Annotator làm gì? | Reviewer xem gì? |
| --- | --- | --- | --- | --- |
| Phân loại ảnh |  |  |  |  |
| Phát hiện vật thể |  |  |  |  |
| Instance segmentation |  |  |  |  |
|---|---|---|---|---|
| Phân loại ảnh (Image Classification) | Một hoặc nhiều class label gán cho toàn bộ ảnh. | Ảnh có nhiều đối tượng, dễ nhầm lẫn giữa các lớp hoặc khó xác định lớp chính. | Quan sát toàn ảnh, chọn nhãn phù hợp theo guideline. | Kiểm tra nhãn có đúng nội dung ảnh, đúng taxonomy và nhất quán không. |
| Phát hiện vật thể (Object Detection) | Bounding Box (tọa độ x1, y1, x2, y2) và class label cho từng vật thể. | Box có thể lệch, thiếu vật thể nhỏ hoặc vật thể bị che khuất. | Vẽ bounding box bao sát từng vật thể và gán class tương ứng. | Kiểm tra box có bao đủ vật thể, đúng vị trí, đúng class và không bỏ sót. |
| Instance Segmentation | Polygon/Mask (vùng pixel) riêng cho từng instance, kèm class label. | JSON traffic-001 có polygon 120 điểm; vật thể traffic-010 nằm sát mép trái ảnh (x = 0), cần kiểm tra mask có bị cắt thiếu không. | Tạo mask riêng cho từng vật thể, chỉnh đường viền sát hình dạng thực tế. | Kiểm tra mask, ranh giới, instance, class và các vùng bị thiếu hoặc chồng lấn. |
## 5. An toàn dữ liệu

1. Quy tắc bảo vệ dữ liệu của bạn cho bài lab này là gì? (viết 1 câu)
Không chia sẻ ảnh, dữ liệu JSON hoặc thông tin nhận diện của bài lab ra bên ngoài,chỉ sử dụng dữ liệu đúng mục đích và tuân thủ quyền truy cập được cấp.
2. Nếu gặp ảnh hoặc dữ liệu không đúng phạm vi, bạn sẽ dừng và báo cho ai?
Báo cho mentor hoặc người trực tiếp đứng lớp.
## 6. Checklist trước khi nộp link repo

Kiểm tra trên GitHub trong thư mục `report/`:

- [ ] `REPORT.md` (file này đã điền)
- [ ] `day1_lab_outputs/classification_predictions.json`
- [ ] `day1_lab_outputs/detection_predictions.json`
- [ ] `day1_lab_outputs/segmentation_predictions.json`
- [ ] `day1_lab_outputs/IMAGE_ATTRIBUTION.md`
- [ ] `day1_lab_outputs/visuals/classification_top5.png`
- [ ] `day1_lab_outputs/visuals/detection_predictions.png`
- [ ] `day1_lab_outputs/visuals/segmentation_prediction.png`
- [ ] Ô validation cuối notebook báo `PASS`
- [ ] Không có họ tên, MSSV hoặc dữ liệu nhạy cảm trong báo cáo/output
- [ ] Đã nộp **link repository** trên LMS
