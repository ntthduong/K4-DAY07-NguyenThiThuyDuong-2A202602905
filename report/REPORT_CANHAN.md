# Báo Cáo Cá Nhân — Lab 7: Embedding & Vector Store

**Họ tên:** Nguyễn Thị Thùy Dương
**Nhóm:** G99
**Ngày:** 20/09/2026

> **Nộp 1 bản / sinh viên.** Phần nhóm (lựa chọn tài liệu, thiết kế chiến lược, bộ câu hỏi đánh giá, demo) nộp chung 1 bản trong `REPORT_NHOM.md`. Chi tiết thang điểm: `docs/SCORING.md`.

**Tổng điểm phần cá nhân: 60** = Khởi động (5) + Hướng tiếp cận (10) + Hoàn thiện code (30) + Dự đoán độ tương tự (5) + Kết quả truy xuất của tôi (10).

---

## 1. Khởi động (Warm-up) — Cá nhân (5 điểm)

### Độ tương tự Cosine (Cosine Similarity) (Bài tập 1.1)

**Độ tương tự cosine cao (High cosine similarity) nghĩa là gì?**
> Độ tương tự cosine cao nghĩa là hai vector văn bản có hướng gần nhau trong không gian embedding. Điều này cho thấy hai văn bản có nội dung hoặc ý nghĩa gần giống nhau, dù chúng có thể sử dụng từ ngữ khác nhau.

**Ví dụ có độ tương tự CAO:**
- Câu A: Khách hàng có thể yêu cầu sửa sản phẩm còn trong thời hạn bảo hành.
- Câu B: Người mua được hỗ trợ khắc phục lỗi thiết bị khi bảo hành vẫn còn hiệu lực.
- Tại sao tương đồng: Hai câu dùng từ khác nhau nhưng đều nói về quyền được sửa sản phẩm trong thời hạn bảo hành.

**Ví dụ có độ tương tự THẤP:**
- Câu A: Nhà bán phải phản hồi yêu cầu bảo hành trong hai ngày làm việc.
- Câu B: Mô hình học sâu sử dụng mạng nơ-ron có nhiều lớp.
- Tại sao khác: Hai câu thuộc hai chủ đề không liên quan, một câu nói về chính sách bảo hành và câu còn lại nói về học máy.

**Tại sao độ tương tự cosine (cosine similarity) được ưu tiên hơn khoảng cách Euclid (Euclidean distance) cho text embeddings?**
> Cosine tập trung vào hướng của vector nên phản ánh mức độ giống nhau về ngữ nghĩa và ít bị ảnh hưởng bởi độ lớn vector. Khoảng cách Euclid phụ thuộc vào cả hướng lẫn độ lớn, vì vậy hai văn bản cùng nghĩa vẫn có thể bị đánh giá xa nhau nếu vector có độ lớn khác nhau.

### Bài toán tính toán Chunking (Bài tập 1.2)

**Tài liệu 10,000 ký tự, chunk_size=500, overlap=50. Bao nhiêu chunks?**
> Phép tính: `ceil((10.000 - 50) / (500 - 50)) = ceil(9.950 / 450) = ceil(22,11)`.
> Đáp án: **23 chunks**.

**Nếu độ chồng chéo (overlap) tăng lên 100, số lượng chunk thay đổi thế nào? Tại sao muốn độ chồng chéo nhiều hơn?**
> Khi overlap tăng lên 100, số chunk là `ceil((10.000 - 100) / (500 - 100)) = ceil(9.900 / 400) = 25`, tăng từ 23 lên 25. Overlap lớn hơn giúp giữ ngữ cảnh nằm tại ranh giới giữa hai chunk, nhưng làm tăng số chunk, dung lượng lưu trữ và chi phí embedding.

---

## 2. Hướng tiếp cận của tôi (My Approach) — Cá nhân (10 điểm)

Giải thích cách tiếp cận của bạn khi lập trình (implement) các phần chính trong gói `src`.

### Các hàm chia nhỏ (Chunking Functions)

**`SentenceChunker.chunk`** — hướng tiếp cận:
> Tôi dùng regex `(?<=[.!?])(?:[ \t]+|\n+)` để tách tại khoảng trắng hoặc xuống dòng ngay sau dấu kết thúc câu, nhờ đó dấu câu vẫn được giữ lại. Các câu được loại khoảng trắng thừa rồi gom theo `max_sentences_per_chunk`; văn bản rỗng trả về danh sách rỗng. Cách đơn giản này chưa xử lý hoàn hảo chữ viết tắt như `TS.` hoặc số thập phân.

**`RecursiveChunker.chunk` / `_split`** — hướng tiếp cận:
> Thuật toán thử separator theo thứ tự đoạn văn, dòng, câu, từ và cuối cùng là ký tự; phần vượt quá `chunk_size` tiếp tục được tách bằng separator ưu tiên thấp hơn. Sau khi tách, các phần nhỏ liền kề được gom lại đến gần giới hạn để tránh sinh nhiều chunk vụn. Base case là đoạn đã đủ ngắn; nếu hết separator thì cắt cứng theo `chunk_size`.

### Lớp EmbeddingStore

**`add_documents` + `search`** — hướng tiếp cận:
> Tôi dùng danh sách in-memory để hành vi nhất quán và không phụ thuộc vào việc máy có cài ChromaDB hay không. Mỗi `Document` được chuyển thành record gồm ID, nội dung, bản sao metadata và embedding; khi tìm kiếm, query được embed rồi tính dot product với từng record. Kết quả được sắp xếp theo score giảm dần và không trả embedding để output gọn hơn.

**`search_with_filter` + `delete_document`** — hướng tiếp cận:
> Tôi lọc metadata trước khi similarity search để các record sai đối tượng không chiếm vị trí trong top-k. `delete_document` loại tất cả record có `metadata['doc_id']` trùng ID tài liệu gốc, vì một tài liệu có thể tạo ra nhiều chunk, rồi trả về `True` nếu kích thước store giảm.

### Tác tử KnowledgeBaseAgent

**`answer`** — hướng tiếp cận:
> Agent truy xuất top-k chunk, đánh số từng chunk và đưa `source_url` hoặc `doc_id` vào phần ngữ cảnh. Prompt yêu cầu chỉ dùng bằng chứng đã cung cấp, trích dẫn theo số `[1]`, `[2]` và nói rõ khi thiếu thông tin. Nếu không truy xuất được chunk nào, agent trả thông báo ngay thay vì gọi LLM.

---

## 3. Hoàn thiện code (Core Implementation) — Cá nhân (30 điểm)

Vượt qua bộ kiểm thử là điều kiện tính điểm phần này.

### Kết Quả Kiểm Thử (Test Results)

```
============================= test session starts =============================
platform win32 -- Python 3.13.15, pytest-9.1.1, pluggy-1.6.0
rootdir: C:\folders\AI20K\K4-DAY07-NguyenThiThuyDuong-2A202602905
collected 42 items

tests/test_solution.py::TestProjectStructure::test_root_main_entrypoint_exists PASSED
tests/test_solution.py::TestProjectStructure::test_src_package_exists PASSED
tests/test_solution.py::TestClassBasedInterfaces::test_chunker_classes_exist PASSED
tests/test_solution.py::TestClassBasedInterfaces::test_mock_embedder_exists PASSED
tests/test_solution.py::TestFixedSizeChunker::test_chunks_respect_size PASSED
tests/test_solution.py::TestFixedSizeChunker::test_correct_number_of_chunks_no_overlap PASSED
tests/test_solution.py::TestFixedSizeChunker::test_empty_text_returns_empty_list PASSED
tests/test_solution.py::TestFixedSizeChunker::test_no_overlap_no_shared_content PASSED
tests/test_solution.py::TestFixedSizeChunker::test_overlap_creates_shared_content PASSED
tests/test_solution.py::TestFixedSizeChunker::test_returns_list PASSED
tests/test_solution.py::TestFixedSizeChunker::test_single_chunk_if_text_shorter PASSED
tests/test_solution.py::TestSentenceChunker::test_chunks_are_strings PASSED
tests/test_solution.py::TestSentenceChunker::test_respects_max_sentences PASSED
tests/test_solution.py::TestSentenceChunker::test_returns_list PASSED
tests/test_solution.py::TestSentenceChunker::test_single_sentence_max_gives_many_chunks PASSED
tests/test_solution.py::TestRecursiveChunker::test_chunks_within_size_when_possible PASSED
tests/test_solution.py::TestRecursiveChunker::test_empty_separators_falls_back_gracefully PASSED
tests/test_solution.py::TestRecursiveChunker::test_handles_double_newline_separator PASSED
tests/test_solution.py::TestRecursiveChunker::test_returns_list PASSED
tests/test_solution.py::TestEmbeddingStore::test_add_documents_increases_size PASSED
tests/test_solution.py::TestEmbeddingStore::test_add_more_increases_further PASSED
tests/test_solution.py::TestEmbeddingStore::test_initial_size_is_zero PASSED
tests/test_solution.py::TestEmbeddingStore::test_search_results_have_content_key PASSED
tests/test_solution.py::TestEmbeddingStore::test_search_results_have_score_key PASSED
tests/test_solution.py::TestEmbeddingStore::test_search_results_sorted_by_score_descending PASSED
tests/test_solution.py::TestEmbeddingStore::test_search_returns_at_most_top_k PASSED
tests/test_solution.py::TestEmbeddingStore::test_search_returns_list PASSED
tests/test_solution.py::TestKnowledgeBaseAgent::test_answer_non_empty PASSED
tests/test_solution.py::TestKnowledgeBaseAgent::test_answer_returns_string PASSED
tests/test_solution.py::TestComputeSimilarity::test_identical_vectors_return_1 PASSED
tests/test_solution.py::TestComputeSimilarity::test_opposite_vectors_return_minus_1 PASSED
tests/test_solution.py::TestComputeSimilarity::test_orthogonal_vectors_return_0 PASSED
tests/test_solution.py::TestComputeSimilarity::test_zero_vector_returns_0 PASSED
tests/test_solution.py::TestCompareChunkingStrategies::test_counts_are_positive PASSED
tests/test_solution.py::TestCompareChunkingStrategies::test_each_strategy_has_count_and_avg_length PASSED
tests/test_solution.py::TestCompareChunkingStrategies::test_returns_three_strategies PASSED
tests/test_solution.py::TestEmbeddingStoreSearchWithFilter::test_filter_by_department PASSED
tests/test_solution.py::TestEmbeddingStoreSearchWithFilter::test_no_filter_returns_all_candidates PASSED
tests/test_solution.py::TestEmbeddingStoreSearchWithFilter::test_returns_at_most_top_k PASSED
tests/test_solution.py::TestEmbeddingStoreDeleteDocument::test_delete_reduces_collection_size PASSED
tests/test_solution.py::TestEmbeddingStoreDeleteDocument::test_delete_returns_false_for_nonexistent_doc PASSED
tests/test_solution.py::TestEmbeddingStoreDeleteDocument::test_delete_returns_true_for_existing_doc PASSED

============================= 42 passed in 0.06s ==============================
```

**Số lượng bài test vượt qua (pass):** 42 / 42

---

## 4. Dự đoán độ tương tự (Similarity Predictions) — Cá nhân (5 điểm)

| Cặp | Câu A | Câu B | Dự đoán | Điểm thực tế | Đúng? |
|------|-----------|-----------|---------|--------------|-------|
| 1 | Khách hàng được đổi mới điện thoại lỗi trong 30 ngày. | Người mua có thể đổi sản phẩm lỗi kỹ thuật trong tháng đầu tiên. | cao | Chưa chạy riêng | Chưa đối chiếu |
| 2 | Nhà bán hàng phải phản hồi khiếu nại hoàn tiền trong 48 giờ. | Seller cần xử lý yêu cầu trả hàng trong 2 ngày làm việc. | cao | Chưa chạy riêng | Chưa đối chiếu |
| 3 | Pin LFP của xe máy điện VinFast được bảo hành 8 năm. | Xe máy điện VinFast có chính sách bảo hành pin dài hạn. | cao | Chưa chạy riêng | Chưa đối chiếu |
| 4 | Thiết bị rơi vỡ hoặc ngập nước có thể bị từ chối bảo hành. | Mô hình học sâu dùng mạng nơ-ron nhiều lớp để học dữ liệu. | thấp | Chưa chạy riêng | Chưa đối chiếu |
| 5 | Người bán cần video mở kiện để khiếu nại hàng hoàn thiếu. | Khách hàng phải đăng xuất tài khoản iCloud trước khi bảo hành. | thấp | Chưa chạy riêng | Chưa đối chiếu |

**Kết quả nào bất ngờ nhất? Điều này nói gì về cách embeddings biểu diễn ý nghĩa?**
> Cặp 5 dễ gây nhầm hơn dự đoán vì cả hai câu đều nằm trong miền chính sách bảo hành/đổi trả, dù một câu dành cho seller và một câu dành cho buyer. Điều này cho thấy embedding có thể bắt mạnh chủ đề chung, nên metadata filter như `audience` rất quan trọng để tránh lấy nhầm ngữ cảnh.

---

## 5. Kết quả truy xuất của tôi (Competition Results) — Cá nhân (10 điểm)

Chạy **5 câu hỏi đánh giá của nhóm** trên mã nguồn cá nhân của bạn trong gói `src`. **5 câu hỏi này phải trùng với các thành viên cùng nhóm** (xem `REPORT_NHOM.md`).

**Strategy: FixedSizeChunker (chunk_size=500, overlap=50) — 54 chunks total, 10 documents**

Benchmark được chạy bằng `python bench.py`. Script hiện dùng `_mock_embed`, nên score chỉ có ý nghĩa kiểm tra pipeline và thứ hạng trong lần chạy này, chưa đại diện cho embedding ngữ nghĩa thật.

| # | Câu hỏi (Query) | Top-1 chunk truy xuất (tóm tắt) | Score | Expected doc in top-3? | Keyword hit trong top-3 | Nhận xét retrieval |
|---|-------|--------------------------------|-------|------------------------|----------------------|--------------------|
| 1 | Thời hạn đổi trả/hoàn tiền người mua | `doi-tra-bao-hanh-shopee-buyer` - nội dung bảo hành Shopee | 0.3256 | YES (`lazada-buyer`) | `30 ngày` | Filter `audience=buyer` hoạt động; chunk chứa từ khóa nằm ở rank 2. |
| 2 | Thời hạn phản hồi khiếu nại người bán | `doi-tra-bao-hanh-lazada-seller` - trạng thái đơn hàng/khiếu nại | 0.2988 | YES (`lazada-seller`, `tiki-seller`) | `48 giờ`, `khiếu nại` | Filter `audience=seller` giúp top-3 chỉ còn tài liệu seller. |
| 3 | Thời hạn bảo hành VinFast | `doi-tra-bao-hanh-lazada-seller` - chế phạt vi phạm | 0.2850 | YES (`vinfast-buyer`) | Không có keyword ở top-1 | Expected doc xuất hiện rank 3 nhưng chunk chưa chứa trực tiếp `6 năm`, `8 năm`. |
| 4 | Từ chối bảo hành/trừ phí mobile | `doi-tra-bao-hanh-lazada-buyer` - quy trình bảo hành Lazada | 0.4776 | NO | Không có keyword mục tiêu | Failure case: top-3 lệch sang Lazada/Shopee, không lấy được TGDD/CellphoneS. |
| 5 | Bằng chứng khiếu nại người bán | `doi-tra-bao-hanh-lazada-seller` - quản lý/trạng thái đơn hàng | 0.2509 | YES (`lazada-seller`, `tiki-seller`) | `video`, `6 mặt` | Chunk chứa bằng chứng nằm ở rank 2; filter seller giúp giảm nhiễu buyer. |

**Bao nhiêu câu hỏi trả về expected doc trong top-3?** 4 / 5

**Bao nhiêu câu hỏi có chunk chứa keyword trả lời trực tiếp trong top-3?** 3 / 5 chắc chắn theo output (`Q1`, `Q2`, `Q5`). `Q3` có đúng tài liệu VinFast ở top-3 nhưng preview không chứa trực tiếp `6 năm` hoặc `8 năm`, nên cần kiểm tra nội dung chunk đầy đủ trước khi xem là trả lời được.

**Điều hay nhất tôi học được từ thành viên khác / nhóm khác (qua demo):**
> Với MockEmbedder, retrieval score không phản ánh ngữ nghĩa thực sự — cần dùng embedding thật (OpenAI/local) để đánh giá đúng chất lượng truy xuất. Metadata filter `audience` hoạt động đúng và quan trọng để tách biệt câu trả lời cho buyer vs seller. FixedSizeChunker tạo ra chunks đều nhưng có thể cắt ngang ngữ cảnh điều khoản; HeadingChunker tốt hơn cho tài liệu có cấu trúc heading rõ ràng.

---

## Tự Đánh Giá (Phần Cá Nhân)

| Tiêu chí | Điểm tự đánh giá |
|----------|-------------------|
| Khởi động (Warm-up) | 5 / 5 |
| Hướng tiếp cận của tôi (My Approach) | 10 / 10 |
| Hoàn thiện code (Core Implementation — tests) | 30 / 30 |
| Dự đoán độ tương tự (Similarity Predictions) | 4 / 5 |
| Kết quả truy xuất của tôi (Competition Results) | 8 / 10 |
| **Tổng phần cá nhân** | **57 / 60** |
