# Mô tả dự án nghiên cứu và xây dựng mô hình ngôn ngữ lớn song ngữ từ đầu

## 1. Tổng quan dự án

Dự án hướng đến nghiên cứu, thiết kế, triển khai và huấn luyện một mô
hình ngôn ngữ lớn (Large Language Model -- LLM) hỗ trợ tiếng Việt và
tiếng Anh. Mô hình được phát triển theo phương pháp **training from
scratch**: kiến trúc, tokenizer, pipeline dữ liệu, quy trình huấn luyện,
đánh giá và runtime được nghiên cứu, xây dựng và kiểm soát trong phạm vi
dự án.

Mục tiêu không chỉ là tạo ra một ứng dụng sử dụng API AI hoặc tinh chỉnh
một mô hình có sẵn. Trọng tâm là hiểu và triển khai các thành phần nền
tảng tạo nên một mô hình ngôn ngữ: từ toán học, thuật toán tối ưu, mạng
neural, xử lý ngôn ngữ tự nhiên, Transformer, dữ liệu huấn luyện cho đến
inference và triển khai.

Dự án được tổ chức như một chương trình nghiên cứu dài hạn, kết hợp học
lý thuyết, đọc tài liệu khoa học, triển khai mã nguồn, kiểm thử và thực
nghiệm có kiểm soát.

### 1.1. Định hướng kỹ thuật

  -----------------------------------------------------------------------
  Thuộc tính                          Định hướng
  ----------------------------------- -----------------------------------
  Loại mô hình                        Autoregressive, decoder-only
                                      language model

  Ngôn ngữ                            Tiếng Việt và tiếng Anh

  Phương pháp                         Training from scratch, khởi tạo
                                      trọng số ngẫu nhiên

  Kiến trúc nền tảng                  Transformer Decoder

  Tokenizer                           Tự nghiên cứu và triển khai
                                      Byte-level BPE

  Ngôn ngữ phát triển                 Python

  Framework học sâu                   PyTorch, kết hợp NumPy cho các
                                      triển khai nền tảng

  Phần cứng ban đầu                   Máy tính cá nhân, huấn luyện CPU

  Hệ điều hành                        Windows 11

  Lưu trữ                             SSD cục bộ

  Inference ban đầu                   CPU, chạy offline

  Định hướng dữ liệu                  Corpus song ngữ Việt -- Anh
  -----------------------------------------------------------------------

### 1.2. Phạm vi và giới hạn ban đầu

Phần cứng hiện có gồm CPU Intel Core i7-1165G7, RAM 16GB, Intel Iris Xe
và SSD 512GB. Cấu hình này phù hợp để học tập, phát triển mã nguồn, kiểm
thử thuật toán, huấn luyện mô hình cực nhỏ và thực hiện các thí nghiệm
có quy mô giới hạn.

Cấu hình này không phù hợp để huấn luyện một LLM song ngữ có năng lực
thực tế ở quy mô lớn. Vì vậy, giai đoạn đầu tập trung vào mô hình nhỏ,
chuỗi ngắn, batch nhỏ, dữ liệu streaming và đo lường tài nguyên. Mọi
quyết định tăng kích thước mô hình phải dựa trên số liệu RAM, tốc độ
huấn luyện, thời gian chạy và dung lượng lưu trữ thực tế.

Không sử dụng pretrained weights làm nền tảng cho mô hình chính. Các mô
hình có sẵn chỉ có thể được xem xét như đối tượng tham khảo hoặc
baseline đánh giá độc lập, không thay thế quá trình huấn luyện từ đầu.

------------------------------------------------------------------------

## 2. Mục tiêu tổng thể

### 2.1. Mục tiêu nghiên cứu

-   Hiểu nền tảng toán học đứng sau machine learning và deep learning.
-   Tự triển khai các thuật toán học máy và mạng neural cơ bản.
-   Hiểu cách dữ liệu văn bản được chuẩn hóa, mã hóa thành token và đưa
    vào mô hình.
-   Tự thiết kế và triển khai tokenizer phù hợp với tiếng Việt và tiếng
    Anh.
-   Hiểu và tự triển khai kiến trúc Transformer Decoder.
-   Huấn luyện mô hình ngôn ngữ từ trọng số khởi tạo ngẫu nhiên.
-   Xây dựng quy trình đánh giá chất lượng mô hình có thể tái lập.
-   Nghiên cứu cách chuyển một language model thành mô hình làm theo chỉ
    dẫn và hội thoại.
-   Xây dựng runtime inference có khả năng chạy offline.
-   Ghi chép các thí nghiệm, kết quả, giới hạn và quyết định kỹ thuật.

### 2.2. Mục tiêu sản phẩm

Sản phẩm cuối cùng không chỉ là một tệp trọng số. Hệ thống cần có các
thành phần:

1.  Data Acquisition Pipeline -- thu thập dữ liệu.
2.  Data Processing Pipeline -- trích xuất, chuẩn hóa và làm sạch dữ
    liệu.
3.  Tokenizer -- chuyển văn bản thành token IDs và giải mã ngược.
4.  Model Architecture -- định nghĩa kiến trúc mô hình.
5.  Training Engine -- huấn luyện và tiếp tục huấn luyện.
6.  Checkpoint Manager -- lưu và khôi phục trạng thái huấn luyện.
7.  Evaluation Engine -- đánh giá chất lượng và hiệu năng.
8.  Inference Engine -- sinh văn bản từ mô hình.
9.  Model Serving API -- cung cấp giao diện sử dụng mô hình.
10. Experiment Tracking -- theo dõi và tái lập thí nghiệm.
11. Documentation -- tài liệu kỹ thuật, nghiên cứu và hướng dẫn vận
    hành.

------------------------------------------------------------------------

## 3. Lộ trình nghiên cứu tổng thể

Lộ trình tham khảo kéo dài khoảng 12--18 tháng với thời lượng học và
thực hành trung bình khoảng 2 giờ mỗi ngày. Thời gian thực tế có thể
thay đổi theo mức độ tiếp thu, độ phức tạp của implementation và giới
hạn phần cứng.

  ------------------------------------------------------------------------
  Giai đoạn                      Thời gian tham khảo Nội dung
  --------------------- ---------------------------- ---------------------
  1\. Mathematical                        Tháng 1--2 Đại số tuyến tính,
  Foundation                                         giải tích, xác suất,
                                                     thống kê và tối ưu
                                                     hóa

  2\. Computer Science                    Tháng 2--3 Python, cấu trúc dữ
  & Python                                           liệu, thuật toán, hệ
                                                     điều hành và quản lý
                                                     bộ nhớ

  3\. Machine Learning                    Tháng 3--4 Các thuật toán học
                                                     máy và phương pháp
                                                     đánh giá

  4\. Deep Learning                       Tháng 4--5 Neural Network,
                                                     forward pass,
                                                     backpropagation và
                                                     optimizer

  5\. NLP &                               Tháng 5--6 Xử lý văn bản,
  Tokenization                                       embedding, vocabulary
                                                     và BPE

  6\. Transformer                         Tháng 6--7 Attention,
  Architecture                                       Transformer Decoder
                                                     và Mini GPT

  7\. Dataset                             Tháng 7--8 Thu thập, lọc, chuẩn
  Engineering                                        hóa, khử trùng lặp và
                                                     đóng gói corpus

  8\. Pre-training                       Tháng 8--10 Huấn luyện language
                                                     model từ trọng số
                                                     ngẫu nhiên

  9\. Evaluation &                      Tháng 10--11 Benchmark, phân tích
  Optimization                                       loss, hiệu năng và
                                                     tối ưu

  10\. Instruction                      Tháng 11--12 Supervised
  Tuning                                             Fine-Tuning và định
                                                     dạng hội thoại

  11\. Inference &                      Tháng 12--14 Sinh văn bản, runtime
  Deployment                                         offline và API

  12\. Research &                       Tháng 14--18 Nghiên cứu kiến trúc,
  Iteration                                          dữ liệu, tối ưu và mở
                                                     rộng
  ------------------------------------------------------------------------

Các giai đoạn có thể chồng lấn. Không chuyển giai đoạn chỉ dựa trên thời
gian; cần đáp ứng tiêu chí nghiệm thu tương ứng.

------------------------------------------------------------------------

## 4. Giai đoạn 1 -- Nền tảng toán học

Đây là nền tảng để hiểu bản chất của các phép tính trong machine
learning, deep learning và Transformer. Mục tiêu là có thể giải thích và
tự triển khai những phép toán quan trọng, không chỉ gọi thư viện.

### 4.1. Đại số tuyến tính (Linear Algebra)

Nội dung nghiên cứu:

-   Scalar, Vector, Matrix và Tensor.
-   Vector Space và các phép toán trên vector.
-   Dot Product và Matrix Multiplication.
-   Transpose và Inverse.
-   Norm và Distance.
-   Eigenvalues và Eigenvectors.
-   Matrix Decomposition ở mức nền tảng.
-   Broadcasting và các phép toán ma trận trong tính toán số.

Ứng dụng trong mô hình:

-   Token Embedding.
-   Linear Layer.
-   Attention.
-   Weight Matrix.
-   Biểu diễn trạng thái ẩn.

Sản phẩm thực hành: thư viện Linear Algebra nhỏ bằng Python và NumPy, có
unit test cho các phép toán chính.

### 4.2. Giải tích (Calculus)

Nội dung nghiên cứu:

-   Function và Derivative.
-   Partial Derivatives.
-   Chain Rule.
-   Gradient.
-   Jacobian ở mức cần thiết.
-   Multivariable Calculus.
-   Computational Graph.
-   Quy tắc đạo hàm của các hàm thường dùng.

Ứng dụng:

-   Forward Propagation.
-   Backpropagation.
-   Tính gradient cho tham số mô hình.
-   Tối ưu hóa hàm loss.

Sản phẩm thực hành: Automatic Differentiation Engine đơn giản, tham khảo
cách tổ chức của micrograd nhưng tự triển khai và kiểm thử.

### 4.3. Xác suất và thống kê

Nội dung nghiên cứu:

-   Probability và Conditional Probability.
-   Bayes' Theorem.
-   Random Variables.
-   Probability Distributions.
-   Expectation và Variance.
-   Entropy.
-   Cross Entropy.
-   KL Divergence.
-   Maximum Likelihood Estimation.

Ứng dụng:

-   Softmax và phân phối xác suất token.
-   Next-token prediction.
-   Language Modeling Loss.
-   Sampling khi sinh văn bản.
-   Đánh giá mức độ bất ngờ của mô hình đối với dữ liệu.

### 4.4. Tối ưu hóa (Optimization)

Nội dung nghiên cứu:

-   Gradient Descent.
-   Stochastic Gradient Descent.
-   Momentum.
-   RMSProp.
-   Adam và AdamW.
-   Learning Rate Scheduling.
-   Weight Decay.
-   Gradient Clipping.
-   Các vấn đề liên quan đến gradient quá lớn hoặc quá nhỏ.

Sản phẩm thực hành: triển khai Gradient Descent và một số optimizer cơ
bản; so sánh kết quả với implementation trong PyTorch.

**Tiêu chí hoàn thành giai đoạn:** có thể giải thích và triển khai
Gradient Descent, Backpropagation và Cross Entropy ở mức cơ bản mà không
phụ thuộc hoàn toàn vào Autograd.

------------------------------------------------------------------------

## 5. Giai đoạn 2 -- Khoa học máy tính và Python

Mục tiêu là xây dựng năng lực lập trình đủ vững để phát triển các thành
phần dữ liệu, mô hình, huấn luyện và inference.

### 5.1. Python

-   OOP và tổ chức module.
-   Type Hints và cấu trúc package.
-   Iterator và Generator.
-   Decorator và Context Manager.
-   Exception Handling.
-   Logging.
-   Serialization.
-   Quản lý môi trường và dependency.
-   Viết code có thể kiểm thử và bảo trì.

### 5.2. Cấu trúc dữ liệu và thuật toán

-   Array và Dynamic Array.
-   Linked List.
-   Stack và Queue.
-   Hash Map.
-   Tree và Heap.
-   Sorting và Searching.
-   Độ phức tạp thời gian và bộ nhớ.
-   Các kỹ thuật tối ưu thuật toán cơ bản.

### 5.3. Hệ điều hành và quản lý tài nguyên

-   Process và Thread.
-   Scheduling.
-   Stack và Heap.
-   Memory Management.
-   File System.
-   Binary File.
-   Memory Mapping.
-   Multiprocessing và Multithreading.
-   Giới hạn bộ nhớ và xử lý dữ liệu theo luồng.

### 5.4. Tính toán số

-   NumPy Array.
-   Broadcasting.
-   Vectorization.
-   Numerical Stability.
-   Floating-point Precision.
-   Kiểu dữ liệu số và ảnh hưởng đến bộ nhớ.

Sản phẩm thực hành:

-   Tensor Data Structure đơn giản.
-   Mini Matrix Library.
-   Data Loader.
-   Batch Generator.
-   Checkpoint Serializer.
-   Training Logger.

Có thể tổ chức một thư viện tensor thử nghiệm dùng NumPy làm nền tảng
ban đầu. Thư viện này phục vụ học tập, không cần thay thế PyTorch trong
hệ thống huấn luyện chính.

------------------------------------------------------------------------

## 6. Giai đoạn 3 -- Machine Learning

Mục tiêu là hiểu cách mô hình học quy luật từ dữ liệu, cách đánh giá và
các nguyên nhân khiến mô hình tổng quát hóa kém.

### 6.1. Thuật toán cần nghiên cứu

  Thuật toán               Mục đích
  ------------------------ -------------------------
  Linear Regression        Hồi quy
  Logistic Regression      Phân loại
  Decision Tree            Học theo cây quyết định
  Random Forest            Ensemble Learning
  K-Means                  Clustering
  Naive Bayes              Phân loại xác suất
  Support Vector Machine   Classification

### 6.2. Kiến thức nền tảng

-   Training, Validation và Test Set.
-   Overfitting và Underfitting.
-   Bias và Variance.
-   Feature Engineering.
-   Data Leakage.
-   Regularization.
-   Cross Validation.
-   Classification Metrics.
-   Regression Metrics.
-   Khả năng tổng quát hóa.

Sản phẩm thực hành: ML Library nhỏ, tự triển khai tối thiểu Linear
Regression, Logistic Regression và K-Means. Kết quả cần được so sánh với
thư viện chuẩn trên cùng dữ liệu.

------------------------------------------------------------------------

## 7. Giai đoạn 4 -- Deep Learning

Mục tiêu là hiểu cấu trúc Neural Network và cơ chế cập nhật trọng số
thông qua gradient.

### 7.1. Nội dung nghiên cứu

1.  Perceptron.
2.  Multi-Layer Perceptron.
3.  Activation Functions.
4.  Forward Propagation.
5.  Backpropagation.
6.  Loss Functions.
7.  Optimizers.
8.  Weight Initialization.
9.  Batch Normalization.
10. Dropout.
11. Mini-batch Training.
12. Gradient Stability.

### 7.2. Sản phẩm thực hành

-   Neural Network triển khai từ đầu.
-   Mô hình phân loại dữ liệu.
-   Mô hình dự đoán chuỗi số đơn giản.
-   Training Loop.
-   Loss Visualization.
-   Kiểm thử gradient bằng finite differences.
-   So sánh kết quả giữa implementation tự viết và PyTorch.

Sau khi hiểu các thành phần, sử dụng PyTorch để triển khai mô hình có
cấu trúc rõ ràng và chuẩn bị cho giai đoạn Transformer.

------------------------------------------------------------------------

## 8. Giai đoạn 5 -- NLP và Tokenization

Mục tiêu là hiểu cách biểu diễn văn bản để mô hình có thể xử lý chuỗi
token.

### 8.1. Natural Language Processing

Nội dung nghiên cứu:

-   Text Normalization.
-   Unicode và UTF-8.
-   Word Segmentation.
-   Character-level Representation.
-   Subword Tokenization.
-   Vocabulary.
-   N-gram Language Model.
-   Word Embedding.
-   Positional Representation.
-   Language Modeling.

### 8.2. Đặc điểm dữ liệu tiếng Việt và tiếng Anh

Tiếng Việt cần được xử lý cẩn thận về:

-   Dấu thanh và ký tự có dấu.
-   Unicode normalization.
-   Ranh giới âm tiết và từ.
-   Từ ghép.
-   Văn bản trộn tiếng Việt và tiếng Anh.
-   Dấu câu, viết tắt và biến thể cách viết.

Tiếng Anh cần được xử lý về:

-   Chữ hoa và chữ thường.
-   Dấu câu.
-   Từ ghép và hậu tố.
-   Từ hiếm.
-   Văn bản có ký tự đặc biệt.
-   Cách biểu diễn số và ký hiệu.

Không nên áp dụng một bước chuẩn hóa làm mất thông tin ngôn ngữ cần
thiết. Cần kiểm tra round-trip encode/decode để bảo đảm văn bản có thể
được khôi phục chính xác theo thiết kế tokenizer.

### 8.3. Xây dựng Tokenizer

Thứ tự nghiên cứu và triển khai:

1.  Character Tokenizer.
2.  Word Tokenizer.
3.  Byte-level Tokenizer.
4.  Byte Pair Encoding (BPE).

Định hướng cấu hình thử nghiệm ban đầu:

  Thành phần       Giá trị định hướng
  ---------------- -----------------------
  Vocabulary       8.000--16.000 tokens
  Input            UTF-8 text
  Algorithm        Byte-level BPE
  Special Tokens   BOS, EOS, PAD, UNK
  Languages        Vietnamese và English

Vocabulary thực tế phải được quyết định dựa trên corpus, tỷ lệ phân mảnh
token, khả năng xử lý văn bản và giới hạn bộ nhớ.

Sản phẩm: tokenizer độc lập có khả năng train vocabulary, encode,
decode, lưu và nạp cấu hình, xử lý văn bản song ngữ và kiểm thử Unicode.

------------------------------------------------------------------------

## 9. Giai đoạn 6 -- Transformer Architecture

Đây là thành phần kiến trúc cốt lõi của mô hình ngôn ngữ.

### 9.1. Nội dung nghiên cứu

-   Sequence Modeling.
-   Attention Mechanism.
-   Self-Attention.
-   Scaled Dot-Product Attention.
-   Causal Masking.
-   Multi-Head Attention.
-   Positional Encoding hoặc Positional Embedding.
-   Token Embedding.
-   Feed Forward Network.
-   Residual Connections.
-   Layer Normalization.
-   Transformer Decoder.
-   Language Modeling Head.

### 9.2. Kiến trúc mục tiêu

Mô hình sử dụng kiến trúc Autoregressive Decoder-only Transformer.

Luồng xử lý tổng quát:

1.  Nhận chuỗi token IDs.
2.  Chuyển token IDs thành embeddings.
3.  Kết hợp thông tin vị trí.
4.  Đưa qua các Transformer Blocks.
5.  Mỗi block thực hiện causal self-attention và feed-forward
    computation, kết hợp residual connections và normalization.
6.  Đưa hidden states qua language modeling head.
7.  Tạo logits cho vocabulary.
8.  Tính xác suất token tiếp theo và tính loss khi huấn luyện.

Causal mask bảo đảm mỗi vị trí không truy cập thông tin từ các token
tương lai.

### 9.3. Cấu hình Mini GPT thử nghiệm

Các cấu hình sau chỉ là điểm bắt đầu cho thí nghiệm trên CPU, không phải
cấu hình LLM sản phẩm.

  Thông số                    Model A     Model B
  ----------------------- ----------- -----------
  Parameters mục tiêu       Khoảng 1M   Khoảng 5M
  Layers                            2           4
  Attention Heads                   2           4
  Embedding Dimension              64         128
  Context Length                  128         256
  Vocabulary thử nghiệm            4K          8K

Cần tính lại số tham số thực tế từ cấu hình triển khai. Model A phải
được đo bộ nhớ và tốc độ trước khi thử Model B.

Sản phẩm: Transformer Decoder và Mini GPT tự triển khai, có unit test
cho attention, causal masking, embedding, loss và forward pass.

------------------------------------------------------------------------

## 10. Giai đoạn 7 -- Dataset Engineering

Mục tiêu là xây dựng corpus song ngữ có nguồn gốc, giấy phép, chất lượng
và phiên bản rõ ràng.

### 10.1. Nguồn dữ liệu có thể nghiên cứu

  -----------------------------------------------------------------------
  Nguồn                   Loại dữ liệu            Mục đích
  ----------------------- ----------------------- -----------------------
  Common Crawl            Web Corpus              Văn bản tổng quát

  Hugging Face Datasets   Public Datasets         Corpus và benchmark

  OPUS                    Parallel Corpus         Dữ liệu song ngữ

  Wikipedia               Encyclopedia            Kiến thức tổng quát

  Statistical Machine     Language Datasets       Nghiên cứu ngôn ngữ
  Translation Data                                
  -----------------------------------------------------------------------

Mọi nguồn dữ liệu phải được kiểm tra điều khoản sử dụng, giấy phép, điều
kiện phân phối và quyền sử dụng cho mục đích huấn luyện. Dữ liệu tải
miễn phí không mặc nhiên được phép dùng trong sản phẩm thương mại.

### 10.2. Data Pipeline

Pipeline dự kiến:

1.  Data Acquisition -- thu thập dữ liệu.
2.  Extraction -- trích xuất nội dung.
3.  Normalization -- chuẩn hóa văn bản.
4.  Language Detection -- phân loại ngôn ngữ.
5.  Deduplication -- loại bỏ bản ghi trùng lặp.
6.  Quality Filtering -- loại bỏ dữ liệu chất lượng thấp.
7.  Safety Filtering -- xử lý dữ liệu độc hại hoặc không phù hợp.
8.  PII Filtering -- phát hiện và loại bỏ thông tin cá nhân nhạy cảm.
9.  Tokenization -- chuyển văn bản thành token IDs.
10. Dataset Packing -- đóng gói thành training sequences.

### 10.3. Phân bổ corpus ban đầu

Tỷ lệ thử nghiệm theo số lượng token:

-   English: 50%.
-   Vietnamese: 50%.

Đây là giả thuyết ban đầu. Cần thử nghiệm các tỷ lệ khác nhau và đánh
giá riêng từng ngôn ngữ. Không nên kết luận tỷ lệ tối ưu chỉ từ số lượng
dữ liệu.

### 10.4. Quản lý chất lượng và nguồn gốc

Mỗi shard hoặc tập dữ liệu cần lưu:

-   Dataset ID và version.
-   Nguồn dữ liệu.
-   Giấy phép hoặc điều khoản sử dụng.
-   Thời điểm thu thập.
-   Quy trình xử lý.
-   Ngôn ngữ.
-   Số lượng bản ghi.
-   Số lượng token.
-   Hash hoặc checksum.
-   Các bước lọc đã áp dụng.

Sản phẩm: pipeline dữ liệu có thể chạy lại, có manifest, log, thống kê
chất lượng và khả năng truy xuất nguồn gốc.

------------------------------------------------------------------------

## 11. Giai đoạn 8 -- Pre-training

Mục tiêu là huấn luyện mô hình từ trọng số khởi tạo ngẫu nhiên bằng mục
tiêu dự đoán token tiếp theo.

### 11.1. Cơ chế học

Với một chuỗi token, mô hình nhận các token trước đó và học dự đoán
token kế tiếp. Trong quá trình huấn luyện, các vị trí được xử lý song
song thông qua causal masking.

Mục tiêu phổ biến là tối thiểu hóa Cross Entropy giữa phân phối dự đoán
và token mục tiêu.

### 11.2. Training Engine

Các thành phần cần triển khai:

-   Dataset Loader.
-   Batch Generator.
-   Forward Pass.
-   Loss Calculation.
-   Backward Pass.
-   Gradient Accumulation.
-   Optimizer.
-   Learning Rate Scheduler.
-   Gradient Clipping.
-   Checkpoint Saving.
-   Training Monitoring.
-   Resume Training.

### 11.3. Các thí nghiệm ban đầu

  Experiment   Mục tiêu
  ------------ ---------------------------------------
  EXP-001      Huấn luyện trên corpus tiếng Anh nhỏ
  EXP-002      Huấn luyện trên corpus tiếng Việt nhỏ
  EXP-003      Huấn luyện trên corpus song ngữ
  EXP-004      Thay đổi Vocabulary
  EXP-005      Thay đổi Context Length
  EXP-006      Thử nghiệm Learning Rate
  EXP-007      So sánh cấu hình mô hình
  EXP-008      Kiểm tra khả năng sinh văn bản

Mỗi thí nghiệm phải có Experiment ID, Dataset Version, Model
Configuration, Random Seed, thời gian huấn luyện, Training Loss,
Validation Loss, checkpoint và kết luận.

Cần thay đổi có kiểm soát từng biến hoặc từng nhóm biến để xác định
nguyên nhân của kết quả.

### 11.4. Chiến lược huấn luyện trên CPU

-   Bắt đầu với mô hình khoảng 1M tham số.
-   Sử dụng sequence length ngắn.
-   Batch size nhỏ.
-   Dùng gradient accumulation khi phù hợp.
-   Dataset streaming hoặc đọc theo shard.
-   Giới hạn số worker để không làm hệ thống quá tải.
-   Theo dõi RAM, CPU utilization, tốc độ token/giây và dung lượng
    checkpoint.
-   Lưu checkpoint định kỳ.
-   Chỉ dùng mixed precision khi backend CPU hỗ trợ và phép đo cho thấy
    có lợi.
-   Giữ đủ tài nguyên cho hệ điều hành và công cụ phát triển.

Không sử dụng toàn bộ 16GB RAM làm ngân sách cho mô hình. Hệ điều hành,
IDE, trình duyệt và tiến trình khác cũng cần bộ nhớ.

------------------------------------------------------------------------

## 12. Giai đoạn 9 -- Evaluation và Optimization

Mục tiêu là đánh giá mô hình bằng dữ liệu độc lập và chỉ số có thể so
sánh.

### 12.1. Chỉ số đánh giá

  Metric              Ý nghĩa
  ------------------- --------------------------------------------
  Training Loss       Sai số trên tập huấn luyện
  Validation Loss     Sai số trên tập validation
  Perplexity          Mức độ bất ngờ của mô hình đối với dữ liệu
  Token Accuracy      Tỷ lệ dự đoán đúng token
  Throughput          Số token xử lý mỗi giây
  Memory Usage        Bộ nhớ tiêu thụ
  Inference Latency   Thời gian phản hồi

Perplexity có thể được tính từ Cross Entropy theo công thức:

\[ PPL = `\exp`{=tex}(`\text{CrossEntropy}`{=tex}) \]

Chỉ so sánh perplexity khi cách tokenize, tập dữ liệu và điều kiện đánh
giá tương thích.

### 12.2. Đánh giá tiếng Việt

-   Hoàn thành câu.
-   Ngữ pháp.
-   Từ vựng.
-   Khả năng duy trì ngữ cảnh.
-   Khả năng sinh văn bản tiếng Việt.
-   Xử lý dấu và Unicode.

### 12.3. Đánh giá tiếng Anh

-   Grammar.
-   Vocabulary.
-   Sentence Completion.
-   Context Understanding.
-   Text Generation.

### 12.4. Đánh giá song ngữ

-   Việt sang Anh.
-   Anh sang Việt.
-   Chuyển đổi giữa hai ngôn ngữ.
-   Văn bản trộn tiếng Việt và tiếng Anh.
-   Khả năng giữ nguyên tên riêng, thuật ngữ và ký hiệu.

Cần xây dựng bộ đánh giá riêng, kiểm soát dữ liệu trùng lặp với training
corpus và ghi rõ cách chấm điểm. Không sử dụng kết quả benchmark không
tương thích để đưa ra kết luận quá mức.

### 12.5. Evaluation Harness

Cấu trúc tham khảo:

``` text
evaluation/
├── datasets/
│   ├── vietnamese/
│   ├── english/
│   └── bilingual/
├── metrics/
│   ├── perplexity.py
│   ├── accuracy.py
│   └── throughput.py
├── evaluators/
│   ├── language_model.py
│   └── generation.py
├── reports/
└── run_evaluation.py
```

Mỗi phiên bản mô hình phải được đánh giá bằng cùng bộ dữ liệu kiểm tra
đã khóa phiên bản. Test set không được sử dụng trong huấn luyện hoặc lựa
chọn hyperparameter.

------------------------------------------------------------------------

## 13. Giai đoạn 10 -- Instruction Tuning

Pre-training giúp mô hình học phân phối ngôn ngữ và dự đoán token tiếp
theo, nhưng không tự động bảo đảm mô hình làm theo chỉ dẫn hoặc phản hồi
theo định dạng hội thoại.

Instruction Tuning là bước tiếp theo để nghiên cứu khả năng làm theo yêu
cầu.

### 13.1. Nội dung

-   Instruction Dataset.
-   Supervised Fine-Tuning (SFT).
-   Chat Template.
-   EOS Handling.
-   Response Masking.
-   Conversation Formatting.
-   Safety Evaluation.
-   Đánh giá khả năng làm theo chỉ dẫn.

### 13.2. Định dạng dữ liệu

Ví dụ:

``` json
{
  "messages": [
    {
      "role": "user",
      "content": "Giải thích biến trong Python."
    },
    {
      "role": "assistant",
      "content": "Biến là một tên dùng để tham chiếu đến dữ liệu..."
    }
  ]
}
```

Cần xác định rõ token phân tách vai trò, token kết thúc lượt và phần nào
trong chuỗi được tính loss.

Instruction tuning không tự tạo ra năng lực suy luận phức tạp cho một mô
hình quá nhỏ. Cần đánh giá năng lực thực tế, không chỉ dựa vào việc câu
trả lời có vẻ trôi chảy.

------------------------------------------------------------------------

## 14. Giai đoạn 11 -- Inference Engine và Deployment

Mục tiêu là chạy mô hình hoàn toàn offline trên máy tính cá nhân.

### 14.1. Kiến trúc triển khai

Luồng xử lý dự kiến:

1.  Client gửi yêu cầu.
2.  Inference API xác thực và kiểm tra tham số.
3.  Generation Engine quản lý context và sampling.
4.  Runtime nạp model weights và tokenizer.
5.  Mô hình tạo token.
6.  Tokenizer giải mã thành văn bản.
7.  Kết quả được trả về theo dạng thông thường hoặc streaming.

### 14.2. Chức năng inference

-   Greedy Decoding.
-   Temperature Sampling.
-   Top-K Sampling.
-   Top-P Sampling.
-   Repetition Penalty.
-   Stop Tokens.
-   Context Window Management.
-   KV Cache.
-   Streaming Generation.
-   Model Loading và Unloading.

KV Cache cần được nghiên cứu và triển khai sau khi inference cơ bản hoạt
động đúng. Trên CPU, cần đo lợi ích thực tế so với chi phí bộ nhớ.

### 14.3. Định dạng đóng gói mô hình

Cấu trúc đề xuất:

``` text
model/
├── config.json
├── tokenizer.json
├── model.safetensors
├── generation_config.json
├── special_tokens.json
└── model_card.md
```

Trọng số được lưu bằng định dạng an toàn phù hợp. Cấu hình, tokenizer và
trọng số phải có phiên bản tương thích, checksum và thông tin nguồn gốc.

### 14.4. API dự kiến

``` http
POST /v1/generate
POST /v1/chat/completions
GET  /v1/models
GET  /health
```

Có thể sử dụng FastAPI để triển khai API ban đầu. Giao diện desktop bằng
Tauri hoặc Wails có thể được phát triển sau khi runtime ổn định; không
phải ưu tiên của giai đoạn nghiên cứu đầu tiên.

------------------------------------------------------------------------

## 15. Giai đoạn 12 -- Research và Optimization

Sau khi có mô hình chạy được, tiếp tục nghiên cứu các hướng cải tiến.

  Chủ đề         Nội dung
  -------------- ------------------------------------------------
  Architecture   Pre-Norm, Post-Norm, RMSNorm
  Attention      Multi-Query Attention, Grouped-Query Attention
  Optimization   AdamW, Muon và các optimizer khác
  Training       Learning Rate Scheduling
  Memory         Gradient Checkpointing
  Inference      KV Cache, Quantization
  Dataset        Data Mixture, Deduplication
  Scaling        Scaling Laws
  Evaluation     Generalization và Benchmark Design

Mục tiêu là dùng thí nghiệm để hiểu mối quan hệ giữa số tham số, lượng
token huấn luyện, chất lượng dữ liệu, compute và chất lượng đầu ra. Chỉ
mở rộng quy mô khi có đủ tài nguyên và số liệu để biện minh cho quyết
định.

------------------------------------------------------------------------

## 16. Cấu trúc mã nguồn đề xuất

Repository cần tách biệt mã nghiên cứu, pipeline dữ liệu, kiến trúc mô
hình, huấn luyện, đánh giá và inference.

``` text
project/
│
├── configs/
│   ├── model/
│   │   ├── tiny.json
│   │   └── small.json
│   ├── training/
│   └── tokenizer/
│
├── data/
│   ├── raw/
│   ├── processed/
│   ├── tokenized/
│   ├── validation/
│   └── manifests/
│
├── research/
│   ├── mathematics/
│   ├── machine_learning/
│   ├── deep_learning/
│   ├── nlp/
│   └── transformer/
│
├── src/
│   ├── core/
│   │   ├── tensor.py
│   │   ├── autograd.py
│   │   └── operations.py
│   │
│   ├── tokenizer/
│   │   ├── normalizer.py
│   │   ├── bpe.py
│   │   └── tokenizer.py
│   │
│   ├── data/
│   │   ├── loader.py
│   │   ├── cleaner.py
│   │   ├── deduplicator.py
│   │   └── dataset.py
│   │
│   ├── models/
│   │   ├── embeddings.py
│   │   ├── attention.py
│   │   ├── transformer.py
│   │   └── language_model.py
│   │
│   ├── training/
│   │   ├── trainer.py
│   │   ├── optimizer.py
│   │   ├── scheduler.py
│   │   └── checkpoint.py
│   │
│   ├── evaluation/
│   │   ├── metrics.py
│   │   ├── benchmark.py
│   │   └── evaluator.py
│   │
│   └── inference/
│       ├── sampler.py
│       ├── generator.py
│       └── runtime.py
│
├── experiments/
├── tests/
├── scripts/
├── checkpoints/
├── docs/
├── pyproject.toml
└── README.md
```

### 16.1. Nguyên tắc tổ chức mã nguồn

-   Tách Research Code khỏi Production Code.
-   Tách Data Pipeline khỏi Model Architecture.
-   Mỗi mô hình có configuration riêng.
-   Hyperparameter không được hardcode rải rác trong source code.
-   Mỗi experiment phải có thông tin để tái lập.
-   Có unit test cho tokenizer, attention, loss, checkpoint và data
    processing.
-   Checkpoint phải lưu model state, optimizer state, scheduler state,
    global step, random state và configuration cần thiết để resume
    training.
-   Có log lỗi và thống kê tài nguyên.
-   Không đưa dữ liệu lớn hoặc checkpoint vào Git repository thông
    thường; quản lý chúng bằng storage và manifest riêng.

------------------------------------------------------------------------

## 17. Quản lý thời gian học và nghiên cứu

Với khoảng 2 giờ mỗi ngày, lịch tham khảo:

  Hoạt động                   Thời lượng
  ------------------------- ------------
  Học lý thuyết                  35 phút
  Đọc tài liệu nghiên cứu        20 phút
  Viết code và thực hành         45 phút
  Kiểm thử và ghi chép           20 phút

Cuối mỗi tuần cần tổng kết:

-   Kiến thức nào đã hiểu?
-   Kiến thức nào còn chưa rõ?
-   Implementation nào đã hoàn thành?
-   Những lỗi nào đã gặp và nguyên nhân?
-   Kết quả thực nghiệm có thể tái lập không?
-   Tuần tiếp theo cần cải thiện điều gì?

Không đánh giá tiến độ chỉ bằng số lượng bài học đã xem hoặc số lượng
dòng code đã viết. Cần đánh giá bằng khả năng giải thích,
implementation, kiểm thử và kết quả thực nghiệm.

------------------------------------------------------------------------

## 18. Tiêu chuẩn nghiệm thu (Milestones)

Mỗi milestone cần có mã nguồn, test, tài liệu giải thích và kết quả thực
nghiệm.

  Milestone   Điều kiện nghiệm thu
  ----------- ---------------------------------------------------
  M1          Hiểu Linear Algebra và Calculus ứng dụng
  M2          Tự triển khai Automatic Differentiation
  M3          Tự triển khai Neural Network
  M4          Hoàn thành BPE Tokenizer
  M5          Hoàn thành Transformer Decoder
  M6          Huấn luyện được Mini GPT từ random weights
  M7          Corpus Việt -- Anh có pipeline và metadata
  M8          Huấn luyện và resume được mô hình song ngữ
  M9          Có benchmark và báo cáo đánh giá
  M10         Hoàn thành thử nghiệm Instruction Tuning
  M11         Chạy inference offline
  M12         Phát hành phiên bản mô hình đầu tiên kèm tài liệu

Một milestone chỉ được đánh dấu hoàn thành khi các tiêu chí kỹ thuật
được kiểm tra. Code chạy được không đồng nghĩa với implementation đã
đúng hoặc mô hình đã đạt năng lực mong muốn.

------------------------------------------------------------------------

## 19. Tuần nghiên cứu đầu tiên

Mục tiêu tuần đầu là thiết lập môi trường và bắt đầu nền tảng toán học,
chưa triển khai Transformer.

  -----------------------------------------------------------------------
  Ngày                    Nội dung                Sản phẩm
  ----------------------- ----------------------- -----------------------
  Ngày 1                  Scalar, Vector, Matrix, Phép toán vector và
                          Tensor                  matrix bằng NumPy

  Ngày 2                  Dot Product và Matrix   Mini Linear Algebra
                          Multiplication          implementation

  Ngày 3                  Transpose, Norm,        Bộ kiểm thử các phép
                          Broadcasting            toán

  Ngày 4                  Function, Derivative,   Các phép tính đạo hàm
                          Partial Derivative      cơ bản

  Ngày 5                  Chain Rule và           Computational Graph đơn
                          Computational Graph     giản

  Ngày 6                  Gradient Descent        Linear Regression tự
                                                  triển khai

  Ngày 7                  Ôn tập và kiểm thử      Báo cáo tuần đầu tiên
  -----------------------------------------------------------------------

### 19.1. Môi trường phát triển ban đầu

-   Python 3.11.
-   NumPy.
-   PyTorch.
-   Pytest.
-   Git.
-   VS Code hoặc PyCharm.
-   Công cụ theo dõi log và tài nguyên hệ thống.

Không cần cài đặt quá nhiều thư viện ngay từ đầu. Chỉ bổ sung dependency
khi có nhu cầu cụ thể và ghi rõ lý do sử dụng.

------------------------------------------------------------------------

## 20. Tài liệu nghiên cứu tham khảo

Các tài liệu nền tảng cần đọc theo từng giai đoạn:

  ---------------------------------------------------------------------------------------
  Tài liệu                                            Chủ đề
  --------------------------------------------------- -----------------------------------
  [Neural Networks: Zero to                           Neural Network và xây dựng GPT từ
  Hero](https://karpathy.ai/zero-to-hero.html)        đầu

  [PyTorch                                            Tensor, Autograd và Training
  Tutorials](https://docs.pytorch.org/tutorials/)     

  [Hugging Face Learn](https://huggingface.co/learn)  NLP, Tokenizer và Transformers

  [Attention Is All You                               Transformer
  Need](https://arxiv.org/abs/1706.03762)             

  [Language Models are Few-Shot                       GPT-3
  Learners](https://arxiv.org/abs/2005.14165)         

  [Training Compute-Optimal Large Language            Compute-optimal training và Scaling
  Models](https://arxiv.org/abs/2203.15556)           Laws

  [SentencePiece](https://arxiv.org/abs/1804.07461)   Tokenization

  [Adam: A Method for Stochastic                      Optimization
  Optimization](https://arxiv.org/abs/1412.6980)      

  [Layer                                              Normalization
  Normalization](https://arxiv.org/abs/1607.06450)    

  [Common Crawl](https://commoncrawl.org/)            Web corpus

  [Hugging Face                                       Dataset
  Datasets](https://huggingface.co/datasets)          

  [OPUS](https://opus.nlpl.eu/)                       Parallel corpus
  ---------------------------------------------------------------------------------------

Paper là nguồn để hiểu nguyên lý và bối cảnh nghiên cứu. Khi triển khai
cần đối chiếu với tài liệu framework, kiểm thử và thực nghiệm thực tế.

------------------------------------------------------------------------

## 21. Nguyên tắc phát triển dự án

1.  Không đánh đổi kiến thức nền tảng lấy tốc độ phát triển.
2.  Không đánh đổi chất lượng dữ liệu lấy số lượng token.
3.  Không tuyên bố mô hình có năng lực khi chưa có benchmark chứng minh.
4.  Không sử dụng pretrained weights làm nền tảng cho mô hình chính.
5.  Không mở rộng quy mô khi chưa đo được tài nguyên và hiệu năng.
6.  Mọi thí nghiệm phải có cấu hình, dữ liệu và kết quả được ghi nhận.
7.  Mọi thành phần quan trọng phải có test.
8.  Tôn trọng giấy phép dữ liệu, quyền riêng tư và nguồn gốc corpus.
9.  Tách biệt kết quả thực nghiệm, giả thuyết và kết luận.
10. Ưu tiên khả năng tái lập, tính chính xác và khả năng bảo trì.

------------------------------------------------------------------------

## 22. Định nghĩa thành công

Thành công ban đầu không được đo bằng việc tạo ra một mô hình có số
lượng tham số lớn hoặc có thể cạnh tranh với các LLM thương mại.

Cột mốc kỹ thuật đầu tiên là xây dựng được một Mini GPT:

-   Có tokenizer do dự án tự xây dựng.
-   Có kiến trúc Transformer Decoder được hiểu và triển khai.
-   Được khởi tạo từ trọng số ngẫu nhiên.
-   Được huấn luyện trên dữ liệu do dự án chuẩn bị.
-   Có thể lưu và khôi phục checkpoint.
-   Có thể sinh token và văn bản.
-   Có bộ kiểm thử và báo cáo đánh giá.
-   Có thể chạy offline trên máy tính cá nhân.

Sau cột mốc đó, dự án tiếp tục phát triển thông qua các thí nghiệm về dữ
liệu, kiến trúc, huấn luyện, đánh giá và tối ưu hóa.

**Định hướng cốt lõi của dự án là tự nghiên cứu và hiểu cách xây dựng
một mô hình ngôn ngữ từ nền tảng toán học đến hệ thống vận hành hoàn
chỉnh, thay vì chỉ biết cách sử dụng một mô hình AI có sẵn.**
