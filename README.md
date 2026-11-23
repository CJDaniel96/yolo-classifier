# AOI YOLO OBB 微服務

在 Windows 上監控 XML 與影像兩個輸入資料夾，只對 Status 為 False Call 且 Type 在 ChipRC BOM 清單內的元件，
依 XML 的 Image 座標裁切原圖，使用 YOLO OBB 模型推論，
輸出更新 `Component.AI_result` 的 XML，再將原始 XML 與影像歸檔。

## Windows 部署

1. 安裝 **64 位元 Python 3.12**，包含 Windows `py` launcher，將專案放到遠端 Windows 的固定位置。
2. 首次雙擊 `setup.bat`，建立 `.venv` 並安裝 CPU 版本的依賴。首次安裝需要網路。
   若使用 NVIDIA GPU，在命令提示字元執行 `setup.bat cu130` 安裝 CUDA 13.0 版本。
   PyTorch、torchvision 與 Ultralytics 版本沿用原 CLI 專案的鎖定版本。
   GPU 與驅動需支援選用的 CUDA 版本，可參考 [PyTorch 安裝文件](https://pytorch.org/get-started/locally/)。
3. 將自行訓練的 OBB 權重放到 `models/best.pt`，或修改 `inference.model`。
4. 編輯 `config.json`，至少確認輸入／輸出路徑、模型與 BOM 清單路徑、`ok_label`、`ok_count` 及 `device`。
   BOM 預設使用隨附的 `bom/1001_T1_BOM_List.txt`，每行一個料號。
5. 雙擊 `start.bat`。啟動會檢查設定、載入模型並驗證 OK 類別，接著開始監控。

`inference.device` 預設為 `"cpu"`；GPU 部署時改為 `0` 或 `"0"`。
`setup.bat cu130` 會選擇 GPU 套件，設定檔仍需自行選擇裝置。
Windows 完整依賴版本固定在 `requirements-windows.lock`；`setup.bat` 會使用該清單安裝。

`start.bat` 是前景常駐程式，視窗需保持開啟。按 Ctrl+C 會等待目前的工作安全結束。
如需開機自動啟動或登出後繼續執行，可用 Windows 工作排程器啟動此批次檔；本專案不會自行註冊系統服務。

尚未在實際 Windows 主機、NVIDIA GPU 或使用者訓練權重上完成端到端驗證。

## 設定方式

設定檔使用 JSON，未知欄位或無效參數會阻止啟動。所有相對路徑都以 **設定檔所在目錄** 為基準。
Windows 絕對路徑建議寫成 `"D:/AOI/XML"`；若使用反斜線，JSON 內需寫成 `"D:\\AOI\\XML"`。
修改設定後重新啟動服務。

| 設定 | 用途／預設 |
| --- | --- |
| `paths.xml_input` | XML 輸入資料夾，預設 `input/xml` |
| `paths.image_input` | 已由其他程式複製完成的影像資料夾，預設 `input/images` |
| `paths.xml_output` | 結果 XML 輸出，預設 `output`，保留原檔名 |
| `paths.archive_dir` | 歸檔根目錄，預設 `input/xml/DONE`；省略或 `null` 時為 XML 輸入下的 `DONE` |
| `paths.state_db` | 本機 SQLite 工作紀錄，預設 `runtime/jobs.sqlite3` |
| `paths.log_dir` | 日誌位置，預設 `logs` |
| `paths.bom_list` | ChipRC 料號清單，預設 `bom/1001_T1_BOM_List.txt` |
| `watch.poll_seconds` | 掃描間隔，預設 2 秒 |
| `watch.stable_seconds` | 檔案大小及修改時間維持不變的等待時間，預設 2 秒 |
| `watch.image_wait_timeout_seconds` | XML 進入待處理狀態後，等待完整 XML／影像的上限，預設 300 秒 |
| `watch.retry_seconds` | 推論、輸出與歸檔錯誤的重試間隔，預設 10 秒 |
| `watch.max_retries` | 推論／資料驗證失敗的嘗試上限，預設 3 次 |
| `watch.failure_statuses` | 需要推論的 Component Status，預設 `["False Call"]`，去除前後空白並忽略大小寫 |
| `archive.after_success` | `archive`（預設）、`delete` 或 `keep` |
| `archive.retention_days` | 歸檔保留天數，預設 `null` 不自動清除 |
| `http.host` / `http.port` | 預設 `127.0.0.1:8000` |

兩個輸入資料夾只檢查**直接子檔案**，不遞迴搜尋子資料夾，因此不會讀取 DONE。
所有影像均透過 XML 引用尋找，不會推論資料夾內尚無 XML 引用的影像。
輸入、結果輸出、歸檔及複查路徑會檢查是否有不安全的重疊。
工作紀錄請放在主機本機磁碟；多個服務不可共用同一輸入資料夾。

## ChipRC 元件篩選與 BOM

推論前先逐一檢查 `Component`，必須同時符合：

1. `Status="False Call"`（由 `watch.failure_statuses` 設定）。Panel 與 Board 的 Status 不作為元件推論條件。
2. `Type` 完整命中 BOM TXT 中的某一行；只移除前後空白，保留大小寫，不做子字串或部分比對，也不使用 PartNo。

例如 BOM 含 `06-210153-05` 時，`Type="06-210153-05"` 會推論，
`Type="06-210153-05-extra"` 或只有 `PartNo="06-210153-05"` 不會推論。
`Fail`、`NG`、`PASS` 或 Type 不在清單內的元件，都保留原有 AI_result（包括空值或原本不存在此屬性的情況）。
跳過的元件不會驗證裁切座標、不等待影像、不推論，也不搬動其影像。

TXT 使用 UTF-8（支援 BOM 標記及 Windows CRLF），每行一個料號，空白行忽略，重複料號只載入一次。
預設檔案為你提供的 62 個料號；可在設定檔更換清單路徑。BOM 遺失、無法讀取或沒有任何料號時會阻止啟動。
BOM 於啟動時載入，修改清單後需重新啟動。由舊版升級時，也請將現有設定檔中的 failure_statuses 改為 `["False Call"]`。

這次提供的線上 XML 元件 `R478_1` 的 Type 為 `RC0402_APL`，不在提供的 BOM 中，依規則會跳過。
若整份 XML 都沒有符合的元件，仍輸出保留原有結果的 XML 並歸檔原始 XML，所有未推論影像保留在輸入資料夾。
日誌與 `/jobs` 的 `component_selection` 會顯示總元件數、選取數，以及依狀態或 BOM 跳過的數量。

### 原 CLI 參數對應

| 原 CLI | JSON 設定 |
| --- | --- |
| `--model` | `inference.model` |
| `--source` | `paths.image_input`；另增加 `paths.xml_input` |
| `--output-dir` | `review.output_dir`；主要 AOI 結果另由 `paths.xml_output` 指定 |
| `--ok-label` | `inference.ok_label`，必須與模型類別名稱完全一致 |
| `--ok-count` | `inference.ok_count`，正整數 |
| `--skip-ok-count` | `inference.ok_count: null`，仍須至少一個 OK 物件 |
| `--ok-overlap-iou` | `inference.ok_overlap_iou`，預設 0.8；0 表示不去重 |
| `--center-tolerance` | `inference.center_tolerance`，以裁切圖中心計算，預設 0.25，範圍 0～0.5 |
| `--ok-angle-target` | `inference.ok_angle_target`，預設 0 度 |
| `--ok-angle-tolerance` | `inference.ok_angle_tolerance`，預設 `null` 關閉限制 |
| `--ok-max-center-distance` | `inference.ok_max_center_distance`，到裁切圖中心的像素距離，預設 `null` |
| `--ok-min-edge-distance` | `inference.ok_min_edge_distance`，裁切圖中物件的像素距離，預設 `null` |
| `--conf` / `--iou` | `inference.conf` / `inference.iou`，預設 0.25 / 0.7 |
| `--imgsz` / `--device` / `--max-det` | `inference.imgsz` / `device` / `max_det`，預設 640 / cpu / 300 |
| `--ext` | `inference.ext`，影像副檔名清單 |
| `--on-exists` | `review.on_exists`，`suffix` / `skip` / `overwrite` |
| `--dry-run` | `runtime.dry_run` |
| `--verbose` / `--quiet` | `runtime.verbose` / `runtime.quiet`，兩者不可同時開啟 |

`review.enabled` 預設 `false`。啟用後，保存裁切圖的框選影像、未畫框裁切圖、VOC XML 及 OBB 量測 JSON，
放在 `review.output_dir/<工作ID>/roi-<範圍ID>/OK/` 或 `NG/<原因>/` 下，與 AOI XML 輸出分開。
`original/` 內的影像是未畫框的裁切圖，與 VOC 尺寸及座標相符；完整原圖保存在 DONE。
VOC 的框為 OBB 的軸對齊外接矩形，旋轉量測保存在 JSON。框與量測座標均以裁切圖左上角為原點。

## Image 裁切座標

每個需要推論的 `Component/CompImage/Image` 都必須提供 `X1`、`Y1`、`X2`、`Y2`：

```xml
<CompImage>
  <Image PicPath="old/path/R478_1_SolderLight.jpg" X1="165" Y1="175" X2="475" Y2="305" />
  <Image PicPath="old/path/R478_1_UniformLight.jpg" X1="165" Y1="175" X2="475" Y2="305" />
</CompImage>
```

線上格式每個 Image 使用 `PicPath` 指向一種光源，各自依該節點的座標裁切；忽略 `PicPathStd` 的標準影像。
同一元件至少需兩個光源影像引用，全部推論 OK 才寫入 OK。亦相容先前一個 Image 包含兩個路徑的格式：

```xml
<Image PicPath1="old/path/top.jpg" PicPath2="old/path/side.jpg"
       X1="10" Y1="10" X2="70" Y2="50" />
```

座標使用**原圖像素**，原點為左上角；`(X1, Y1)` 為裁切左上角，`(X2, Y2)` 為右下邊界。
左／上邊界包含在範圍內，右／下邊界不包含在範圍內；上例產生寬 60、高 40 的裁切圖。
同一 Image 的 PicPath1 與 PicPath2 使用相同座標，分別裁切、分別推論。

- 四個欄位都必須有值且為有限數字；`X1 >= 0`、`Y1 >= 0`、`X2 > X1`、`Y2 > Y1`。
- 支援小數座標：左／上邊界向下取整，右／下邊界向上取整，包含完整指定範圍。
- 每張光源影像都會檢查範圍，`X2` 與 `Y2` 不得超過該影像的寬、高。
- 缺值、非數字、反向或超出影像的範圍視為資料異常：依設定重試，保留 XML 與來源影像，不發布部分結果。
  不會改用整張原圖，也不會自動補邊或截斷超出範圍的座標。
- 所有數量、中心、包含關係與 OBB 距離規則都套用在裁切圖上；YOLO 的 imgsz 縮放由模型處理。
- 相同原圖、相同範圍會共用推論結果；相同原圖的不同範圍分別推論，分別回寫各自的元件。
- 推論只在記憶體中裁切，輸入影像保持完整；歸檔原圖只搬一次。複查裁切圖不影響 DONE 的原圖。

`/jobs` 每筆影像結果包含 `source_image`、`crop` 與 `coordinate_space: "crop"`，
可對應來源檔名、原圖裁切座標，以及以裁切圖為基準的 OBB 量測。

## 處理流程與判定

1. 載入一次 OBB 模型，以單一工作者執行推論；HTTP 狀態查詢在獨立執行緒處理。
2. 掃描現有與新到達的 XML，等待檔案穩定後建立工作紀錄。
3. 逐一檢查 `Panel/Board/Component`，僅選取 False Call 且 Type 在 BOM 內的元件。
4. 取得所選元件 Image 的 `PicPath`，或舊格式的 `PicPath1`、`PicPath2`，將 `/`、`\` 路徑的最後一段作為檔名，
   到 `paths.image_input` 尋找；不使用 `Window` 或 `PicPathStd` 的影像路徑。
5. 等待所有影像到齊且穩定，依每個 Image 的 X1、Y1、X2、Y2 裁切後推論。
   不同 Image 節點也會全部處理；同份 XML 只有「來源檔案與裁切範圍都相同」才共用一次推論。
6. 保留原 CLI 的判定：OK 框去重、精確數量或至少一個 OK、至少一個 OK 中心位於中央區域、
   無完整包含於原始 OK 框內的非 OK 框，以及啟用的 OBB 幾何限制。沒有偵測時為 NG。
   非 OK 框若位於所有 OK 框之外，沿用原程式規則忽略。
7. 兩種光源都完成推論後，全部為 OK 才將該元件寫成 `AI_result="OK"`，任一 NG 即為 `"NG"`。
   第一張 NG 不會跳過第二張；缺圖或執行異常不當作模型 NG。
8. 只修改對應 Component 的 AI_result，其他節點與屬性值保留。XML 會重新序列化為 UTF-8，
   原始 XML 的位元組內容則完整保存於 DONE。XML 的 `Angle` 不會自動取代設定的角度目標。
9. 檢查推論期間來源內容未改變後，先記錄待發布結果，再以同目錄暫存檔發布正式 XML，
   驗證內容雜湊及可解析性後，記錄已輸出狀態。
10. 結果發布成功後才歸檔；歸檔確認完成後移除來源檔，記錄完成並繼續下一份工作。

先前範例包含 17 個 FAIL 元件及 34 個影像引用，依新版預設狀態會跳過。
這份舊範例的 X2、Y2 也都是空值，若將元件納入推論範圍，需由上游補齊像素座標。
測試保留原始範例不變，驗證被選取元件缺座標時不推論、不輸出、不歸檔；另在測試副本中設定
False Call、允許的 Type 與合法範圍，驗證 17 個元件、34 個裁切推論及 Window 的 AI_result 不被修改。
若 XML 沒有需要推論的元件，也會正常輸出及歸檔，不呼叫模型。

## DONE 結構

日期與時間戳來自 XML **檔名**，不使用處理當下日期。
線上 XML 檔名格式為 `YYYYMMDDhhmmss_BoardSN.xml`，不需要最後的 `_T` 或正反面欄位；BoardSN 可以包含底線。
例如 `20261111000316_HA996AQ010070A.xml`，會歸入 `2026-11-11/20261111000316/`。
結果輸出與歸檔都沿用收到的 XML 原檔名，影像檔名依 XML 的 PicPath 或 PicPath1／PicPath2 取得。
也支援這次線上檔名 `20260407133353_20260407 133353T1.xml` 中的空白，不會從 XML 的 BoardSN 重建檔名。

```text
input/xml/DONE/
└── 2026-11-11/
    └── 20261111000316/
        ├── 20261111000316_HA996AQ010070A.xml
        ├── 20261111000316_HA996AQ010070A_T@R7451_5_top.jpg
        ├── 20261111000316_HA996AQ010070A_T@R7451_5_side.jpg
        └── ...其他引用影像
```

原始 XML 與影像同一層，不建立 images 子資料夾。不同板號的同時間戳資料共用時間資料夾。
所有檔案保留原名，歸檔只包含本次需要推論的元件所引用的影像及原始 XML，不搬走無關影像。

歸檔採取「複製 → SHA-256 確認 → 移除來源」，支援跨磁碟。
來源被替換、同名歸檔內容不同或搬移失敗時，保留來源並記錄錯誤。
若影像仍被其他待處理 XML 引用，先保留來源影像；其他 XML 完成後再清理。
遇到尚未解析完成的 XML，會保守延後來源影像清理。

`archive.retention_days` 依**歸檔完成時間**計算，約每小時檢查一次。
只清除工作紀錄中已完成、內容未被修改的歸檔檔案；仍被其他工作保留的共享檔案、未知檔案及輸出 XML 都保留。
請保留 `runtime/jobs.sqlite3`，否則服務會失去歸檔與清理的追蹤紀錄。

## 重試、重啟與檔案衝突

| 狀態 | 行為 |
| --- | --- |
| `waiting` | 等待 XML／影像完整，或等待下次推論重試；其他備齊資料的 XML 可繼續處理 |
| `processing` | 正在推論；服務重啟後回到 waiting |
| `publishing` | 推論已完成，等待結果輸出；重啟後從發布接續，不重新推論 |
| `archiving` | 結果已成功發布，等待歸檔／來源清理；重試不重新推論或再次輸出 |
| `done` | 完成；同一路徑及 XML 內容不重複推論 |
| `failed` | 資料等待逾時或推論／資料驗證嘗試達上限；來源保留 |
| `dry_run` | 已完成預覽推論，未輸出、歸檔或清理來源 |
| `superseded` | 待處理來源被移除或 XML 內容已被新版本取代 |

缺圖、不完整 XML 等待逾時，或影像損壞、模型錯誤達重試上限時，整份 XML 不會發布部分結果。
輸出或歸檔 IO 失敗會持續依 retry_seconds 重試，並在 `/jobs` 顯示原因。

升級推論／篩選規則，或重啟時 BOM 內容、選取狀態設定已變更，尚未發布的舊快取結果會失效，
重新依目前的 BOM、Status 與 ROI 規則處理。
已成功發布的結果不會自動撤回；已完成或正在歸檔的舊工作維持原有紀錄。

結果輸出維持原始檔名：若已有**相同內容**，視為可接續的發布；若已有**不同內容**，
會停在 publishing 並保留來源，請先將既有輸出移走。下游可以在讀取完成後搬走輸出 XML。
歸檔同名且內容不同時也不覆蓋，需先處理衝突檔案。複查影像輸出的 on_exists 不控制 AOI 結果檔。

修復失敗原因後可執行：

```bat
start.bat --retry-failed
```

這會重設 failed 工作的重試與等待時間。相同 XML 刻意重放時，請使用新的時間戳檔名及影像檔名；
已完成的相同輸入不會再推論。

## 查詢與操作

啟動後可開啟：

- `http://127.0.0.1:8000/health`：工作者存活及掃描狀況，異常為 HTTP 503。
- `http://127.0.0.1:8000/status`：目前 XML、各狀態工作數量、最近掃描錯誤。
- `http://127.0.0.1:8000/jobs?limit=50`：最近工作、每張影像判定與量測、失敗原因。
- `http://127.0.0.1:8000/docs`：唯讀 API 文件。

health 表示服務能運作，個別資料的失敗請查看 jobs。API 預設只開放本機，無登入功能。
如需透過管理網路遠端查閱，可設定 `http.host` 為 `0.0.0.0`，並使用 Windows 主機 IP 存取。
日誌輸出到視窗及 `logs/service.log`，每檔 10 MiB，保留 10 份輪替檔。

```bat
rem 驗證設定、模型檔及 BOM 清單（不載入模型）
start.bat --check-config

rem 持續預覽：先將 config.json 的 runtime.dry_run 設為 true
start.bat

rem 單次掃描；檔案穩定等待尚未滿足時將保留 waiting 並回傳非零
start.bat --once
```

dry_run 仍會產生日誌及獨立的工作紀錄，但不輸出 XML、複查影像、歸檔或移除來源，
也不會接續先前正式執行留下的待發布／歸檔工作。改回 false 後可正式處理相同 XML。

## 開發與測試

跨平台測試可在 Python 3.12 虛擬環境執行，無須訓練權重或 PyTorch：

```text
python -m pip install -r requirements-test.txt
python -m pytest -q
```

測試使用可控制的推論替身驗證檔案流程，並用實際 OBB 幾何規則驗證去重、角度、包含關係與距離限制；
模型轉接器以 Ultralytics 介面的替身驗證實際裁切像素、小數座標、範圍檢查、不同 ROI 的獨立判定、
裁切圖中心量測、參數傳遞、OBB 結果轉換與複查輸出。
也使用線上 XML 與 BOM 驗證 False Call／Type 完整比對、跳過元件保持原結果、新版 PicPath 光源格式、
共用影像保留，以及 BOM 更新後未發布結果重新篩選。
這些測試不代表實際模型準確率或 Windows／CUDA 驅動已驗證。

`aoi_service/yolo_rules.py` 保留提供的 `APL-develop-tools/scripts/yolo_classify.py` 中判定與複查輸出函式，
不含其 CLI 入口。`tests/fixtures/` 保存提供的 AOI XML 範例。

完整執行可用 `python -m aoi_service --config config.json` 或 `python main.py`。
程式透過檔案鎖避免同一工作紀錄被重複啟動，沒有使用多個 Uvicorn worker。
