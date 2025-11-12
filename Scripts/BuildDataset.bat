@echo off
REM ====================================================================
REM  Tên file: run_pipeline.bat
REM  Mô tả:  Chạy tự động các script Python theo thứ tự.
REM           1. Kích hoạt môi trường ảo (virtual environment)
REM           2. Chạy script xử lý dữ liệu (ví dụ: create_labels.py)
REM           3. Chạy script huấn luyện model (ví dụ: train_model.py)
REM ====================================================================
SETLOCAL

REM ====================================================================
REM                 --- CAU HINH THAM SO ---
REM     Dat duong dan va cac tham so cua ban o day.
REM     LUU Y: Luon su dung dau ngoac kep " " cho duong dan de
REM     tranh loi neu duong dan co dau cach (space).
REM ====================================================================

REM --- Convert HEIC to JPG ---
SET "RAW_DATA_INPUT_DIR=D:\Pill_Identification\Data\RawData"
SET "PROCESSED_DATA_OUTPUT_DIR=D:\Pill_Identification\Data\ProcessedData"

REM --- Resize JPG to given size
SET "RESIZED_DATA_OUTPUT_DIR=D:\Pill_Identification\Data\ProcessedData"
SET "SIZE=640"

REM --- Building dataset ---
SET "INPUT_DATA_DIR=D:\Pill_Identification\Data\ProcessedData"
SET "DATASET_DIR=D:\Pill_Identification\Data\PILL_JPG_2025"
SET "FLATTEN=True"
SET "NUMBER=18"

REM --- Remove background ---

REM --- Add label ---
SET "LABEL_INPUT_DIR=D:\Pill_Identification\Data\BackgroundRemoveData"
SET "LABEL_OUTPUT_DIR=D:\Pill_Identification\Data\LabeledData"
SET "LABEL_FORMAT=.txt"

echo [INFO] Bat dau quy trinh...

REM --- 1. Kich hoat Moi truong ao (VENV) ---
REM Day la buoc rat quan trong de dam bao ban chay dung phien ban thu vien.
REM Hay thay ".venv" bang ten thu muc moi truong ao cua ban (vi du: env, venv, v.v.)
SET "VENV_PATH=.venv\Scripts\activate.bat"

IF EXIST "%VENV_PATH%" (
    echo [INFO] Dang kich hoat moi truong ao tai: %VENV_PATH%
    call "%VENV_PATH%"
) ELSE (
    echo [CANH BAO] Khong tim thay moi truong ao tai '%VENV_PATH%'.
    echo [CANH BAO] Dang su dung Python mac dinh cua he thong.
)

REM --- 1. Run convert script ---
echo [INFO] Running step 1: Convert HEIC to JPG
call python D:\Pill_Identification\Preprocess\ConvertHEICToJPG.py --input "%RAW_DATA_INPUT_DIR%" --output "%PROCESSED_DATA_OUTPUT_DIR%"

REM Kiem tra loi co ban
IF %ERRORLEVEL% NEQ 0 (
    echo [LOI] Script ConvertHEICToJPG.py da xay ra loi!
    GOTO EndScript
)

REM --- 2. Resize images script ---
echo [INFO] Running step 2: Resize image to %SIZE%x%SIZE%
call python D:\Pill_Identification\Preprocess\ResizeImage.py --input "%RESIZED_DATA_OUTPUT_DIR%" --size "%SIZE%"

REM Kiem tra loi co ban
IF %ERRORLEVEL% NEQ 0 (
    echo [LOI] Script ResizeImage.py da xay ra loi!
    GOTO EndScript
)

REM --- 3. Building dataset script ---
echo [INFO] Running step 3: Building dataset
call python D:\Pill_Identification\Preprocess\BuildDataset.py --input "%INPUT_DATA_DIR%" --output "%DATASET_DIR%" -n "%NUMBER%" -f "%FLATTEN%"

REM Kiem tra loi co ban
IF %ERRORLEVEL% NEQ 0 (
    echo [LOI] Script BuildDataset.py da xay ra loi!
    GOTO EndScript
)

REM --- 4. Remove background ---
echo [INFO] Running step 4: Renmove background
call python D:\Pill_Identification\background_removal_DL\u2net_image.py

REM Kiem tra loi co ban
IF %ERRORLEVEL% NEQ 0 (
    echo [LOI] Script u2net_image.py da xay ra loi!
    GOTO EndScript
)

REM --- 5. Create label ---
echo [INFO] Running step 5: Create label
call python D:\Pill_Identification\Preprocess\Labeling.py --input "%LABEL_INPUT_DIR%" --output "%LABEL_OUTPUT_DIR%" --type "%LABEL_FORMAT%"

REM Kiem tra loi co ban
IF %ERRORLEVEL% NEQ 0 (
    echo [LOI] Script Labeling.py da xay ra loi!
    GOTO EndScript
)

REM --- 5. Hoan thanh ---
echo [THANH CONG] Tat ca cac script da chay hoan tat.

:EndScript
echo [INFO] Nhan phim bat ky de ket thuc.
pause > nul

ENDLOCAL