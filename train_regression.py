import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
# Khai bao thu vien kiem tra kieu du lieu chuan cua Pandas
from pandas.api.types import is_object_dtype, is_string_dtype, is_categorical_dtype

# =====================================================================
# BUOC 1 & 2: DOC VA TIEN XU LY DU LIEU THONG MINH (TUONG THICH PYTHON 3.14+)
# =====================================================================
def load_and_preprocess_data(filename="superstore_sales.csv"):
    print(f"\n--- [Buoc 1 & 2]: Doc & Tien xu ly du lieu tu file {filename} ---")
    
    if not os.path.exists(filename):
        raise FileNotFoundError(f"Khong tim thay file '{filename}'. Hay dam bao file nay nam cung thu muc voi file code Python.")
        
    df = pd.read_csv(filename)
    print(f"Kich thuoc du lieu tho ban dau: {df.shape}")
    
    # Loai bo khoang trang thua o ten cot
    df.columns = df.columns.str.strip()
    
    # 1. Tu dong tim kiem cot muc tieu
    target_col = None
    for col in ['profit', 'Profit', 'PROFIT']:
        if col in df.columns:
            target_col = col
            break
            
    if target_col is None:
        raise KeyError(f"Khong tim thay cot profit/Profit trong file. Cac cot hien tai: {list(df.columns)}")
        
    df = df.dropna(subset=[target_col])
    print(f"Kich thuoc sau khi xoa hang khuyet o cot muc tieu '{target_col}': {df.shape}")
    
    # 2. Tu dong xu ly cot ngay thang neu co xuat hien
    date_col = None
    for col in ['Order Date', 'date', 'Date', 'Order_Date']:
        if col in df.columns:
            date_col = col
            break
            
    if date_col is not None:
        df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
        df['date_dayofyear'] = df[date_col].dt.dayofyear.fillna(0)
        print(f"[+] Da tu dong xu ly cot ngay thang '{date_col}' thanh dac trung so.")
    else:
        print("[!] Khong thay cot ngay thang, bo qua buoc xu ly thoi gian.")
        
    # Danh sach cac cot ID hoac van ban dac thiet khong dung de suy luan mo hinh
    ignored_cols = [target_col, 'Row ID', 'Order ID', 'Customer ID', 'Customer Name', 'Product ID', 'Product Name']
    if date_col is not None:
        ignored_cols.append(date_col)
        
    # 3. TU DONG TIM CAC COT CHU BANG HAM CHUAN CUA PANDAS API (SUA LOI TRIET DE)
    categorical_columns = []
    for col in df.columns:
        if col not in ignored_cols:
            # Loai bo khoang trang thua neu cột do chua du lieu chuoi
            if is_object_dtype(df[col]) or is_string_dtype(df[col]):
                df[col] = df[col].astype(str).str.strip()
                categorical_columns.append(col)
            elif is_categorical_dtype(df[col]):
                categorical_columns.append(col)
            
    if len(categorical_columns) > 0:
        # Ep kieu du lieu dummies ra dang so nguyen int (0 va 1) de chac chan khong bi loi chuoi
        df = pd.get_dummies(df, columns=categorical_columns, drop_first=True, dtype=int)
        print(f"[+] Da tu dong phat hien va ma hoa One-hot cho cac cot chu: {categorical_columns}")
    
    # 4. Gom cac cot dac trung so hoc con lai vao bien X va muc tieu vao bien y
    feature_columns = [col for col in df.columns if col not in ignored_cols]
    
    X = df[feature_columns].values.astype(np.float32)
    y = df[target_col].values.astype(np.float32).reshape(-1, 1)
    
    print(f"Tong so luong tinh chat dau vao (Features) dung de huan luyen: {X.shape[1]}")
    print(f"Danh sach cac cot dac trung: {feature_columns}")
    
    # 5. Chia tap du lieu thanh Train (80%) va Validation (20%)
    if len(df) < 5:
        print("[!] Tap du lieu thu nghiem qua nho, dung chung cho ca Train va Validation.")
        X_train, X_val, y_train, y_val = X, X, y, y
    else:
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 6. Chuan hoa mien gia tri (Standardization)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    
    return X_train, X_val, y_train, y_val

# =====================================================================
# BUOC 3: KHOI TAO DATALOADER TRONG PYTORCH
# =====================================================================
def build_pytorch_dataloaders(X_train, X_val, y_train, y_val, batch_size=16):
    print("\n--- [Buoc 3]: Dong goi du lieu thanh cac Batch & Thiet lap DataLoader ---")
    
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32)
    X_val_t = torch.tensor(X_val, dtype=torch.float32)
    y_val_t = torch.tensor(y_val, dtype=torch.float32)
    
    train_dataset = TensorDataset(X_train_t, y_train_t)
    val_dataset = TensorDataset(X_val_t, y_val_t)
    
    current_batch_size = min(batch_size, len(X_train))
    
    train_loader = DataLoader(train_dataset, batch_size=current_batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=current_batch_size, shuffle=False)
    
    print(f"Kich thuoc Batch duoc dung: {current_batch_size}")
    print(f"So luong Batches trong tap Train: {len(train_loader)}")
    return train_loader, val_loader

# =====================================================================
# BUOC 4: KIEU KIEN TRUC MANG NOI-RON HOI QUY (MLP REGRESSOR)
# =====================================================================
class ProfitMLPRegressor(nn.Module):
    def __init__(self, input_dim):
        super(ProfitMLPRegressor, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 64),      # Lop an 1: 64 nodes
            nn.ReLU(),                     # Chuyen doi phi tuyen tinh
            nn.Dropout(0.2),               # Chong hoc vet
            
            nn.Linear(64, 32),             # Lop an 2: 32 nodes
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(32, 1)               # Lop du doan dau ra (1 gia tri profit)
        )
        
    def forward(self, x):
        return self.network(x).squeeze(-1)

# =====================================================================
# BUOC 5: TIEN HANH CHAY VONG LAP HUAN LUYEN (TRAINING LOOP)
# =====================================================================
def execute_model_training(model, train_loader, criterion, optimizer, total_epochs=100):
    print("\n--- [Buoc 5]: Bat dau qua trinh huan luyen ---")
    
    for epoch in range(total_epochs):
        model.train()  
        accumulated_loss = 0.0
        
        for batch_features, batch_targets in train_loader:
            optimizer.zero_grad()
            output_predictions = model(batch_features)
            loss_value = criterion(output_predictions, batch_targets.squeeze(-1))
            loss_value.backward()
            optimizer.step()
            
            accumulated_loss += loss_value.item() * batch_features.size(0)
            
        average_epoch_loss = accumulated_loss / len(train_loader.dataset)
        
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch [{epoch+1:03d}/{total_epochs}] | MSE Loss: {average_epoch_loss:.4f}")

# =====================================================================
# BUOC 6: DOC THONG SO VA DANH GIA SAI SO
# =====================================================================
def evaluate_trained_model(model, val_loader):
    print("\n--- [Buoc 6]: Tien hanh danh gia sai so tren tap Validation ---")
    
    model.eval()  
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for batch_features, batch_targets in val_loader:
            preds = model(batch_features)
            all_predictions.extend(preds.numpy())
            all_targets.extend(batch_targets.squeeze(-1).numpy())
            
    all_predictions = np.array(all_predictions)
    all_targets = np.array(all_targets)
    
    mae_score = np.mean(np.abs(all_predictions - all_targets))
    
    residual_sum_squares = np.sum((all_targets - all_predictions) ** 2)
    total_sum_squares = np.sum((all_targets - np.mean(all_targets)) ** 2)
    
    if total_sum_squares == 0:
        r2_metric = 1.0
    else:
        r2_metric = 1 - (residual_sum_squares / total_sum_squares)
    
    print("====================================================")
    print(f"[*] KET QUA KIEM THU:")
    print(f" -> Mean Absolute Error (MAE): {mae_score:.4f}")
    print(f" -> R2 Score (He so xac dinh): {r2_metric:.4f}")
    print("====================================================")

# =====================================================================
# DIEM CHAY CHUONG TRINH CHINH (MAIN)
# =====================================================================
if __name__ == "__main__":
    
    # Dang chay test voi file nho:
    TARGET_DATA_FILE = "sample_sales_data.csv" 
    
    # Chú ý: Khi nào chạy file lớn chính thức của bài Lab thì bỏ dấu # ở dòng dưới ra nhé:
    # TARGET_DATA_FILE = "superstore_sales.csv" 
    
    X_train, X_val, y_train, y_val = load_and_preprocess_data(TARGET_DATA_FILE)
    train_loader, val_loader = build_pytorch_dataloaders(X_train, X_val, y_train, y_val, batch_size=16)
    
    input_features_count = X_train.shape[1]
    regression_model = ProfitMLPRegressor(input_dim=input_features_count)
    
    loss_criterion = nn.MSELoss()
    weight_optimizer = optim.Adam(regression_model.parameters(), lr=0.01)
    
    execute_model_training(regression_model, train_loader, loss_criterion, weight_optimizer, total_epochs=100)
    evaluate_trained_model(regression_model, val_loader)