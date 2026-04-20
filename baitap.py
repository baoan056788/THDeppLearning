import numpy as np
import pandas as pd

def giai_bai_tap(bai_so):
    if bai_so == 1:
        print("\n[Bài 1] Kiểm tra tất cả phần tử khác 0")
        arr = np.random.randint(0, 21, size=(3, 3))
        print("Mảng:\n", arr)
        # Xóa  ở dòng dưới đây
        print("Tất cả khác 0:", np.all(arr != 0))

    elif bai_so == 2:
        print("\n[Bài 2] Kiểm tra tồn tại ít nhất một phần tử khác 0")
        arr = np.random.randint(0, 21, size=(3, 3))
        print("Mảng:\n", arr)
        print("Tồn tại phần tử khác 0:", np.any(arr != 0))

    elif bai_so == 3:
        print("\n[Bài 3] So sánh 2 mảng")
        a, b = np.array([1, 5]), np.array([2, 4])
        print(f"a:{a}, b:{b}")
        print("Greater:", np.greater(a, b))
        print("Less Equal:", np.less_equal(a, b))

    elif bai_so == 4:
        print("\n[Bài 4] Tạo mảng 10 số 1, 10 số 0, 10 số 5")
        arr = np.concatenate([np.ones(10), np.zeros(10), np.full(10, 5)])
        print(arr)

    elif bai_so == 5:
        print("\n[Bài 5] Mảng số chẵn [30, 70]")
        print(np.arange(30, 71, 2))

    elif bai_so == 6:
        print("\n[Bài 6] Ma trận đơn vị 3x3")
        print(np.identity(3))

    elif bai_so == 7:
        print("\n[Bài 7] Mảng 10 phần tử [15, 55], bỏ đầu và cuối")
        arr = np.linspace(15, 55, 10)
        print("Mảng gốc:", arr)
        print("Kết quả:", arr[1:-1])

    elif bai_so == 8:
        print("\n[Bài 8] Đổi dấu số trong khoảng [9, 15]")
        arr = np.arange(21)
        arr[(arr >= 9) & (arr <= 15)] *= -1
        print(arr)

    elif bai_so == 9:
        print("\n[Bài 9] Ma trận 3x4 giá trị [10, 21]")
        print(np.random.randint(10, 22, size=(3, 4)))

    elif bai_so == 10:
        print("\n[Bài 10] Ma trận 10x10 biên 1, trong 0")
        arr = np.ones((10, 10))
        arr[1:-1, 1:-1] = 0
        print(arr)

    elif bai_so == 11:
        print("\n[Bài 11] Đường chéo chính 1, 2, 3, 4, 5")
        print(np.diag([1, 2, 3, 4, 5]))

    elif bai_so == 12:
        print("\n[Bài 12] Tổng theo dòng/cột mảng 3x3x3")
        arr = np.random.randint(1, 10, size=(3, 3, 3))
        print("Tổng theo cột (axis 0):\n", arr.sum(axis=0))

    elif bai_so == 13:
        print("\n[Bài 13] Inner product 2 vector")
        v1, v2 = np.random.rand(10), np.random.rand(10)
        print("Dot product:", np.dot(v1, v2))

    elif bai_so == 14:
        print("\n[Bài 14] Cộng vector vào từng dòng ma trận")
        A = np.random.rand(4, 3)
        y = np.random.rand(3)
        print("Kết quả:\n", A + y)

    elif bai_so == 15:
        print("\n[Bài 15] Tính toán trên 2 Series")
        s1 = pd.Series([2, 4, 6, 8, 10])
        s2 = pd.Series([1, 3, 5, 7, 10])
        print("Cộng:\n", s1 + s2)

    elif bai_so == 16:
        print("\n[Bài 16] Chuyển cột đầu thành Series")
        df = pd.DataFrame({'col1': [1, 4, 7], 'col2': [2, 5, 8], 'col3': [3, 6, 9]})
        print(df.iloc[:, 0])

    elif bai_so == 17:
        print("\n[Bài 17] Sắp xếp DataFrame 4x3")
        df = pd.DataFrame(np.random.rand(4, 3), columns=['A', 'B', 'C'])
        print(df.sort_values(by='A'))

    elif bai_so == 18:
        print("\n[Bài 18] Thay đổi index Series")
        s = pd.Series([1, 2, 3], index=['A', 'B', 'C'])
        print(s.reindex(['B', 'A', 'C']))

    elif bai_so == 19:
        print("\n[Bài 19] Phần tử có trong s1 nhưng không có trong s2")
        s1, s2 = pd.Series([1, 2, 3, 4, 5]), pd.Series([2, 4, 6, 8])
        print(s1[~s1.isin(s2)])

    elif bai_so == 20:
        print("\n[Bài 20] Hội và Giao 2 Series")
        x = np.random.randint(0, 10, 20)
        y = np.random.randint(0, 10, 20)
        print("Hội (Union):", np.union1d(x, y))
        print("Giao (Intersect):", np.intersect1d(x, y))

    elif bai_so == 21:
        print("\n[Bài 21] Nối Series dọc và ngang")
        x, y = pd.Series([0, 1, 2]), pd.Series(['p', 'q', 'r'])
        print("Dọc:\n", pd.concat([x, y], axis=0))
        print("Ngang:\n", pd.concat([x, y], axis=1))

    # Dữ liệu cho bài 22-30
    exam_data = {
        'name': ['Anastasia', 'Dima', 'Katherine', 'James', 'Emily', 'Michael', 'Matthew', 'Laura', 'Kevin', 'Jonas'],
        'score': [12.5, 9, 16.5, np.nan, 9, 20, 14.5, np.nan, 8, 19],
        'attempts': [1, 3, 2, 3, 2, 3, 1, 1, 2, 1],
        'qualify': ['yes', 'no', 'yes', 'no', 'no', 'yes', 'yes', 'no', 'no', 'yes']
    }
    labels = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j']
    df = pd.DataFrame(exam_data, index=labels)

    if bai_so == 22: print(df)
    elif bai_so == 23: print(df.head(3))
    elif bai_so == 24: print(df[['name', 'score']])
    elif bai_so == 25: print(df[df['attempts'] > 2])
    elif bai_so == 26: print(f"Rows: {df.shape[0]}, Cols: {df.shape[1]}")
    elif bai_so == 27: print(df[df['score'].isnull()])
    elif bai_so == 28: print(df[df['score'].between(15, 20)])
    elif bai_so == 29: print(df[(df['attempts'] > 2) & (df['score'].between(15, 20))])
    elif bai_so == 30:
        df.loc['d', 'score'] = 19
        print(df.loc[['d']])

while True:
    try:
        chon = int(input("\nNhập số bài (1-30) hoặc 0 để thoát: "))
        if chon == 0: break
        if 1 <= chon <= 30:
            giai_bai_tap(chon)
        else:
            print("Vui lòng nhập từ 1 đến 30.")
    except ValueError:
        print("Vui lòng nhập số nguyên!")