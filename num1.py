import numpy as np

my_list =[1,2,3,4,5,6,7]
print("Python list:",my_list)

#Chuyen list sang numpy array
my_array = np.array(my_list)
print("Numpy array:",my_array)

#Cach 2:
my_array2 = np.array([1,2,3,4])
print("Numpy array 2:",my_array2)

c= np.array([(1,2,3),(4,5,6),(7,8,9)])
print("Numpy array 3:",c)

e= np.zeros((3,4))
print("Numpy array 4:",e)
f= np.ones((3,3))
print("Numpy array 5:",f)

#Reshappe thay doi kich thuoc ma tran
e =np.array([[1.0,2.0,3.0],[4.0,5.0,6.0]])
print(e)

print(e.shape)#Kich thuoc cua ma tran e la 2 hang va 3 cot
f=e.reshape(3,2)
print(f)

print("Sinh mảng ngẫu nhiên:")
g = np.random.randint(100,size=(5,))#Sinh ra 5 số nguyên ngẫu nhiên từ 0 đến 99 và lưu vào mảng g
print(g)

print("Sinh ngẫu nhiêu từ mảng cho trước:")
h = np.random.choice(g,size=(3,))#Sinh ra 3 số nguyên ngẫu nhiên từ mảng g và lưu vào mảng h
print(h)
