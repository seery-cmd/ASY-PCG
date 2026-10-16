import ssgetpy
##[序号,Group,row,col,NNZ,存储类型,2D/3D Discretization,SPD(对称正定),图像对称性,数值对称性
min_row=1000         #最小行数
max_row=1000000       #最大行数
# min_row=715176         #最小行数
# max_row=715176      #最大行数
limit_num=300           #下载个数
is_spd=True          #是否正定
data_type='real'      #数据类型-实数
fotmat_type='MM'       #下载矩阵存储格式-mtx
download_path="/home/hw_EMatrix/seery/src/dataset" #下载路径
name="ex5"

result1 = ssgetpy.search(rowbounds=(min_row,max_row),limit=limit_num,isspd=is_spd,dtype=data_type)

#result1 = ssgetpy.search(name_or_id=name,limit=limit_num)
number=1
for i in result1:
    print("number:",number,i)
    i.download(format=fotmat_type, destpath=download_path,extract=True)
    print("下载成功,DONE")
    number=number+1



