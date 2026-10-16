c++头文件,要用cpp文件编程


git clone https://github.com/eigenteam/eigen-git-mirror
#github安装镜像文件

#安装
cd eigen-git-mirror
mkdir build
cd build
cmake ..
sudo make install

#安装后,头文件安装在/usr/local/include/eigen3/

#这里可以修改安装路径make install prefix=/path/want/to/install/

#移动头文件
sudo cp -r /usr/local/include/eigen3/Eigen /usr/local/include


求解:
1.CG算法默认预处理为Jacobi预处理
2.默认x=0, solveWithGuess()来进行初始化x控制
3.通过 setMaxIterations（） 和 setTolerance（）来控制最大迭代次数和,误差范围
4.通过saveMarket()保存矩阵,loadMarket()加载矩阵

5.预处理目前CG只有Jacobi预处理,NONE预处理,ICCG预处理
Diagonal precondition
Identity precondition
Incomplete LUT,为啥迭代次数是0