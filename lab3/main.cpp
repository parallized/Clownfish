#include <iostream>

using namespace std;

// g++ lab3/main.cpp -o lab3/main.exe

// g++.exe 是一个程序
// main.exe 是一个程序
// main.cpp 是一个源文件

// arg = argument 参数
// argc = argument count 参数个数
// argv = argument vector 参数数组

// 主入口函数：程序的入口点
int main(int argc, char *argv[]) {
  for (int i = 0; i < argc; i++) {
    cout << "ARG" << i << ": " << argv[i] << endl;
  }
  return 0;
}
