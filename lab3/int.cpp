#include <bitset>
#include <iostream>

using namespace std;

int getPlantHealth() { return 100; }

int main() {
  int a = 2000;
  cout << bitset<32>(a) << endl;
  return 0;
}
