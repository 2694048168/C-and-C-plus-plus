#include <iostream>

// test
class Fish
{
public:
  virtual void Swim()
  {
    std::cout << "Fish swims ini water." << std::endl;
  }

  // base class should always have virtual destructor
  virtual ~Fish() {}
};

class Tuna : public Fish
{
public:
  void Swim()
  {
    std::cout << "Tuna swims real fast in the sea." << std::endl;
  }

  void BecomeDinner()
  {
    std::cout << "Tuna become dinner in Sushi." << std::endl;
  }
};

class Carp : public Fish
{
public:
  void Swim()
  {
    std::cout << "Carp swims real slow in the lake." << std::endl;
  }

  void Talk()
  {
    std::cout << "Carp talked Carp!" << std::endl;
  }
};

void DetectFishType(Fish* objFish)
{
  Tuna* objTuna = dynamic_cast<Tuna*>(objFish);
  // if (objTuna) to check success of cast
  if (objTuna)
  {
    std::cout << "Detected Tuna. Making Tuna dinner: " << std::endl;
    objTuna->BecomeDinner();
  }

  Carp* objCarp = dynamic_cast<Carp*>(objFish);
  // if (objTuna) to check success of cast
  if (objCarp)
  {
    std::cout << "Detected Carp. Making Tuna dinner: " << std::endl;
    objCarp->Talk();
  }

  std::cout << "Verifying type using virtual Fish::Swim: " << std::endl;
  objFish->Swim(); // calling virtual function
}

int main(int argc, char** argv)
{
  Fish* ptrFish = new Tuna;
  // 使用 static_cast 进行类型转换
  Tuna* ptrTuna = static_cast<Tuna*> (ptrFish);

  // Tuna::BecomeDinner will work only using valid Tuna*
  ptrTuna->BecomeDinner();

  // virtual destructor in Fish ensures invocation of ~Tuna()
  delete ptrFish;
  
  return 0;
}

// $ g++ -o main 13.2_test_stctic_cast.cpp 
// $ ./main.exe 
// Tuna become dinner in Sushi