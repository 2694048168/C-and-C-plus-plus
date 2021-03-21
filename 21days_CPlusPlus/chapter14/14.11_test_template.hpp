#ifndef TEST_TEMPLATE
#define TEST_TEMPLATE

template <typename T1, typename T2>
class TwoArray
{
public:
  T1& GetT1Element(int index) 
  {
    return T1 arrayOne[index];
  }

  T2& GetT2Element(int index) 
  {
    return T2 arrayOne[index];
  }

private:
  T1 arrayOne [10];
  T2 arrayTwo [10];
};

#endif  // TEST_TEMPLATE