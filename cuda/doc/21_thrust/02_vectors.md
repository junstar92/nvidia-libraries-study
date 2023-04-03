# Table of Contents

- [Table of Contents](#table-of-contents)
- [Vectors](#vectors)
  - [Thrust Namespace](#thrust-namespace)
  - [Iterators and Static Dispatching](#iterators-and-static-dispatching)
- [References](#references)

<br>

# Vectors

Thrust에서는 2가지 벡터 컨테이너, `host_vector`와 `device_vector`를 제공한다. 이름에서 알 수 있듯이 `host_vector`는 host memory에 저장되고, `device_vector`는 GPU device memory에 저장된다. 이 벡터 컨테이너는 C++ STL의 `std::vector`와 같으며, 당연히 제너릭 컨테이너이면서 동적으로 사이즈를 조절할 수 있다. 아래 코드는 thrust vector 컨테이너를 사용하는 방법을 보여준다.

```c++
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include <iostream>

int main(void)
{
    // H has storage for 4 integers
    thrust::host_vector<int> H(4);

    // initialize individual elements
    H[0] = 14;
    H[1] = 20;
    H[2] = 38;
    H[3] = 46;

    // H.size() returns the size of vector H
    std::cout << "H has size " << H.size() << std::endl;

    // print contents of H
    for(int i = 0; i < H.size(); i++)
        std::cout << "H[" << i << "] = " << H[i] << std::endl;

    // resize H
    H.resize(2);

    std::cout << "H now has size " << H.size() << std::endl;

    // Copy host_vector H to device_vector D
    thrust::device_vector<int> D = H;

    // elements of D can be modified
    D[0] = 99;
    D[1] = 88;

    // print contents of D
    for(int i = 0; i < D.size(); i++)
        std::cout << "D[" << i << "] = " << D[i] << std::endl;

    // H and D are automatically deleted when the function returns
    return 0;
}
```

위 예제에서 `=` 연산자는 `host_vector`를 `device_vector`로 복사하는데 사용할 수 있으며, 그 반대도 동일하다. 물론, `host_vector` to `host_vector`와 `device_vector` to `device_vector`도 가능하다. Host 측에서 `device_vector`의 각 요소는 대괄호와 인덱스를 통해 액세스할 수 있다. 하지만, 각 액세스에서는 `cudaMemcpy` 호출이 사용되므로 자주 사용하는 것은 좋지 않다. 아래에서 조금 더 효율적인 기법들에 대해서 살펴본다.

벡터의 모든 요소를 특정 값으로 초기화하거나 한 벡터에서 다른 벡터로 특정 부분 집합의 값들만 복사하는 것이 종종 사용된다. 이를 위해 Thrust에서는 몇 가지 방법들을 제공한다.

```c++
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include <thrust/copy.h>
#include <thrust/fill.h>
#include <thrust/sequence.h>

#include <iostream>

int main(void)
{
    // initialize all ten integers of a device_vector to 1
    thrust::device_vector<int> D(10, 1);

    // set the first seven elements of a vector to 9
    thrust::fill(D.begin(), D.begin() + 7, 9);

    // initialize a host_vector with the first five elements of D
    thrust::host_vector<int> H(D.begin(), D.begin() + 5);

    // set the elements of H to 0, 1, 2, 3, ...
    thrust::sequence(H.begin(), H.end());

    // copy all of H back to the beginning of D
    thrust::copy(H.begin(), H.end(), D.begin());

    // print D
    for(int i = 0; i < D.size(); i++)
        std::cout << "D[" << i << "] = " << D[i] << std::endl;

    return 0;
}
```
위 예제 코드에서는 `fill`, `copy`, `sequence` 함수를 사용했다. `fill`과 `copy`는 STL 알고리즘과 사용 방법이 동일하다. `sequence`의 경우에는 `std::iota`와 같다.

## Thrust Namespace

위에서 살펴본 예제 코드에서는 `thurst::host_vector` 또는 `thrust::copy`와 같이 사용했다. `thrust::`는 c++에서의 namespace이며, 당연히 STL과 구분하여 사용할 수 있도록 해준다.

## Iterators and Static Dispatching

C++에서 컨테이너의 `begin()`이나 `end()` 멤버 함수의 결과는 iterator이다. 벡터 컨테이너의 경우는 실제로 배열이기 때문에 iterator는 배열 요소에 대한 포인터로 생각할 수 있다. 따라서, `H.begin()`은 H 벡터 내부에 저장된 배열의 첫 번째 요소를 가리키는 iterator이다. 마찬가지로 `H.end()`는 H 벡터의 마지막 요소의 다음 위치를 가리킨다.


벡터의 iterator는 포인터와 유사하지만, 조금 더 많은 정보를 전달한다. 예를 들어, 아래 코드에서 `thrust::fill` 함수는 `device_vector`의 iterator에서 동작할 것이다.
```c++
// initialize all ten integers of a device_vector to 1
thrust::device_vector<int> D(10, 1);

// set the first seven elements of a vector to 9
thrust::fill(D.begin(), D.begin() + 7, 9);
```
`thrust::fill`이 device에서 동작한다는 사실은 `D.begin()`이 반환한 iterator의 타입을 통해 알아낸다. Thrust 함수가 호출되면 iterator의 타입을 검사하여 host 구현 또는 device 구현을 사용할지를 결정하게 된다. 이러한 host/device dispatch는 컴파일 타입에 결정되므로, 이에 대한 런타임 오버헤드는 없다.

STL 알고리즘의 경우, raw 포인터를 전달할 수도 있다. STL과 마찬가지로 Thrust에서도 raw 포인터를 전달할 수 있는데, 기본적으로는 host로 디스패치한다. 만약 device memory에서 수행하고자 한다면, raw 포인터를 `thrust::device_ptr`로 래핑하여 전달해야 한다.
```c++
size_t N = 10;

// raw pointer to device memory
int * raw_ptr;
cudaMalloc((void **) &raw_ptr, N * sizeof(int));

// wrap raw pointer with a device_ptr
thrust::device_ptr<int> dev_ptr(raw_ptr);

// use device_ptr in thrust algorithms
thrust::fill(dev_ptr, dev_ptr + N, (int) 0);
```

반대로 `device_ptr`로부터 raw 포인터를 추출하려면, 아래와 같이 `raw_pointer_cast`를 사용해야 한다.
```c++
size_t N = 10;

// create a device_ptr
thrust::device_ptr<int> dev_ptr = thrust::device_malloc<int>(N);

// extract raw pointer from device_ptr
int * raw_ptr = thrust::raw_pointer_cast(dev_ptr);
```

Iterator와 Pointer를 구분하는 또 다른 이유는 iterator를 사용하면 많은 종류의 데이터 구조를 탐색할 수 있다는 것이다. 예를 들어, STL의 경우에는 bidirectional iterator를 제공하는 링크드 리스트 컨테이너(`std::list`)를 지원한다. Thrust에서 이러한 컨테이너에 대한 device 구현은 제공하지 않지만, 이들과 호환은 된다 (아래 예제 코드 참고).
```c++
#include <thrust/device_vector.h>
#include <thrust/copy.h>
#include <list>
#include <vector>

int main(void)
{
    // create an STL list with 4 values
    std::list<int> stl_list;

    stl_list.push_back(10);
    stl_list.push_back(20);
    stl_list.push_back(30);
    stl_list.push_back(40);

    // initialize a device_vector with the list
    thrust::device_vector<int> D(stl_list.begin(), stl_list.end());

    // copy a device_vector into an STL vector
    std::vector<int> stl_vector(D.size());
    thrust::copy(D.begin(), D.end(), stl_vector.begin());

    return 0;
}
```

> 지금까지 위에서 언급한 iterator는 유용하지만 상당히 기본적인 것들만 다루었다. 이러한 기본 iterator 이외에도 thrust에는 `counting_iterator`와 `zip_iterator`와 같은 fancy iterator를 제공한다 ([Fancy Iterators](/cuda/doc/21_thrust/04_fancy_iterators.md) 참조).

<br>

# References

- [NVIDIA CUDA Documentation: Thrust - Vectors](https://docs.nvidia.com/cuda/thrust/index.html#vectors)