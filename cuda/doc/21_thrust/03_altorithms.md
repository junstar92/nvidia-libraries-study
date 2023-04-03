# Table of Contents

- [Table of Contents](#table-of-contents)
- [Algorithms](#algorithms)
  - [Transformations](#transformations)
  - [Reductions](#reductions)
  - [Prefix-Sums](#prefix-sums)
  - [Reordering](#reordering)
  - [Sorting](#sorting)
- [References](#references)

<br>

# Algorithms

Thrust에서는 다양한 common parallel algorithms을 제공한다. 제공되는 알고리즘 중 다수는 STL을 참조하고 있으며, STL 함수와 동일한 이름의 thrust 알고리즘이 있을 때 네임스페이스를 통해 선택하면 된다 (e.g. `thrust::sort`와 `std::sort`).

Thrust의 모든 알고리즘은 host와 device에 대해 모두 구현되어 있다. Host iterator가 전달되어 호출되면 host path로 디스패치되며, 알고리즘의 range를 정의할 때 device iterator가 사용되면 device 구현으로 호출된다.

`thrust::copy`를 제외한(host와 device 간의 복사) 모든 thrust 알고리즘에 대한 모든 iterator 인자들은 동일한 위치에 있어야 한다. 즉, 모두 host memory에 있거나, 모두 device memory에 있어야 한다. 이 요구사항이 위반되면 컴파일 에러가 발생한다.

## Transformations

Transformations는 입력 범위의 각 요소에 어떤 연산을 적용하는 알고리즘이며, 결과를 전달된 결과 범위에 저장한다. [Vectors](/cuda/doc/21_thrust/02_vectors.md)에서 이미 `thrust::fill`에 대해서 간단히 살펴봤는데, 이 함수는 지정된 범위의 모든 요소들을 특정 값으로 설정한다. 다른 transformation 함수로는 `thrust::sequence`, `thrust::replace`, `thrust::transform` 등이 있다.

> Transformation 알고리즘 리스트는 [Transformations](https://thrust.github.io/doc/group__transformations.html)를 참조

아래 코드는 몇 가지 transformation 알고리즘을 사용하는 방법에 대해서 보여준다. `thrust::negate`와 `thrust::modulus`는 C++에서 사용하는 functor이다. Thrust에서 사용할 수 있는 일반적인 functor는 `thrust/functional.h`에서 제공한다.
```c++
#include <thrust/device_vector.h>
#include <thrust/transform.h>
#include <thrust/sequence.h>
#include <thrust/copy.h>
#include <thrust/fill.h>
#include <thrust/replace.h>
#include <thrust/functional.h>
#include <iostream>

int main(void)
{
    // allocate three device_vectors with 10 elements
    thrust::device_vector<int> X(10);
    thrust::device_vector<int> Y(10);
    thrust::device_vector<int> Z(10);

    // initialize X to 0,1,2,3, ....
    thrust::sequence(X.begin(), X.end());

    // compute Y = -X
    thrust::transform(X.begin(), X.end(), Y.begin(), thrust::negate<int>());

    // fill Z with twos
    thrust::fill(Z.begin(), Z.end(), 2);

    // compute Y = X mod 2
    thrust::transform(X.begin(), X.end(), Z.begin(), Y.begin(), thrust::modulus<int>());

    // replace all the ones in Y with tens
    thrust::replace(Y.begin(), Y.end(), 1, 10);

    // print Y
    thrust::copy(Y.begin(), Y.end(), std::ostream_iterator<int>(std::cout, "\n"));

    return 0;
}
```

`thrust/functional.h`에서 제공하는 functor는 대부분의 산술 및 비교 연산을 지원한다. 다른 연산, 예를 들어, `y <- a * x + y`와 같은 벡터 연산(BLAS's SAXPY 연산)이 필요할 수도 있다. 이러한 연산들에 대해서 몇 가지 옵션들이 있는데, 첫 번째는 2개의 transformation(addition and multiplication)을 `a` 값으로 채워진 임시 벡터와 함께 사용하는 것이다. 조금 더 괜찮은 두 번째 방법은 원하는 연산을 수행하는 user-defined functor를 사용하여 하나의 transformation으로 사용하는 것이다. SAXPY 연산 구현은 아래와 같이 구현할 수 있다.
```c++
struct saxpy_functor
{
    const float a;

    saxpy_functor(float _a) : a(_a) {}

    __host__ __device__
        float operator()(const float& x, const float& y) const {
            return a * x + y;
        }
};

void saxpy_fast(float A, thrust::device_vector<float>& X, thrust::device_vector<float>& Y)
{
    // Y <- A * X + Y
    thrust::transform(X.begin(), X.end(), Y.begin(), Y.begin(), saxpy_functor(A));
}

void saxpy_slow(float A, thrust::device_vector<float>& X, thrust::device_vector<float>& Y)
{
    thrust::device_vector<float> temp(X.size());

    // temp <- A
    thrust::fill(temp.begin(), temp.end(), A);

    // temp <- A * X
    thrust::transform(X.begin(), X.end(), temp.begin(), temp.begin(), thrust::multiplies<float>());

    // Y <- A * X + Y
    thrust::transform(temp.begin(), temp.end(), Y.begin(), Y.begin(), thrust::plus<float>());
}
```

위의 `saxpy_fast`와 `saxpy_slow` 구현은 모두 유효한 SAXPY 구현이지만, `saxpy_fast`가 훨씬 더 빠르다. `temp` 벡터 할당에 대한 비용을 무시하면, 두 함수 구현에서의 연산 비용은 다음과 같다.

- `saxpy_fast` : 2N reads and N writes
- `saxpy_slow` : 4N reads and 3N writes

SAXPY는 memory bound(성능이 memory bandwidth에 의해 제한됨)이므로 메모리 read/write 횟수가 많을수록 `saxpy_slow`의 비용은 훨씬 더 커진다. 반대로 `saxpy_fast`의 경우에는 최적화된 BLAS 구현에서의 SAXPY만큼 빠르다.

> SAXPY와 같은 memory bound 알고리즘에서는 일반적으로 memory transaction의 수를 최소화하기 위해 kernel fusion(여러 작업을 단일 커널로 결합)을 적용하는 것이 좋다.

`thrust::transform`은 오직 하나 또는 두 개의 입력 인자에 대한 연산만 제공한다 ($f(x) \rightarrow y$ and $f(x, x) \rightarrow y$). 만약 두 개 이상의 인자에 대한 연산이 필요하다면 다른 접근 방식이 필요하다. [arbitrary_transformation](https://github.com/NVIDIA/thrust/blob/master/examples/arbitrary_transformation.cu) 예제 코드에서는 `thrust::zip_iterator`와 `thrust::for_each`를 사용한 방법을 보여준다.

## Reductions

Reduction 알고리즘은 binary operation을 사용하여 input sequence를 하나의 값으로 줄인다. 예를 들면, plus operation을 사용하여 배열 값들의 합을 구하는 sum reduction이 있다. Thrust에서는 `thrust::reduce`를 아래와 같이 사용하여 배열의 합을 구할 수 있다.
```c++
int sum = thrust::reduce(D.begin(), D.end(), (int)0, thrust::plus<int>());
```
첫 번째와 두 번째 인자는 알고리즘을 적용할 값의 범위이며, 세 번째와 네 번째 인자는 초깃값과 reduction operator이다. 이런 reduction 연산들은 일반적으로 기본값 또는 operator 없이 사용할 수도 있다. 아래의 코드는 모두 동일하다.
```c++
int sum = thrust::reduce(D.begin(), D.end(), (int)0, thrust::plus<int>());
int sum = thrust::reduce(D.begin(), D.end(), (int)0);
int sum = thrust::reduce(D.begin(), D.end());
```

`thrust::reduce`와 여러 operator를 사용하여 다양한 reduction을 수행할 수 있는데, 편의성을 위해 thrust에서는 STL에서 제공하는 것과 같이 몇 가지 함수들을 추가로 제공한다. 예를 들어, `thrust::count`는 주어진 시퀀스에 대한 특정 값이 몇 개 있는지 반환한다.
```c++
#include <thrust/count.h>
#include <thrust/device_vector.h>
...
// put three 1s in a device_vector
thrust::device_vector<int> vec(5,0);
vec[1] = 1;
vec[3] = 1;
vec[4] = 1;

// count the 1s
int result = thrust::count(vec.begin(), vec.end(), 1);
// result is three
```
이외에도 `thrust::count_if`, `thrust::min_element`, `thrust::max_element`, `thrust::is_sorted`, `thrust::inner_product` 등의 연산들을 지원한다. 제공되는 전체 연산 리스트는 [Reductions](https://thrust.github.io/doc/group__reductions.html)에서 확인할 수 있다.

[Transformations](#transformations)에서 SAXPY 예제를 통해 kernel fusion이 어떻게 적용될 수 있는지를 살펴봤는데, `thrust::transform_reduce`를 사용하여 reduction kernel에 kernel fusion을 적용할 수 있다. 아래 예제 코드는 이를 적용하여 벡터의 norm을 계산하는 코드를 구현한다.
```c++
#include <thrust/transform_reduce.h>
#include <thrust/functional.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <cmath>

// square<T> computes the square of a number f(x) -> x*x
template <typename T>
struct square
{
    __host__ __device__
        T operator()(const T& x) const {
            return x * x;
        }
};

int main(void)
{
    // initialize host array
    float x[4] = {1.0, 2.0, 3.0, 4.0};

    // transfer to device
    thrust::device_vector<float> d_x(x, x + 4);

    // setup arguments
    square<float>        unary_op;
    thrust::plus<float> binary_op;
    float init = 0;

    // compute norm
    float norm = std::sqrt( thrust::transform_reduce(d_x.begin(), d_x.end(), unary_op, init, binary_op) );

    std::cout << norm << std::endl;

    return 0;
}
```

위 예제 코드에서 `thrust::transform_reduce`는 먼저 벡터의 각 요소에 `unary_op`(square 연산)을 적용한 뒤, `binary_op`로 reduction을 수행한다. 이는 multiple-pass로 구현하는 것보다 더 빠르다.

## Prefix-Sums

Parallel prefix-sums(or scan operations)는 stream compaction과 radix sort와 같은 병렬 알고리즘에서 중요한 building block이다. 아래 예제 코드는 기본으로 사용되는 `plus` operator를 사용하여 inclusive scan 연산을 수행한다.
```c++
#include <thrust/scan.h>

int data[6] = {1, 0, 2, 2, 1, 3};

thrust::inclusive_scan(data, data + 6, data); // in-place scan

// data is now {1, 1, 3, 5, 6, 9}
```

Exclusive scan 연산의 경우에는 다음과 같이 작성할 수 있다.
```c++
#include <thrust/scan.h>

int data[6] = {1, 0, 2, 2, 1, 3};

thrust::exclusive_scan(data, data + 6, data); // in-place scan

// data is now {0, 1, 1, 3, 5, 6}
```

Thrust에서는 scan 연산을 수행하기 전에 unary function을 input sequence에 적용하는 `transform_exclusive_scan`과 `transform_inclusive_scan`도 제공한다. 전체 리스트는 [Prefix Sums](https://thrust.github.io/doc/group__prefixsums.html)에서 제공한다.

## Reordering

Thrust에는 아래의 알고리즘을 통해 partitioning과 stream compaction을 지원한다.

- `copy_if` : copy elements that pass a predicate test
- `partition` : reorder elements according to a predicate (true values precede false values)
- `remove` and `remove_if` : remove elements that fail a predicate test
- `unique` : remove consecutive duplicates within a sequence

전체 알고리즘 리스트는 [Reordering](https://thrust.github.io/doc/group__reordering.html)에서 확인할 수 있다.

## Sorting

Thrust에는 주어진 조건에 따라 데이터를 정렬/재정렬하는 몇 가지 함수들을 제공한다. `thrust::sort`와 `thrust::stable_sort` 함수는 STL의 `std::sort`와 `std::stable_sort`에 대응한다.

```c++
#include <thrust/sort.h>

...
const int N = 6;
int A[N] = {1, 4, 2, 8, 5, 7};

thrust::sort(A, A + N);

// A is now {1, 2, 4, 5, 7, 8}
```

또한, key-value로 정렬하는 `thrust:sort_by_key`와 `thrust::stable_sort_by_key`도 제공한다.
```c++
#include <thrust/sort.h>

...
const int N = 6;
int    keys[N] = {  1,   4,   2,   8,   5,   7};
char values[N] = {'a', 'b', 'c', 'd', 'e', 'f'};

thrust::sort_by_key(keys, keys + N, values);

// keys is now   {  1,   2,   4,   5,   7,   8}
// values is now {'a', 'c', 'b', 'e', 'f', 'd'}
```

STL과 동일하게, 정렬 함수는 user-defined comparision operator를 전달받을 수 있다.
```c++
#include <thrust/sort.h>
#include <thrust/functional.h>

...
const int N = 6;
int A[N] = {1, 4, 2, 8, 5, 7};

thrust::stable_sort(A, A + N, thrust::greater<int>());

// A is now {8, 7, 5, 4, 2, 1}
```

<br>

# References

- [NVIDIA CUDA Documentation: Thrust - Algorithms](https://docs.nvidia.com/cuda/thrust/index.html#algorithms)
- [Thrust Doxygen Documentation](https://thrust.github.io/doc/)