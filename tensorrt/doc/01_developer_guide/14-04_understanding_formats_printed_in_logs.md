# Table of Contents

- [Table of Contents](#table-of-contents)
- [Understanding Formats Printed in Logs](#understanding-formats-printed-in-logs)
- [References](#references)

<br>

# Understanding Formats Printed in Logs

TensorRT 로그를 살펴보면 아래와 같이 타입과 함께 stride와 vectorization 정보가 함께 출력된다.
```
Half(60,1:8,12,3)
```

`Half`는 element type(`DataType::kHalf`), 즉, 16비트 부동소수점을 나타낸다. 이어서 나오는 `:8`은 이 포맷이 두 번째 축을 따라 벡터당 8개의 요소가 패킹되었다는 것을 나타낸다. 나머지 숫자들은 stride를 의미한다. 위 로그에서 나타내는 텐서의 좌표 `(n,c,h,w)`는 아래의 주소로 매핑된다.
```
((half*)base_address) + (60*n + 1*floor(c/8) + 12*h + 3*w) * 8 + (c mod 8)

= ((half*)base_address) + (60*n + 12*h + 3*w + 1*floor(c/8)) * 8 + (c mod 8)
```
`1:`은 `NHWC` 포맷에서 공통으로 사용된다. 즉, 위 로그에서 나타내는 텐서의 포맷은 `NHWC` 포맷이다.

`NCHW` 포맷인 아래 예시를 살펴보자.
```
Int8(105,15:4,3,1)
```

위 로그는 element type이 `DataType::kINT8`이라는 것을 나타낸다. `:4`는 벡터 크기가 4라는 것을 의미한다. 이 텐서에서 `(n,c,h,w)` 좌표는 다음의 주소로 매핑된다.
```
(int8_t*)base_address + (105*n + 15*floor(c/4) + 3*h + w) * 4 + (c mod 4)
```

스칼라 포맛은 벡터 크기가 1이며, `:1`는 생략하여 출력한다.

일반적으로 좌표와 매핑되는 주소는 아래의 형태를 갖는다.
```
(type*)base_address + (vec_coordinate ⋅ strides) * vec_size + vec_mod
```

위 표기에서 의미하는 바는 다음과 같다.

- dot(⋅)은 inner product를 의미한다.
- strides는 로그에 출력되는 strides이다.
- `vec_size`는 벡터 당 elements의 갯수이다.
- `vec_coordinate`는 벡터화된 축을 따라 좌표를 `vec_size`로 나눈 original 좌표이다.
- `vec_mod`는 벡터화된 축을 따라 `vec_size`로 모듈러 연산을 한 original 좌표이다.

<br>

# References

- [NVIDIA TensorRT Documentation: Understanding Formats Printed in Logs](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#format-printed-logs)