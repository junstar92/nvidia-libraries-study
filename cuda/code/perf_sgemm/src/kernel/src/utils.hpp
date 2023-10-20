#pragma once

inline int cdiv(int const x, int const div)
{
    return static_cast<int>((x + div - 1) / div);
}