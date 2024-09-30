/* Copyright (c) 2019-2020, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */
#pragma once
#include <cuda.h>
#include <vector>
#include <cassert>

namespace cuvector
{

// vector wrapper class
template<typename T, class Allocator>
class Vector {
public:
    Vector(CUcontext ctx) : m_allocator{ctx} {}
    
    CUresult reserve(size_t num) {
        return m_allocator.reserve(num * sizeof(T));
    }
    CUresult grow(size_t num) {
        return m_allocator.grow(num * sizeof(T));
    }
    T* getPointer() const {
        return (T*)m_allocator.getPointer();
    }
    size_t getSize() const {
        return m_allocator.getSize();
    }

private:
    Allocator m_allocator;
};

// vector allocator using device memory allocation
class VectorMemAlloc {
public:
    VectorMemAlloc(CUcontext context) : m_ctx{context}, m_dev_ptr{}, m_alloc_sz{} {}
    ~VectorMemAlloc() {
        cuMemFree(m_dev_ptr);
    }

    CUresult reserve(size_t new_sz) {
        CUresult status = CUDA_SUCCESS;
        CUdeviceptr new_ptr{};
        CUcontext prev_ctx;

        if (new_sz <= m_alloc_sz) {
            return status;
        }
        cuCtxGetCurrent(&prev_ctx);
        // make sure to allcoate on the correct context
        if ((status = cuCtxSetCurrent(m_ctx)) != CUDA_SUCCESS) {
            return status;
        }
        // allocate the bigger buffer
        if ((status = cuMemAlloc(&new_ptr, new_sz)) == CUDA_SUCCESS) {
            if ((status = cuMemcpyAsync(new_ptr, m_dev_ptr, m_alloc_sz, CU_STREAM_PER_THREAD)) == CUDA_SUCCESS) {
                // free the smaller buffer.
                // we don't need to synchronize CU_STREAM_PER_THREAD, since cuMemFree synchronizes for us.
                cuMemFree(m_dev_ptr);
                m_dev_ptr = new_ptr;
                m_alloc_sz = new_sz;
            }
            else {
                // failed to copy, free the new one
                cuMemFree(new_ptr);
            }
        }
        // make sure to alywas return to the previous context the caller had
        status = cuCtxSetCurrent(prev_ctx);

        return status;
    }
    CUresult grow(size_t new_sz) {
        return reserve(new_sz);
    }
    CUdeviceptr getPointer() const {
        return m_dev_ptr;
    }
    size_t getSize() const {
        return m_alloc_sz;
    }

private:
    CUcontext m_ctx;
    CUdeviceptr m_dev_ptr;
    size_t m_alloc_sz;
};

// vector allocator using managed memory allocation
class VectorMemAllocManaged {
public:
    VectorMemAllocManaged(CUcontext context): m_ctx{context}, m_dev_ptr{}, m_alloc_sz{}, m_reserve_sz{}, m_dev{CU_DEVICE_INVALID}
    {
        CUcontext prev_ctx;
        cuCtxGetCurrent(&prev_ctx);
        if (cuCtxSetCurrent(context) == CUDA_SUCCESS) {
            cuCtxGetDevice(&m_dev);
            cuCtxSetCurrent(prev_ctx);
        }
        cuDeviceGetAttribute(&m_supportsConcurrentManagedAccess, CU_DEVICE_ATTRIBUTE_CONCURRENT_MANAGED_ACCESS, m_dev);
    }
    ~VectorMemAllocManaged() {
        cuMemFree(m_dev_ptr);
    }

    CUresult reserve(size_t new_sz) {
        CUresult status = CUDA_SUCCESS;
        CUcontext prev_cnt;
        CUdeviceptr new_ptr{};

        if (new_sz <= m_reserve_sz) {
            return CUDA_SUCCESS;
        }
        cuCtxGetCurrent(&prev_cnt);
        if ((status = cuCtxSetCurrent(m_ctx)) != CUDA_SUCCESS) {
            return status;
        }
        // allocate the bigger buffer
        if ((status = cuMemAllocManaged(&new_ptr, new_sz, CU_MEM_ATTACH_GLOBAL)) == CUDA_SUCCESS) {
            cuMemAdvise(new_ptr, new_sz, CU_MEM_ADVISE_SET_PREFERRED_LOCATION, m_dev);
            if (m_alloc_sz > 0) {
                if ((status = cuMemcpyAsync(new_ptr, m_dev_ptr, m_alloc_sz, CU_STREAM_PER_THREAD)) == CUDA_SUCCESS) {
                    cuMemFree(m_dev_ptr);
                }
                else {
                    cuMemFree(new_ptr);
                }
            }
        }
        if (status == CUDA_SUCCESS) {
            m_dev_ptr = new_ptr;
            m_reserve_sz = new_sz;
        }

        return status;
    }
    CUresult grow(size_t new_sz) {
        CUresult status = CUDA_SUCCESS;
        CUcontext prev_ctx;

        if (new_sz <= m_alloc_sz) {
            return status;
        }
        if ((status = reserve(new_sz)) != CUDA_SUCCESS) {
            return status;
        }
        cuCtxGetCurrent(&prev_ctx);
        if ((status = cuCtxSetCurrent(m_ctx)) != CUDA_SUCCESS) {
            return status;
        }
        // commit the needed memory
        if (m_supportsConcurrentManagedAccess && 
            (status = cuMemPrefetchAsync(m_dev_ptr + m_alloc_sz, (new_sz - m_alloc_sz), m_dev, CU_STREAM_PER_THREAD)) == CUDA_SUCCESS) {
            if ((status = cuStreamSynchronize(CU_STREAM_PER_THREAD)) == CUDA_SUCCESS) {
                m_alloc_sz = new_sz;
            }
        }
        status = cuCtxSetCurrent(prev_ctx);

        return status;
    }
    CUdeviceptr getPointer() const {
        return m_dev_ptr;
    }
    size_t getSize() const {
        return m_alloc_sz;
    }

private:
    CUcontext m_ctx;
    CUdeviceptr m_dev_ptr;
    size_t m_alloc_sz;
    size_t m_reserve_sz;
    CUdevice m_dev;
    int m_supportsConcurrentManagedAccess;
};

// vector allocator using virtual memory
class VectorMemMap {
public:
    VectorMemMap(CUcontext context) : m_dev_ptr{}, m_prop{}, m_handles{}, m_alloc_sz{}, m_reserve_sz{}, m_chunk_sz{}
    {
        CUdevice dev;
        CUcontext prev_ctx;
        CUresult status = CUDA_SUCCESS;

        status = cuCtxGetCurrent(&prev_ctx);
        assert(status == CUDA_SUCCESS);
        if (cuCtxSetCurrent(context) == CUDA_SUCCESS) {
            status = cuCtxGetDevice(&dev);
            assert(status == CUDA_SUCCESS);
            status = cuCtxSetCurrent(prev_ctx);
            assert(status == CUDA_SUCCESS);
        }
        
        m_prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
        m_prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
        m_prop.location.id = static_cast<int>(dev);
        m_prop.win32HandleMetaData = NULL;

        m_access_desc.location = m_prop.location;
        m_access_desc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;

        status = cuMemGetAllocationGranularity(&m_chunk_sz, &m_prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM);
        assert(status == CUDA_SUCCESS);
    }
    ~VectorMemMap() {
        auto status = CUDA_SUCCESS;
        if (m_dev_ptr != 0ULL) {
            status = cuMemUnmap(m_dev_ptr, m_alloc_sz);
            assert(status == CUDA_SUCCESS);
            for (auto const& va_range : m_va_ranges) {
                status = cuMemAddressFree(va_range.start, va_range.sz);
                assert(status == CUDA_SUCCESS);
            }
            for (auto& handle : m_handles) {
                status = cuMemRelease(handle);
                assert(status == CUDA_SUCCESS);
            }
        }
    }

    CUresult reserve(size_t new_sz) {
        auto status = CUDA_SUCCESS;
        CUdeviceptr new_ptr{};

        if (new_sz <= m_reserve_sz) {
            return status;
        }

        size_t const aligned_sz = ((new_sz + m_chunk_sz - 1) / m_chunk_sz) * m_chunk_sz;
        // try to reserve at the end of old pointer(m_dev_ptr)
        status = cuMemAddressReserve(&new_ptr, (aligned_sz - m_reserve_sz), 0ULL, m_dev_ptr + m_reserve_sz, 0ULL);
        if (status != CUDA_SUCCESS || (new_ptr != m_dev_ptr + m_reserve_sz)) {
            // something went wrong.
            if (new_ptr != 0ULL) {
                cuMemAddressFree(new_ptr, (aligned_sz - m_reserve_sz));
            }
            // slow path - try to find a new address reservation big enough
            status = cuMemAddressReserve(&new_ptr, aligned_sz, 0ULL, 0U, 0);
            if (status == CUDA_SUCCESS && m_dev_ptr != 0ULL) {
                CUdeviceptr ptr = new_ptr;
                // found one, now unamp the previous allocations
                status = cuMemUnmap(m_dev_ptr, m_alloc_sz);
                assert(status == CUDA_SUCCESS);
                for (size_t i = 0; i < m_handles.size(); i++) {
                    size_t const handle_sz = m_handle_sizes[i];
                    // and remap them, enabling their access
                    if ((status = cuMemMap(ptr, handle_sz, 0ULL, m_handles[i], 0ULL)) != CUDA_SUCCESS) break;
                    if ((status = cuMemSetAccess(ptr, handle_sz, &m_access_desc, 1ULL)) != CUDA_SUCCESS) break;
                    ptr += handle_sz;
                }
                if (status != CUDA_SUCCESS) {
                    // failed the mapping.. clean up
                    status = cuMemUnmap(new_ptr, aligned_sz);
                    assert(status == CUDA_SUCCESS);
                    status = cuMemAddressFree(new_ptr, aligned_sz);
                    assert(status == CUDA_SUCCESS);
                }
                else {
                    // clean up old VA reservations
                    for (auto const& va_range : m_va_ranges) {
                        cuMemAddressFree(va_range.start, va_range.sz);
                    }
                    m_va_ranges.clear();
                }
            }
            // assuming everything went well, update everything
            if (status == CUDA_SUCCESS) {
                Range r;
                m_dev_ptr = new_ptr;
                m_reserve_sz = aligned_sz;
                r.start = new_ptr;
                r.sz = aligned_sz;
                m_va_ranges.push_back(r);
            }
        }
        else {
            Range r;
            r.start = new_ptr;
            r.sz = aligned_sz - m_reserve_sz;
            m_va_ranges.push_back(r);
            if (m_dev_ptr == 0ULL) {
                m_dev_ptr = new_ptr;
            }
            m_reserve_sz = aligned_sz;
        }

        return status;
    }
    CUresult grow(size_t new_sz) {
        auto status = CUDA_SUCCESS;
        CUmemGenericAllocationHandle handle;

        if (new_sz <= m_alloc_sz) {
            return status;
        }

        size_t const size_diff = new_sz - m_alloc_sz;
        // round up to the next chunk size
        size_t const sz = ((size_diff + m_chunk_sz - 1) / m_chunk_sz) * m_chunk_sz;

        if ((status = reserve(m_alloc_sz + sz)) != CUDA_SUCCESS) {
            return status;
        }

        if ((status = cuMemCreate(&handle, sz, &m_prop, 0)) == CUDA_SUCCESS) {
            if ((status = cuMemMap(m_dev_ptr + m_alloc_sz, sz, 0ULL, handle, 0ULL)) == CUDA_SUCCESS) {
                if ((status = cuMemSetAccess(m_dev_ptr + m_alloc_sz, sz, &m_access_desc, 1ULL)) == CUDA_SUCCESS) {
                    m_handles.push_back(handle);
                    m_handle_sizes.push_back(sz);
                    m_alloc_sz += sz;
                }
                if (status != CUDA_SUCCESS) {
                    cuMemUnmap(m_dev_ptr + m_alloc_sz, sz);
                }
            }
            if (status != CUDA_SUCCESS) {
                cuMemRelease(handle);
            }
        }

        return status;
    }
    CUdeviceptr getPointer() const {
        return m_dev_ptr;
    }
    size_t getSize() const {
        return m_alloc_sz;
    }

private:
    CUdeviceptr m_dev_ptr;
    CUmemAllocationProp m_prop;
    CUmemAccessDesc m_access_desc;
    struct Range {
        CUdeviceptr start;
        size_t sz;
    };
    std::vector<Range> m_va_ranges;
    std::vector<CUmemGenericAllocationHandle> m_handles;
    std::vector<size_t> m_handle_sizes;
    size_t m_alloc_sz;
    size_t m_reserve_sz;
    size_t m_chunk_sz;
};

} // namespace cuvector