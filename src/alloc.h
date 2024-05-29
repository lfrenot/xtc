#ifndef _RUNTIME_ALLOC_H
#define _RUNTIME_ALLOC_H

#include <stdint.h>

extern void *CHostMemoryAllocate(size_t size);
extern void *CHostMemoryAllocateAligned(size_t size, size_t alignment);
extern void CHostMemoryFree(void *ptr);
#endif
