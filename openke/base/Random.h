#ifndef RANDOM_H
#define RANDOM_H
#include "Setting.h"
#include <cstdlib>

// the random seeds for all threads.
unsigned long long *next_random;

// reset the random seeds for all threads
extern "C"
void randReset() {
    //calloc: 在内存的动态存储区中分配workThreads个长度为size的连续空间，函数返回一个指向分配起始地址的指针；如果分配不成功，返回NULL。
	next_random = (unsigned long long *)calloc(workThreads, sizeof(unsigned long long));
	for (INT i = 0; i < workThreads; i++)
		next_random[i] = rand();
}

// get a random interger for the id-th thread with the corresponding random seed.
unsigned long long randd(INT id) {
	next_random[id] = next_random[id] * (unsigned long long)(25214903917) + 11;
	return next_random[id];
}

// get a random interger from the range [0,x) for the id-th thread.
INT rand_max(INT id, INT x) {
	INT res = randd(id) % x;
	while (res < 0)
		res += x;
	return res;
}

// get a random interger from the range [a,b) for the id-th thread.
INT rand(INT a, INT b){
	return (rand() % (b-a))+ a;
}
#endif
