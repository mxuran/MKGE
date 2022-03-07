#include "Setting.h"
#include "Random.h"
#include "Reader.h"
#include "Corrupt.h"
#include "Test.h"
#include <cstdlib>
#include <pthread.h>

extern "C"
void setInPath(char *path);

extern "C"
void setTrainPath(char *path);

extern "C"
void setValidPath(char *path);

extern "C"
void setTestPath(char *path);

extern "C"
void setEntPath(char *path);

extern "C"
void setRelPath(char *path);

extern "C"
void setOutPath(char *path);

extern "C"
void setWorkThreads(INT threads);

extern "C"
void setBern(INT con);

extern "C"
INT getWorkThreads();

extern "C"
INT getEntityTotal();

extern "C"
INT getRelationTotal();

extern "C"
INT getTripleTotal();

extern "C"
INT getTrainTotal();

extern "C"
INT getTestTotal();

extern "C"
INT getValidTotal();

extern "C"
void randReset();

extern "C"
void importTrainFiles();

struct Parameter {
	INT id;
	INT *batch_h;
	INT *batch_t;
	INT *batch_r;
	REAL *batch_y;
	INT batchSize;
	INT negRate;
	INT negRelRate;
	bool p;
	bool val_loss;
	INT mode;
	bool filter_flag;
};

void* getBatch(void* con) {
	Parameter *para = (Parameter *)(con);       //将参数con赋值给para，也就是将sampling函数中的para和threads
	//将para相应的值存到对应的局部变量中
	INT id = para -> id;
	INT *batch_h = para -> batch_h;
	INT *batch_t = para -> batch_t;
	INT *batch_r = para -> batch_r;
	REAL *batch_y = para -> batch_y;
	INT batchSize = para -> batchSize;
	INT negRate = para -> negRate;
	INT negRelRate = para -> negRelRate;
	bool p = para -> p;
	bool val_loss = para -> val_loss;
	INT mode = para -> mode;
	bool filter_flag = para -> filter_flag;
	INT lef, rig;
	if (batchSize % workThreads == 0) {                //如果batchSize刚好能被线程数整除，也就是一个batch的大小刚好能被均分到每一个线程
		lef = id * (batchSize / workThreads);
		rig = (id + 1) * (batchSize / workThreads);
	} else {                                           //否则
		lef = id * (batchSize / workThreads + 1);
		rig = (id + 1) * (batchSize / workThreads + 1);
		if (rig > batchSize) rig = batchSize;
	}
	REAL prob = 500;
	if (val_loss == false) {    //采样负例三元组
		for (INT batch = lef; batch < rig; batch++) {
		    //根据进程ID，随机采样训练三元组
			INT i = rand_max(id, trainTotal);
			batch_h[batch] = trainList[i].h;
			batch_t[batch] = trainList[i].t;
			batch_r[batch] = trainList[i].r;
			batch_y[batch] = 1;
			INT last = batchSize;
			for (INT times = 0; times < negRate; times ++) {
				if (mode == 0){
					if (bernFlag)
						prob = 1000 * right_mean[trainList[i].r] / (right_mean[trainList[i].r] + left_mean[trainList[i].r]);
					if (randd(id) % 1000 < prob) {
						batch_h[batch + last] = trainList[i].h;
						batch_t[batch + last] = corrupt_head(id, trainList[i].h, trainList[i].r);   //根据头实体、关系，通过二分搜索，获取交换后（错误）的尾实体
						batch_r[batch + last] = trainList[i].r;
					} else {
						batch_h[batch + last] = corrupt_tail(id, trainList[i].t, trainList[i].r);   //根据尾实体、关系，通过二分搜索，获取交换后（错误）的头实体
						batch_t[batch + last] = trainList[i].t;
						batch_r[batch + last] = trainList[i].r;
					}
					batch_y[batch + last] = -1;
					last += batchSize;
				} else {
					if(mode == -1){
						batch_h[batch + last] = corrupt_tail(id, trainList[i].t, trainList[i].r);
						batch_t[batch + last] = trainList[i].t;
						batch_r[batch + last] = trainList[i].r;
					} else {
						batch_h[batch + last] = trainList[i].h;
						batch_t[batch + last] = corrupt_head(id, trainList[i].h, trainList[i].r);
						batch_r[batch + last] = trainList[i].r;
					}
					batch_y[batch + last] = -1;
					last += batchSize;
				}
			}
			for (INT times = 0; times < negRelRate; times++) {
				batch_h[batch + last] = trainList[i].h;
				batch_t[batch + last] = trainList[i].t;
				batch_r[batch + last] = corrupt_rel(id, trainList[i].h, trainList[i].t, trainList[i].r, p); //根据头实体、尾实体，通过二分搜索，获取交换后（错误）的关系
				batch_y[batch + last] = -1;
				last += batchSize;
			}
		}
	}
	else
	{   //验证集
		for (INT batch = lef; batch < rig; batch++)
		{
			batch_h[batch] = validList[batch].h;
			batch_t[batch] = validList[batch].t;
			batch_r[batch] = validList[batch].r;
			batch_y[batch] = 1;
		}
	}
	pthread_exit(NULL);     //线程终止
}

extern "C"
void sampling(
		INT *batch_h, 
		INT *batch_t, 
		INT *batch_r, 
		REAL *batch_y, 
		INT batchSize, 
		INT negRate = 1, 
		INT negRelRate = 0, 
		INT mode = 0,
		bool filter_flag = true,
		bool p = false, 
		bool val_loss = false
) {
	pthread_t *pt = (pthread_t *)malloc(workThreads * sizeof(pthread_t));      //根据线程数量，向内存分配指定的大小
	Parameter *para = (Parameter *)malloc(workThreads * sizeof(Parameter));     //根据线程数量，以及Parameter结构体的大小，向内存分配指定的大小
	for (INT threads = 0; threads < workThreads; threads++) {
		para[threads].id = threads;
		para[threads].batch_h = batch_h;
		para[threads].batch_t = batch_t;
		para[threads].batch_r = batch_r;
		para[threads].batch_y = batch_y;
		para[threads].batchSize = batchSize;
		para[threads].negRate = negRate;
		para[threads].negRelRate = negRelRate;
		para[threads].p = p;
		para[threads].val_loss = val_loss;
		para[threads].mode = mode;
		para[threads].filter_flag = filter_flag;
		/*
		    创建线程
            int pthread_create(
                 pthread_t *restrict tidp,   //新创建的线程ID指向的内存单元。
                 const pthread_attr_t *restrict attr,  //线程属性，默认为NULL
                 void *(*start_rtn)(void *), //新创建的线程从start_rtn函数的地址开始运行
                 void *restrict arg //默认为NULL。若上述函数需要参数，将参数放入结构中并将地址作为arg传入。
                  );
        */

		pthread_create(&pt[threads], NULL, getBatch, (void*)(para+threads));
	}

	/*
	int pthread_join( pthread_t  thread, void * * value_ptr );
    函数pthread_join的作用是，等待一个线程终止。
    调用pthread_join的线程将被挂起直到参数thread所代表的线程终止时为止。pthread_join是一个线程阻塞函数，调用它的函数将一直等到被等待的线程结束为止。
    如果value_ptr不为NULL，那么线程thread的返回值存储在该指针指向的位置。该返回值可以是由pthread_exit给出的值，或者该线程被取消而返回PTHREAD_CANCELED。
	*/

	for (INT threads = 0; threads < workThreads; threads++)
		pthread_join(pt[threads], NULL);

	free(pt);       //将通过malloc分配pt、para的内存释
	free(para);
}

int main() {
	importTrainFiles();
	return 0;
}