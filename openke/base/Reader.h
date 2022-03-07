#ifndef READER_H
#define READER_H
#include "Setting.h"
#include "Triple.h"
#include <cstdlib>
#include <algorithm>
#include <iostream>
#include <cmath>

INT *freqRel, *freqEnt;
INT *lefHead, *rigHead;
INT *lefTail, *rigTail;
INT *lefRel, *rigRel;
REAL *left_mean, *right_mean;
REAL *prob;

Triple *trainList;
Triple *trainHead;
Triple *trainTail;
Triple *trainRel;

INT *testLef, *testRig;
INT *validLef, *validRig;

extern "C"
void importProb(REAL temp){
    if (prob != NULL)
        free(prob);
    FILE *fin;
    fin = fopen((inPath + "kl_prob.txt").c_str(), "r");
    printf("Current temperature:%f\n", temp);
    prob = (REAL *)calloc(relationTotal * (relationTotal - 1), sizeof(REAL));
    INT tmp;
    for (INT i = 0; i < relationTotal * (relationTotal - 1); ++i){
        tmp = fscanf(fin, "%f", &prob[i]);
    }
    REAL sum = 0.0;
    for (INT i = 0; i < relationTotal; ++i) {
        for (INT j = 0; j < relationTotal-1; ++j){
            REAL tmp = exp(-prob[i * (relationTotal - 1) + j] / temp);
            sum += tmp;
            prob[i * (relationTotal - 1) + j] = tmp;
        }
        for (INT j = 0; j < relationTotal-1; ++j){
            prob[i*(relationTotal-1)+j] /= sum;
        }
        sum = 0;
    }
    fclose(fin);
}

extern "C"
void importTrainFiles() {

	printf("The toolkit is importing datasets.\n");
	FILE *fin;
	int tmp;

    //读取关系数据集
    if (rel_file == "")
	    fin = fopen((inPath + "relation2id.txt").c_str(), "r");
    else
        fin = fopen(rel_file.c_str(), "r");
	tmp = fscanf(fin, "%ld", &relationTotal);
	printf("The total of relations is %ld.\n", relationTotal);
	fclose(fin);

    //读取实体数据集
    if (ent_file == "")
        fin = fopen((inPath + "entity2id.txt").c_str(), "r");
    else
        fin = fopen(ent_file.c_str(), "r");
	tmp = fscanf(fin, "%ld", &entityTotal);
	printf("The total of entities is %ld.\n", entityTotal);
	fclose(fin);

    //读取训练数据集，三元组，头实体ID，尾实体ID，关系ID
    if (train_file == "")
        fin = fopen((inPath + "train2id.txt").c_str(), "r");
    else
        fin = fopen(train_file.c_str(), "r");
	tmp = fscanf(fin, "%ld", &trainTotal);

	//根据训练集的大小，内存分配对应的空间大小给trainList、trainHead、trainTail、trainRel
	trainList = (Triple *)calloc(trainTotal, sizeof(Triple));
	trainHead = (Triple *)calloc(trainTotal, sizeof(Triple));
	trainTail = (Triple *)calloc(trainTotal, sizeof(Triple));
	trainRel = (Triple *)calloc(trainTotal, sizeof(Triple));

	//根据关系总数，分配内存给freqRel，freqRel表示关系的频率
	freqRel = (INT *)calloc(relationTotal, sizeof(INT));
	//根据实体总数，分配内存给freqEnt，freqEnt表示实体的频率
	freqEnt = (INT *)calloc(entityTotal, sizeof(INT));
	for (INT i = 0; i < trainTotal; i++) {
	    //将train2id.txt中的三列数据，分别保存到trainList中
		tmp = fscanf(fin, "%ld", &trainList[i].h);
		tmp = fscanf(fin, "%ld", &trainList[i].t);
		tmp = fscanf(fin, "%ld", &trainList[i].r);
	}
	fclose(fin);

	//按照头实体ID的大小，对trainList进行排序，若头实体ID相等，则判断关系ID；若头实体、关系都相等，则判断尾实体ID；并以升序的方式排列
	std::sort(trainList, trainList + trainTotal, Triple::cmp_head);

	tmp = trainTotal; trainTotal = 1;
	trainHead[0] = trainTail[0] = trainRel[0] = trainList[0];
	freqEnt[trainList[0].t] += 1;   //以trainList[0]的尾实体作为数组freqEnt的下标，对应的值+1
	freqEnt[trainList[0].h] += 1;   //以trainList[0]的头实体作为数组freqEnt的下标，对应的值+1
	freqRel[trainList[0].r] += 1;   //以trainList[0]的关系作为数组freqEnt的下标，对应的值+1

	//从i=1到train2id.txt中总的训练行数，遍历trainList
	for (INT i = 1; i < tmp; i++)
	    //如果第i的一个的头实体不与i-1的头实体相等，或者i的关系不与i-1对应的关系相等，或者i的尾实体不与i-1的尾实体相等
	    //即，第i的一个训练三元组不与第i-1的训练三元组相同
		if (trainList[i].h != trainList[i - 1].h || trainList[i].r != trainList[i - 1].r || trainList[i].t != trainList[i - 1].t) {
             //排除相邻且相同的三元组后，剩下不重复的训练三元组
			trainHead[trainTotal] = trainTail[trainTotal] = trainRel[trainTotal] = trainList[trainTotal] = trainList[i];
			trainTotal++;
			freqEnt[trainList[i].t]++;
			freqEnt[trainList[i].h]++;
			freqRel[trainList[i].r]++;
		}

    //按照头实体的大小，对trainHead进行排序，以升序的方式，若头实体ID相等，则判断关系ID；若头实体、关系都相等，则判断尾实体ID；
	std::sort(trainHead, trainHead + trainTotal, Triple::cmp_head);
	//按照尾实体的大小，对trainTail进行排序，以升序的方式，若尾实体ID相等，则判断关系ID；若尾实体、关系都相等，则判断尾实体ID；
	std::sort(trainTail, trainTail + trainTotal, Triple::cmp_tail);
	//按照头实体的大小，对trainRel进行排序，以升序的方式，若头实体ID相等，则判断尾实体；若头实体、尾实体都相等，则判断关系ID；
	std::sort(trainRel, trainRel + trainTotal, Triple::cmp_rel);

	printf("The total of train triples is %ld.\n", trainTotal);

    //以实体总数，分配内存空间给lefHead、lefHead、lefTail、rigTail、lefRel、rigRel
	lefHead = (INT *)calloc(entityTotal, sizeof(INT));
	rigHead = (INT *)calloc(entityTotal, sizeof(INT));
	lefTail = (INT *)calloc(entityTotal, sizeof(INT));
	rigTail = (INT *)calloc(entityTotal, sizeof(INT));
	lefRel = (INT *)calloc(entityTotal, sizeof(INT));
	rigRel = (INT *)calloc(entityTotal, sizeof(INT));

	//对数组rigHead、rigTail、rigRel初始化为-1
	memset(rigHead, -1, sizeof(INT)*entityTotal);
	memset(rigTail, -1, sizeof(INT)*entityTotal);
	memset(rigRel, -1, sizeof(INT)*entityTotal);

	//从i=1，到trainTotal
    //ritTail保存的是尾实体ID较小的对应的trainT下标
    //lefTail保存的是尾实体ID较大的对应的trainT下标
    //rigHead、lefHead、rigRel、lefRel同理

	for (INT i = 1; i < trainTotal; i++) {
	    //如果trainTail，第i中的尾实体与i-1中的尾实体不一样
	    //即，如果相邻两个训练尾实体不相同，则以前者尾实体为rigTail的下标，将i-1替换对应的-1
	    // 将后者尾实体为lefTail的下标，将i替换对应的-1
		if (trainTail[i].t != trainTail[i - 1].t) {
			rigTail[trainTail[i - 1].t] = i - 1;    //将i-1赋值给以trainTail[i-1]的尾实体为下标，对应的rigTail值-1
			lefTail[trainTail[i].t] = i;    //将i赋值给以trainTail[i]的尾实体为下标，对应的lefTail值-1
		}
		if (trainHead[i].h != trainHead[i - 1].h) {
			rigHead[trainHead[i - 1].h] = i - 1;
			lefHead[trainHead[i].h] = i;
		}
		if (trainRel[i].h != trainRel[i - 1].h) {
			rigRel[trainRel[i - 1].h] = i - 1;
			lefRel[trainRel[i].h] = i;
		}
	}
	//将以0作为下标的值赋值为0，以及以训练集的最后一位作为下标，赋值为trainTotal-1
	lefHead[trainHead[0].h] = 0;
	rigHead[trainHead[trainTotal - 1].h] = trainTotal - 1;
	lefTail[trainTail[0].t] = 0;
	rigTail[trainTail[trainTotal - 1].t] = trainTotal - 1;
	lefRel[trainRel[0].h] = 0;
	rigRel[trainRel[trainTotal - 1].h] = trainTotal - 1;

    //为left_mean、right_mean分配实数型的内存，元素个数为relationTotal，大小为REAL
	left_mean = (REAL *)calloc(relationTotal,sizeof(REAL));
	right_mean = (REAL *)calloc(relationTotal,sizeof(REAL));
	for (INT i = 0; i < entityTotal; i++) {
		for (INT j = lefHead[i] + 1; j <= rigHead[i]; j++)
			if (trainHead[j].r != trainHead[j - 1].r)
				left_mean[trainHead[j].r] += 1.0;        //相邻训练头实体对应的关系不等情况下，对头实体的出边+1
		if (lefHead[i] <= rigHead[i])
			left_mean[trainHead[lefHead[i]].r] += 1.0;         //如果左实体的大小小于等于右实体的大小，则以左实体对应的出边+1
		for (INT j = lefTail[i] + 1; j <= rigTail[i]; j++)
			if (trainTail[j].r != trainTail[j - 1].r)
				right_mean[trainTail[j].r] += 1.0;
		if (lefTail[i] <= rigTail[i])
			right_mean[trainTail[lefTail[i]].r] += 1.0;
	}
	for (INT i = 0; i < relationTotal; i++) {
		left_mean[i] = freqRel[i] / left_mean[i];         //实体的个数除以对应实体的出边
		right_mean[i] = freqRel[i] / right_mean[i];         //实体的个数除以对应实体的入边
	}
}

Triple *testList;
Triple *validList;
Triple *tripleList;

extern "C"
void importTestFiles(char *test_file) {

    INT len = strlen(test_file);
    std::string TestF = "";
    for (INT i = 0; i < len; i++)
        TestF = TestF + test_file[i];
    printf("Test File Path : %s\n", TestF.c_str());

    FILE *fin;
    INT tmp;
    
    if (rel_file == "")
	    fin = fopen((inPath + "relation2id.txt").c_str(), "r");
    else
        fin = fopen(rel_file.c_str(), "r");
    tmp = fscanf(fin, "%ld", &relationTotal);
    fclose(fin);

    if (ent_file == "")
        fin = fopen((inPath + "entity2id.txt").c_str(), "r");
    else
        fin = fopen(ent_file.c_str(), "r");
    tmp = fscanf(fin, "%ld", &entityTotal);
    fclose(fin);

    FILE* f_kb1, * f_kb2, * f_kb3;
    if (train_file == "")
        f_kb2 = fopen((inPath + "train2id.txt").c_str(), "r");
    else
        f_kb2 = fopen(train_file.c_str(), "r");

    if (test_file == "")
        f_kb1 = fopen((inPath + "test2id.txt").c_str(), "r");
    else
        f_kb1 = fopen((inPath + TestF).c_str(), "r");


    if (valid_file == "")
        f_kb3 = fopen((inPath + "valid2id.txt").c_str(), "r");
    else
        f_kb3 = fopen(valid_file.c_str(), "r");

    tmp = fscanf(f_kb1, "%ld", &testTotal);
    tmp = fscanf(f_kb2, "%ld", &trainTotal);
    tmp = fscanf(f_kb3, "%ld", &validTotal);
    tripleTotal = testTotal + trainTotal + validTotal;
    testList = (Triple *)calloc(testTotal, sizeof(Triple));
    validList = (Triple *)calloc(validTotal, sizeof(Triple));
    tripleList = (Triple *)calloc(tripleTotal, sizeof(Triple));
    for (INT i = 0; i < testTotal; i++) {
        tmp = fscanf(f_kb1, "%ld", &testList[i].h);
        tmp = fscanf(f_kb1, "%ld", &testList[i].t);
        tmp = fscanf(f_kb1, "%ld", &testList[i].r);
        tripleList[i] = testList[i];
    }
    for (INT i = 0; i < trainTotal; i++) {
        tmp = fscanf(f_kb2, "%ld", &tripleList[i + testTotal].h);
        tmp = fscanf(f_kb2, "%ld", &tripleList[i + testTotal].t);
        tmp = fscanf(f_kb2, "%ld", &tripleList[i + testTotal].r);
    }
    for (INT i = 0; i < validTotal; i++) {
        tmp = fscanf(f_kb3, "%ld", &tripleList[i + testTotal + trainTotal].h);
        tmp = fscanf(f_kb3, "%ld", &tripleList[i + testTotal + trainTotal].t);
        tmp = fscanf(f_kb3, "%ld", &tripleList[i + testTotal + trainTotal].r);
        validList[i] = tripleList[i + testTotal + trainTotal];
    }
    fclose(f_kb1);
    fclose(f_kb2);
    fclose(f_kb3);

    std::sort(tripleList, tripleList + tripleTotal, Triple::cmp_head);
    std::sort(testList, testList + testTotal, Triple::cmp_rel2);
    std::sort(validList, validList + validTotal, Triple::cmp_rel2);
    printf("The total of test triples is %ld.\n", testTotal);
    printf("The total of valid triples is %ld.\n", validTotal);

    testLef = (INT *)calloc(relationTotal, sizeof(INT));
    testRig = (INT *)calloc(relationTotal, sizeof(INT));
    memset(testLef, -1, sizeof(INT) * relationTotal);
    memset(testRig, -1, sizeof(INT) * relationTotal);
    for (INT i = 1; i < testTotal; i++) {
	if (testList[i].r != testList[i-1].r) {
	    testRig[testList[i-1].r] = i - 1;
	    testLef[testList[i].r] = i;
	}
    }
    testLef[testList[0].r] = 0;
    testRig[testList[testTotal - 1].r] = testTotal - 1;

    validLef = (INT *)calloc(relationTotal, sizeof(INT));
    validRig = (INT *)calloc(relationTotal, sizeof(INT));
    memset(validLef, -1, sizeof(INT)*relationTotal);
    memset(validRig, -1, sizeof(INT)*relationTotal);
    for (INT i = 1; i < validTotal; i++) {
	if (validList[i].r != validList[i-1].r) {
	    validRig[validList[i-1].r] = i - 1;
	    validLef[validList[i].r] = i;
	}
    }
    validLef[validList[0].r] = 0;
    validRig[validList[validTotal - 1].r] = validTotal - 1;
}


INT* head_lef;
INT* head_rig;
INT* tail_lef;
INT* tail_rig;
INT* head_type;
INT* tail_type;

extern "C"
void importTypeFiles() {

    head_lef = (INT *)calloc(relationTotal, sizeof(INT));
    head_rig = (INT *)calloc(relationTotal, sizeof(INT));
    tail_lef = (INT *)calloc(relationTotal, sizeof(INT));
    tail_rig = (INT *)calloc(relationTotal, sizeof(INT));
    INT total_lef = 0;
    INT total_rig = 0;
    FILE* f_type = fopen((inPath + "type_constrain.txt").c_str(),"r");
    INT tmp;
    tmp = fscanf(f_type, "%ld", &tmp);
    for (INT i = 0; i < relationTotal; i++) {
        INT rel, tot;
        tmp = fscanf(f_type, "%ld %ld", &rel, &tot);
        for (INT j = 0; j < tot; j++) {
            tmp = fscanf(f_type, "%ld", &tmp);
            total_lef++;
        }
        tmp = fscanf(f_type, "%ld%ld", &rel, &tot);
        for (INT j = 0; j < tot; j++) {
            tmp = fscanf(f_type, "%ld", &tmp);
            total_rig++;
        }
    }
    fclose(f_type);
    head_type = (INT *)calloc(total_lef, sizeof(INT)); 
    tail_type = (INT *)calloc(total_rig, sizeof(INT));
    total_lef = 0;
    total_rig = 0;
    f_type = fopen((inPath + "type_constrain.txt").c_str(),"r");
    tmp = fscanf(f_type, "%ld", &tmp);
    for (INT i = 0; i < relationTotal; i++) {
        INT rel, tot;
        tmp = fscanf(f_type, "%ld%ld", &rel, &tot);
        head_lef[rel] = total_lef;
        for (INT j = 0; j < tot; j++) {
            tmp = fscanf(f_type, "%ld", &head_type[total_lef]);
            total_lef++;
        }
        head_rig[rel] = total_lef;
        std::sort(head_type + head_lef[rel], head_type + head_rig[rel]);
        tmp = fscanf(f_type, "%ld%ld", &rel, &tot);
        tail_lef[rel] = total_rig;
        for (INT j = 0; j < tot; j++) {
            tmp = fscanf(f_type, "%ld", &tail_type[total_rig]);
            total_rig++;
        }
        tail_rig[rel] = total_rig;
        std::sort(tail_type + tail_lef[rel], tail_type + tail_rig[rel]);
    }
    fclose(f_type);
}


#endif
